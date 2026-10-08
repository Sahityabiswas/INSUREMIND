"""Portable linear PPO with masked categorical sampling and minibatch updates."""
import os
import numpy as np

def advantages(rewards, values, next_values, terminated, boundaries, gamma, lam):
    result = np.zeros(len(rewards))
    carry = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (not terminated[i]) - values[i]
        carry = delta + gamma * lam * (not boundaries[i]) * carry
        result[i] = carry
    return result, result + np.asarray(values)


class LinearPPO:
    def __init__(self, obs_dim, n_act, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, seed=0):
        self.rng = np.random.RandomState(seed)
        self.W = self.rng.randn(obs_dim, n_act) * 0.01
        self.b = np.zeros(n_act)
        self.vw = self.rng.randn(obs_dim) * 0.01
        self.vb = 0.0
        self.lr, self.gamma, self.lam, self.clip = lr, gamma, lam, clip
        self.history = []
        self.optimizer_m, self.optimizer_v, self.optimizer_step = {}, {}, 0

    def probs(self, o, mask=None):
        z = np.asarray(o) @ self.W + self.b
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if np.any(~np.any(mask, axis=-1)):
                raise ValueError("At least one action must be allowed")
            z = np.where(mask, z, -np.inf)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def act(self, o, mask=None, deterministic=False):
        p = self.probs(o, mask)
        action = int(np.argmax(p) if deterministic else self.rng.choice(len(p), p=p))
        return action, float(np.log(max(p[action], 1e-12)))

    def value(self, o):
        return np.asarray(o) @ self.vw + self.vb

    def update(self, observations, actions, old_logp, returns, adv, masks,
               epochs=4, batch_size=256, entropy_coef=0.01, max_grad_norm=0.5):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        losses = []
        for _ in range(epochs):
            order = self.rng.permutation(len(actions))
            for start in range(0, len(order), batch_size):
                ids = order[start:start + batch_size]
                x, a, m = observations[ids], actions[ids], masks[ids]
                p = self.probs(x, m)
                lp = np.log(np.maximum(p, 1e-12))
                ratio = np.exp(lp[np.arange(len(ids)), a] - old_logp[ids])
                aa = adv[ids]
                active = np.where(aa >= 0, ratio <= 1 + self.clip, ratio >= 1 - self.clip)
                # Gradient of -min(ratio*A, clip(ratio)*A), plus entropy regularization.
                g = p.copy()
                g[np.arange(len(ids)), a] -= 1
                g *= (aa * ratio * active)[:, None]
                entropy = -(p * lp).sum(axis=1)
                g += entropy_coef * p * (lp + entropy[:, None])
                g /= len(ids)
                err = (self.value(x) - returns[ids]) / len(ids)
                grads = [x.T @ g, g.sum(axis=0), x.T @ err, err.sum()]
                magnitude = np.sqrt(sum(np.sum(v * v) for v in grads))
                scale = min(1.0, max_grad_norm / (magnitude + 1e-8))
                self.optimizer_step += 1
                for name, grad in zip(("W", "b", "vw", "vb"), grads):
                    grad = scale * grad
                    m = 0.9 * self.optimizer_m.get(name, np.zeros_like(grad)) + 0.1 * grad
                    v = 0.999 * self.optimizer_v.get(name, np.zeros_like(grad)) + 0.001 * grad ** 2
                    self.optimizer_m[name], self.optimizer_v[name] = m, v
                    adjusted = (m / (1 - 0.9 ** self.optimizer_step)) / (np.sqrt(v / (1 - 0.999 ** self.optimizer_step)) + 1e-5)
                    setattr(self, name, getattr(self, name) - self.lr * adjusted)
                losses.append(float(np.mean((self.value(x) - returns[ids]) ** 2)))
        return float(np.mean(losses))

def train(env, total_steps=20000, seed=42, lr=3e-4, init_W=None,
          gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_steps=1024,
          batch_size=256, n_epochs=4, entropy_coef=0.01, max_grad_norm=0.5):
    if min(total_steps, n_steps, batch_size, n_epochs) <= 0:
        raise ValueError("Training sizes must be positive")
    agent = LinearPPO(env.observation_space.shape[0], env.action_space.n,
                      lr, gamma, gae_lambda, clip_range, seed)
    if init_W is not None:
        w = np.asarray(init_W, dtype=float)
        if w.shape != agent.W.shape:
            raise ValueError("Supervised checkpoint has incompatible state dimensions; retrain it")
        agent.W = w.copy()
    o, info = env.reset(seed=seed)
    ep_rews, ep_total, steps = [], 0.0, 0
    while steps < total_steps:
        obs, acts, logps, rewards, values, next_values, terms, boundaries, masks = ([] for _ in range(9))
        for _ in range(min(n_steps, total_steps - steps)):
            mask = info["action_mask"]
            a, lp = agent.act(o, mask)
            v = float(agent.value(o))
            o2, r, terminated, truncated, info2 = env.step(a)
            obs.append(o); acts.append(a); logps.append(lp); rewards.append(r)
            values.append(v); next_values.append(float(agent.value(o2)))
            terms.append(terminated); boundaries.append(terminated or truncated); masks.append(mask)
            steps += 1
            ep_total += r
            o, info = o2, info2
            if terminated or truncated:
                ep_rews.append(ep_total)
                ep_total = 0.0
                o, info = env.reset(seed=seed + len(ep_rews))
        adv, returns = advantages(rewards, values, next_values, terms, boundaries, gamma, gae_lambda)
        loss = agent.update(np.asarray(obs), np.asarray(acts), np.asarray(logps), returns, adv,
                            np.asarray(masks), n_epochs, batch_size, entropy_coef, max_grad_norm)
        agent.history.append({"steps": steps, "episodes": len(ep_rews),
                              "mean_reward": float(np.mean(ep_rews[-100:])) if ep_rews else ep_total,
                              "value_loss": loss})
    return agent, float(np.mean(ep_rews)) if ep_rews else ep_total

def save(agent, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, W=agent.W, b=agent.b, vw=agent.vw, vb=np.array(agent.vb),
             state_dim=np.array(agent.W.shape[0]), version=np.array(2))

def load(path, obs_dim, n_act):
    with np.load(path + ("" if path.endswith(".npz") else ".npz")) as d:
        if d["W"].shape != (obs_dim, n_act):
            raise ValueError("PPO checkpoint has incompatible dimensions; run run_all.py")
        a = LinearPPO(obs_dim, n_act)
        a.W, a.b, a.vw, a.vb = d["W"].copy(), d["b"].copy(), d["vw"].copy(), float(d["vb"])
    return a
