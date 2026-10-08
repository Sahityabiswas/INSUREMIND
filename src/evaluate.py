"""Phase 13-15: baseline comparison, ablations (emotion/reward/strategy/generator), metrics + report.
Stdlib-only CSV/markdown (pandas blocked on this host)."""
import os, json, csv, random, math
from statistics import mean, stdev
from .common import load_config, ACTIONS
from .sim_env import InsuranceEnv
from . import agents as AG
from . import ppo_numpy as PN
from .generation import OllamaClient, ResponseGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def _md(rows):
    if not rows: return "(empty)"
    keys = list(rows[0].keys())
    return ("| " + " | ".join(keys) + " |\n| " + " | ".join(["---"] * len(keys)) + " |\n"
            + "\n".join("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows))

def summarize(runs, group="agent"):
    out = []
    for name in dict.fromkeys(r[group] for r in runs):
        subset = [r for r in runs if r[group] == name]
        row = {group: name, "seeds": len(subset)}
        keys = [k for k, v in subset[0].items() if isinstance(v, (float, int)) and k != "seed"]
        for key in keys:
            values = [r[key] for r in subset]
            avg = mean(values)
            sd = stdev(values) if len(values) > 1 else 0.0
            # Student-t critical values for a small number of independent seed runs.
            critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(values), 1.96)
            half = critical * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row.update({key: round(avg, 4), key + "_std": round(sd, 4),
                        key + "_ci95_low": round(avg - half, 4), key + "_ci95_high": round(avg + half, 4)})
        out.append(row)
    return out


def _save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def compare_all(weights, episodes=200, seed=42, seeds=None, max_turns=20):
    runs, episodes_log, trajectories, skipped = [], [], [], []
    for trial_seed in seeds or [seed]:
        env = InsuranceEnv(reward_weights=weights, max_turns=max_turns)
        rng = random.Random(trial_seed)
        model = PN.load(os.path.join(ROOT, "results", "checkpoints", f"ppo_seed_{trial_seed}.npz"),
                        env.observation_space.shape[0], env.action_space.n)
        policies = {"random": lambda o, m, i: rng.choice([a for a, ok in enumerate(m) if ok]),
                    "rule": AG.rule_policy(), "supervised": AG.supervised_policy(), "ppo": AG.ppo_policy(model)}
        if OllamaClient().configured:
            policies["direct_llm"] = AG.direct_llm_policy()
        for name, policy in policies.items():
            print(f"evaluate {name}, seed={trial_seed}", flush=True)
            try:
                result = AG.run_policy(env, policy, episodes, trial_seed, name, episodes_log, trajectories)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                if name != "direct_llm":
                    raise
                skipped.append({"experiment": name, "seed": trial_seed, "reason": type(exc).__name__})
                continue
            runs.append({**result, "seed": trial_seed})
    if not OllamaClient().configured:
        skipped.append({"experiment": "direct_llm", "reason": "INSURANCE_LLM_MODEL is not configured"})
    # A failed LLM seed invalidates its whole aggregate rather than selecting successful seeds.
    if any(r["experiment"] == "direct_llm" for r in skipped):
        runs = [r for r in runs if r["agent"] != "direct_llm"]
    _write_csv(os.path.join(ROOT, "results", "metrics", "rl_per_seed.csv"), runs)
    _write_csv(os.path.join(ROOT, "results", "metrics", "episode_metrics.csv"), episodes_log)
    _save_jsonl(os.path.join(ROOT, "results", "trajectories", "evaluation.jsonl"), trajectories)
    _save_jsonl(os.path.join(ROOT, "data", "processed", "customer_reaction_transitions.jsonl"),
                [r for r in trajectories if r["agent"] == "ppo"])
    with open(os.path.join(ROOT, "results", "metrics", "skipped_experiments.json"), "w") as f:
        json.dump(skipped, f, indent=2)
    out = summarize(runs)
    _write_csv(os.path.join(ROOT, "results", "metrics", "rl_comparison.csv"), out)
    return out

def ablations(weights, episodes=100, seed=42, seeds=None, ppo_config=None, max_turns=20):
    settings = ppo_config or load_config()["ppo"]
    reduced = {"DISCOVER_NEEDS", "ASK_CLARIFYING_QUESTION", "EXPLAIN_PRODUCT", "HANDLE_OBJECTION",
               "BUILD_TRUST", "PERSONALIZE", "ASK_FOR_COMMITMENT", "FOLLOW_UP", "RESPECT_REJECTION"}
    variants = {"full_ppo": {}, "no_emotion": {"disabled_features": ["emotion"]},
                "no_objection": {"disabled_features": ["objection"]},
                "no_yes_yes": {"banned_actions": ["YES_YES_FRAMING"]},
                "reduced_actions": {"banned_actions": [a for a in ACTIONS if a not in reduced]},
                "conversion_only_reward": {}, "no_supervised_init": {}}
    runs = []
    for config, options in variants.items():
        for trial_seed in seeds or [seed]:
            tag = f"{config}_seed_{trial_seed}"
            env_options = {"max_turns": max_turns, **options}
            if config == "full_ppo":
                model = PN.load(os.path.join(ROOT, "results", "checkpoints", f"ppo_seed_{trial_seed}.npz"),
                                InsuranceEnv().observation_space.shape[0], len(ACTIONS))
            else:
                train_weights = {k: (v if k == "suitable_purchase" else 0.0) for k, v in weights.items()} if config == "conversion_only_reward" else weights
                print(f"train ablation {config}, seed={trial_seed}", flush=True)
                model = AG.train_ppo(train_weights, settings["total_timesteps"], trial_seed,
                                     settings, tag, env_options, warm_start=config != "no_supervised_init")
            # All reward variants are evaluated with the same original reward objective.
            env = InsuranceEnv(reward_weights=weights, **env_options)
            result = AG.run_policy(env, AG.ppo_policy(model), episodes, trial_seed, "ppo")
            runs.append({**result, "config": config, "seed": trial_seed})
    if OllamaClient().configured:
        for trial_seed in seeds or [seed]:
            env = InsuranceEnv(reward_weights=weights, max_turns=max_turns, generator=ResponseGenerator("ollama"))
            model = PN.load(os.path.join(ROOT, "results", "checkpoints", f"ppo_seed_{trial_seed}.npz"),
                            env.observation_space.shape[0], len(ACTIONS))
            result = AG.run_policy(env, AG.ppo_policy(model), episodes, trial_seed, "ppo")
            runs.append({**result, "config": "ppo_ollama_with_validated_fallback", "seed": trial_seed})
    _write_csv(os.path.join(ROOT, "results", "metrics", "ablations_per_seed.csv"), runs)
    res = summarize(runs, "config")
    _write_csv(os.path.join(ROOT, "results", "metrics", "ablations.csv"), res)
    return res

def dataset_table():
    from .data_pipeline import EXTERNAL
    stats = json.load(open(os.path.join(ROOT, "data", "processed", "insurance_sales", "stats.json")))
    out = [{"Dataset": name, "Size": "metadata only; not used", "Source": url,
            "Purpose": purpose, "License": lic, "Real/Synthetic": "external reference"}
           for name, url, lic, purpose in EXTERNAL]
    out.append({"Dataset": "insurance_sales_dialogue_dataset v2", "Size": f"{stats['conversations']} conversations / {stats['turns']} turns",
                "Source": "scenario-generator", "Purpose": "action/strategy supervision", "License": "synthetic-internal",
                "Real/Synthetic": "synthetic"})
    out.append({"Dataset": "insurance_customer_profiles", "Size": stats["profiles"], "Source": "profile-generator",
                "Purpose": "customer simulation", "License": "synthetic-internal", "Real/Synthetic": "synthetic"})
    return out

def write_report(nlp_metrics, comparison, abl):
    rep = os.path.join(ROOT, "results", "reports")
    os.makedirs(rep, exist_ok=True)
    def display(rows, name):
        keys = ("avg_reward", "conversion", "qualified_lead", "objection_resolution", "satisfaction",
                "avg_turns", "pressure_violations", "unsupported", "unsuitable_recommendation")
        return [{name: r[name], "seed_runs": r["seeds"], **{k: f"{r[k]:.3f} +/- {r[k + '_std']:.3f}" for k in keys}}
                for r in rows]
    stats = json.load(open(os.path.join(ROOT, "data", "processed", "insurance_sales", "stats.json")))
    skipped = json.load(open(os.path.join(ROOT, "results", "metrics", "skipped_experiments.json")))
    lines = ["# Insurance Sales RL Agent - Results Report", "",
        "This is an offline research prototype. No public corpus was used in this run; upstream dataset references and licenses are documented.",
        f"Synthetic evidence: {stats['conversations']} conversations, {stats['turns']} turns, {stats['profiles']} synthetic profiles. Splits: {stats['splits']}.",
        "Simulated evidence: customer transitions and conversions are generated by the environment, not observed human outcomes.",
        "Experimental evidence: independently trained PPO seed runs, paired held-out scenario seeds, identical products and turn limits.",
        "Tables report seed means +/- sample standard deviation. CSVs include Student-t 95% intervals over seed means; three seeds provide limited statistical power.",
        "Objection resolution is measured over episodes initially containing an objection. Violations count every turn, not only the final turn.", "",
        "## Table A - Datasets", _md(dataset_table()), "",
        "## Table B - NLP", _md([{"Model": k, **v} for k, v in nlp_metrics.items()]), "",
        "## Table C - RL comparison", _md(display(comparison, "agent")), "",
        "## Table D - Ablations", _md(display(abl, "config")), "",
        "Ablations are retrained with identical step budgets. Removed features are disabled during training and evaluation. Reward ablations use a shared evaluation reward.",
        "The simulator reacts to actions and eligibility; it does not semantically evaluate wording. LLM wording comparisons measure generator failures, not human preference.", "",
        "## Skipped Experiments", _md(skipped) if skipped else "None.",
        "PPO + Ollama wording ablation is skipped when INSURANCE_LLM_MODEL is unset.", "",
        "## Reward Curves", "Each experiments/*/run.json records rolling training reward and value loss across rollout updates.",
        "Optional matplotlib produces corresponding *_reward_curve.png plots. These curves measure training reward, not held-out performance.", "",
        "## Representative Conversations"]
    by_agent = {r["agent"]: r for r in comparison}
    if "ppo" in by_agent and "rule" in by_agent:
        ppo, rule = by_agent["ppo"], by_agent["rule"]
        interpretation = (f"PPO mean simulated reward is {ppo['avg_reward']:.3f}; rule baseline is {rule['avg_reward']:.3f}. "
                          f"Simulated conversion is {ppo['conversion']:.1%} for PPO versus {rule['conversion']:.1%} for rule. "
                          "Reward and conversion can favor different policies. These seed runs do not establish real-world effectiveness or statistical superiority.")
        lines[1:1] = ["", "## Interpretation", interpretation, ""]
    run_config = os.path.join(ROOT, "results", "metrics", "run_config.json")
    if os.path.exists(run_config) and json.load(open(run_config)).get("quick"):
        lines[1:1] = ["", "SMOKE RUN: reduced training and evaluation budgets; not a full research experiment."]
    curve = "../metrics/ppo_seed_42_reward_curve.png"
    if os.path.exists(os.path.join(rep, curve)):
        position = lines.index("## Reward Curves") + 1
        lines[position:position] = [f"![PPO training reward for seed 42]({curve})", ""]
    trajectory_path = os.path.join(ROOT, "results", "trajectories", "evaluation.jsonl")
    with open(trajectory_path) as f:
        examples = [json.loads(line) for line in f if '"agent": "ppo"' in line][:8]
    for r in examples:
        lines.extend([f"### Seed {r['seed']}, episode {r['episode']}, turn {r['turn']}",
                      f"Strategy: {r['seller_strategy']}. Reward: {r['reward']}.",
                      f"Agent: {r['seller_response']}", f"Synthetic customer: {r['customer_reply']}", ""])
    lines.extend(["## Limitations", "- Synthetic templates have repeated wording; scenario grouping reduces leakage but cannot establish human generalization.",
                  "- Emotion/stage labels are generated heuristically and are not validated human annotations.",
                  "- Linear NumPy PPO is a portable reference backend; production use should benchmark a maintained RL library.",
                  "- Synthetic product premium bands are not insurer quotations; some needs have no matching products.",
                  "- Response validation checks numeric claims, certain guarantees, and repetition; it is not a complete semantic factuality proof.",
                  "- External NLP transfer and real LLM results need downloaded corpora and a configured local model. Speech is deferred.", "",
                  "## Method Sources", "- PPO: https://arxiv.org/abs/1707.06347",
                  "- Gymnasium: https://gymnasium.farama.org/api/env/",
                  "- Ollama: https://docs.ollama.com/api/chat"])
    with open(os.path.join(rep, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return os.path.join(rep, "report.md")
