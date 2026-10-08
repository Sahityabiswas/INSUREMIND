# Methodology

The research system preserves `understanding -> state -> PPO action -> strategy -> verified facts -> wording`.
The PPO policy never emits text, and the optional Ollama verbalizer cannot change its assigned strategy.

## Data and NLP

The default experiment is offline and uses a custom synthetic dataset. Generated intent/emotion labels,
reaction labels, purchase intent, and outcomes are explicitly marked synthetic rather than human observations.
Scenario variants and identical normalized transcripts share a split. Repeated template wording remains a
limitation, so high synthetic intent accuracy is not evidence of real-world accuracy.

Naive Bayes intent, emotion, objection and sales-stage models are evaluated on the held-out conversation
split. Macro-F1 averages classes equally; weighted-F1 uses true class support. Per-task confusion matrices
are saved. Emotion probabilities, normalized behavioral estimates and compact history enter the 94-feature state.
In the simulator, state features reflect simulated ground truth. In the live text session, they are NLP estimates;
trust and purchase intent remain unobserved defaults. This distribution difference limits policy transfer.

## PPO and Constraints

The portable NumPy actor is a linear masked softmax and the critic is linear. Training samples from the
masked categorical distribution and saves the matching log probability. Generalized advantage estimation
uses unnormalized returns for the critic and normalized advantages for the clipped actor objective.
True terminal outcomes disable value bootstrapping; time limits bootstrap value but stop GAE across resets.
Updates use shuffled minibatches, Adam, entropy regularization and gradient clipping.

Supervised action weights initialize PPO unless an experiment disables initialization. Each configured seed
produces its own trained checkpoint. The NumPy backend supports constrained Windows environments; it is
a reference implementation, not a substitute for comparison against a maintained RL library. The supplied
Windows environment blocks `torch_python.dll` with Application Control error 4551, preventing the installed
Stable Baselines3 from loading; NumPy and matplotlib load successfully.

The product engine returns only products meeting need, age and budget constraints. Unmatched needs return
no recommendation. Masks prevent product actions before discovery, premature commitment, and continuation
after explicit rejection. Response guards reject unsupported numeric terms, certain guarantees and repetition.
They do not prove arbitrary language-model statements are semantically factual.

## Experiments

Baselines are random, rule, supervised and PPO. A real direct-LLM policy is available with a configured
Ollama model; an unavailable model is reported as skipped. No heuristic is labelled as an LLM result.
Customer transitions depend on strategy and eligibility, not a semantic score of generated wording.

All baseline policies receive the same held-out scenario seeds, product database and maximum turns.
The initial customer conditions and random source are identical; differing actions can consume randomness
differently, so subsequent trajectories can diverge. Evaluation seeds are disjoint from training seeds.
Pressure and unsupported-claim counters accumulate across every turn. Objection resolution uses episodes
that initially contain an objection as its denominator.

Independent runs produce mean, sample standard deviation and Student-t 95% confidence intervals over seed
means. Three seeds have limited statistical power. Reward variants are trained on their own objectives but
evaluated with the original shared reward. Feature and action ablations are retrained with the same budgets.

Sources: [PPO paper](https://arxiv.org/abs/1707.06347),
[Gymnasium environment contract](https://gymnasium.farama.org/api/env/),
[Ollama chat endpoint](https://docs.ollama.com/api/chat).
