# Experiments and Reproduction

For the current MiniLM/RoBERTa/acoustic implementation and voice v3 training, use
[the understanding guide](understanding.md). The commands below describe the historical text experiment;
`run_all.py` overwrites its outputs, so archive them before explicitly reproducing it.

Run from the project root with the supplied Python environment:

```powershell
& '..\term_project\Scripts\python.exe' -m pytest tests -q
$env:INSURANCE_NLP_BACKEND='nb'
& '..\term_project\Scripts\python.exe' -u run_all.py
Remove-Item Env:INSURANCE_NLP_BACKEND
& '..\term_project\Scripts\python.exe' chat.py --debug --age 35
```

`configs/base.yaml` controls data, reward, environment and training. `configs/evaluation.yaml` controls
seed runs and episode counts. `--config path.yaml` selects an alternative base config. `--quick` runs a
small smoke experiment and overwrites outputs; it is labelled in `results/metrics/run_config.json`.

The full run trains PPO for 20,000 steps for each of seeds 42, 43 and 44, then evaluates 200 held-out
episodes per baseline and seed. Each ablation uses the same training budget and 100 evaluation episodes
per seed. Ablations remove emotion, objection, yes-yes, reduce actions, use conversion-only training reward,
and omit supervised initialization. All use the original evaluation reward.

Artifacts:

- `experiments/*/run.json`: actual seeds, versions, hyperparameters, reward weights and rollout histories.
- `results/checkpoints/`: PPO checkpoints for every seed and ablation; `ppo.npz` is the first seed for chat.
- `results/metrics/rl_per_seed.csv`, `rl_comparison.csv`: baseline results and seed uncertainty.
- `results/metrics/episode_metrics.csv`: complete episode outcomes and violation counts.
- `results/metrics/ablations_per_seed.csv`, `ablations.csv`: measured retrained ablations.
- `results/trajectories/evaluation.jsonl`: text responses, state transitions, rewards and provenance.
- `data/processed/customer_reaction_transitions.jsonl`: derived PPO simulation transitions.
- `results/reports/report.md`: dataset, NLP, baseline, ablation and limitation tables.

The template generator runs without a model or network. To enable a previously installed local Ollama
model, set `INSURANCE_LLM_MODEL` to its exact installed name; optionally set `INSURANCE_LLM_URL`.
Run `chat.py --generator ollama`. Rerunning the experiment includes direct-LLM and PPO wording trials;
this can be much slower because every turn calls the model. No models are automatically downloaded.

## Public NLP Corpora

The offline experiment records upstream URLs and license names without claiming downloaded examples.
`src.external_data` reads actual local MultiDoGO TSV, GoEmotions headerless TSV and MELD text CSV:

```powershell
python -m src.external_data --dataset goemotions --input data/raw/goemotions/train.tsv --split train
python -m src.external_data --dataset multidogo --input data/raw/multidogo/train.tsv --split train --intent-map configs/multidogo_intents.json
python -m src.external_data --dataset meld --input data/raw/meld/train_sent_emo.csv --split train
```

Provide an explicit native-to-project intent mapping for MultiDoGO. Emotion compression is a versioned,
derived mapping; conflicting multilabel GoEmotions examples remain unlabeled. Original labels, source
licenses, original split and file SHA-256 are retained. Cross-split conversation overlap is rejected.
Imported corpora remain separate in `data/interim/external`; the default experiment does not mix them
into the generated insurance dataset. NLP transfer training requires an explicit experimental design.
