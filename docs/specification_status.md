# Specification Implementation Status

This document separates implemented offline functionality from experiments needing external resources.

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Scenario dataset, provenance, strategies, profiles, validation and grouped split | `src/data_pipeline.py`, schemas and dataset card | 1,200 conversations / 20,396 turns; 3,000 profiles; tests |
| Intent and multi-label objections | `src/understanding.py`, `train_understanding.py` | Fully fine-tuned MiniLM, sigmoid objections, distinct-sentence synthetic checks |
| Text emotion / entities / stage | `src/understanding.py`, `src/entities.py`, `src/conversation.py` | Explicit RoBERTa mapping, eight-category check, exact numeric rules, observable stage guards |
| Acoustic emotion | `src/acoustic_emotion.py` | Actual waveform ONNX inference; independent four-label estimates, not human-validated |
| Structured normalized state and compact memory | `src/state_products.py` | 94-feature state and deterministic default/ablation tests |
| Product database, age/budget/need eligibility, fact guard | `src/state_products.py` | Eligible-only recommendations; unmatched cases return empty; fact tests |
| Strategy-controlled templates, hybrid routing and LLM verbalizer | `src/generation.py` | Explicit route/timing/fallback traces; strategy/fact rejection and deterministic-task tests |
| Simulator, reward, masks, Gymnasium environment | `src/sim_env.py` | Official Gymnasium checker; reward, rejection, time-limit and transition tests |
| Random, rule, supervised and PPO policies | `src/agents.py`, `src/ppo_numpy.py` | Same held-out scenario seeds and all-turn metrics |
| PPO clipping, masked sampling, GAE, critic, optimization, checkpointing | `src/ppo_numpy.py` | Return/bootstrapping/sampling/checkpoint tests; actual training histories |
| Multiple seeds and uncertainty | `src/evaluate.py`, evaluation config | Seeds 42/43/44, 200 baseline episodes each; seed SD and 95% intervals |
| Emotion, objection, strategy and reward ablations | `src/evaluate.py` | Six independently retrained variants with equal training budgets |
| Text conversation system | `src/conversation.py`, `chat.py` | PPO example and multi-turn entity/rejection tests |
| Core local voice session | `src/speech.py`, `voice.py`, `run_llm.py --voice` | WAV/recognition/synthesis path; silence, confidence, stop and validation tests |
| Speech-conditioned insurance training | `src/voice_training.py`, `train_voice.py` | Three 20K-step seeds; current results under `results/voice_v3_transformer/`; original v1/v2 retained |
| Cooperative intake / transcript correction | `src/dialogue_flow.py`, API and UI | One-question flow, explicit-stop-only closure, review without advancing state, original transcript/audio-estimate audit |
| Observable purchase interest and objections | `src/conversation.py`, `src/sim_env.py` | Negation/conditional-intent tests; versioned speech environment; prior checkpoint comparison |
| Integrated text/voice UI | `demo_server.py`, `web/`, `src/voice_runtime.py` | Shared trained artifacts, typed/spoken memory, native audio API tests and desktop/mobile browser checks |
| Local browser microphone and playback | `web/audio.js`, `src/speech.py` | Synthetic capture, WAV validation, request replay, permission/cancel cleanup and decoded TTS playback |
| Reports, representative conversations, logging, reward curves | `src/evaluate.py`, experiments and results | Generated report, episode metrics, transition JSONL and curve PNGs |
| Public corpus URLs, licenses and readers | `src/external_data.py`, `data/raw/*/README.md` | Local MultiDoGO, GoEmotions and MELD readers with source hashes and leakage guard |

Remaining external experiments:

- No public corpus is loaded into the default training run. Normalize actual downloaded files and define a
  separate transfer experiment before claiming performance from MultiDoGO, GoEmotions or MELD.
- The real direct-LLM baseline and wording ablation require an installed model and running Ollama server.
  The initial benchmark run had no reachable local Ollama server, so these experiments were reported as
  skipped. Installing Ollama later does not change those recorded benchmark results.
- PyTorch CPU now works and performed MiniLM fine-tuning. PPO remains NumPy. A scikit-learn DLL import
  and a spaCy DLL remain blocked; the compatible transformer version and explicit regex fallback are documented.
- Local speech recognition and Windows TTS are now implemented with pretrained models. Insurance NLP
  adaptation and speech-conditioned PPO training have separate outputs. Speech-model/LLM fine-tuning,
  eight-category emotion fine-tuning, labeled acoustic evaluation and human evaluation remain unimplemented. Synthetic speech results
  do not establish microphone, accent or real-buyer performance; see `docs/voice.md`.

The current evidence supports an offline research prototype, not completion of every external-data/LLM
experiment in the full specification. See `results/reports/report.md` for text outcomes and
`results/voice_v3_transformer/report.md` for the current versioned voice experiment. Their environments differ;
the changed conversion rate must not be attributed to retraining alone.
