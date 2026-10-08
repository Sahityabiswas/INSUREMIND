# Voice Training and Evaluation

Run type: configured experiment.
Environment: voice_v3_transformer_explicit_stop. Purchase threshold and reward weights unchanged.
Pretrained Vosk ASR and Windows SAPI TTS; their weights were not updated.

PPO: 20000 steps per seed; seeds [42, 43, 44]; 100 evaluation episodes per seed and policy.

## Speech Recognition

| Split | Examples | Word error rate | Character error rate |
| --- | ---: | ---: | ---: |
| train | 79 | 4.50% | 0.89% |
| val | 79 | 4.15% | 0.70% |
| test | 79 | 4.67% | 0.99% |

## Policy Comparison

All policies receive the same held-out-rate ASR observations and adapted NLP.
The text-trained checkpoint is evaluated here without modifying it.

| Policy | Mean reward | Seed SD | Simulated conversion | Mean violation turns |
| --- | ---: | ---: | ---: | ---: |
| voice_ppo | 5.0857 | 0.5633 | 6.33% | 0.0000 |
| rule | 3.7727 | 0.5589 | 9.00% | 0.9700 |
| text_trained_ppo | 4.0310 | 0.5793 | 8.33% | 0.9700 |
| prior_voice_ppo | 5.1107 | 0.6647 | 8.00% | 0.0000 |

## Limits

- Synthetic system-voice audio, not recordings from real buyers. One speaker; held-out speech rates only.
- NLP uses a separate distinct-sentence synthetic fine-tuning corpus; speech simulator templates repeat.
- Simulator utterances repeat across speech splits; this tests channel variation, not unseen language.
- The policy receives live-style defaults for hidden trust/intent and ASR-derived entities, not hidden simulator state.
- Profile and age are supplied enrollment fields; the simulator retains hidden state to compute reward.
- Low-confidence simulator retries advance one turn; live audio retries leave buyer memory unchanged.
- Acoustic emotion is advisory in live inference, not a PPO feature or trained in this experiment.
- No speech-model or Llama fine-tuning was performed.
- Microphone interaction, accent/noise robustness and human sales outcomes require separate testing.
- The original text experiment and its checkpoints are unchanged.

- Observable-dialogue mode verbalizes existing simulated objections/readiness. Compare policies within this version;
  do not attribute differences from the original voice report solely to retraining.

Detailed NLP results: `../understanding_v1/evaluation.json`. Per-episode results: `episodes.csv`.
