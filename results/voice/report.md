# Voice Training and Evaluation

Run type: configured experiment.
Pretrained Vosk ASR and Windows SAPI TTS; their weights were not updated.

PPO: 20000 steps per seed; seeds [42, 43, 44]; 100 evaluation episodes per seed and policy.

## Speech Recognition

| Split | Examples | Word error rate | Character error rate |
| --- | ---: | ---: | ---: |
| train | 74 | 4.82% | 0.96% |
| val | 74 | 4.45% | 0.75% |
| test | 74 | 5.01% | 1.06% |

## Policy Comparison

All policies receive the same held-out-rate ASR observations and adapted NLP.
The text-trained checkpoint is evaluated here without modifying it.

| Policy | Mean reward | Seed SD | Simulated conversion | Mean violation turns |
| --- | ---: | ---: | ---: | ---: |
| voice_ppo | 6.4493 | 0.3997 | 0.00% | 0.0000 |
| rule | 4.8287 | 0.3000 | 0.00% | 0.9700 |
| text_trained_ppo | 6.0613 | 0.4192 | 0.00% | 0.3100 |

## Limits

- Synthetic system-voice audio, not recordings from real buyers. One speaker; held-out speech rates only.
- NLP uses original conversation splits. Repeated synthetic utterance templates remain a limitation.
- Simulator utterances repeat across speech splits; this tests channel variation, not unseen language.
- The policy receives live-style defaults for hidden trust/intent and ASR-derived entities, not hidden simulator state.
- Profile and age are supplied enrollment fields; the simulator retains hidden state to compute reward.
- Low-confidence simulator retries advance one turn; live audio retries leave buyer memory unchanged.
- No acoustic-emotion model, speech-model fine-tuning, or Llama fine-tuning was performed.
- Microphone interaction, accent/noise robustness and human sales outcomes require separate testing.
- The original text experiment and its checkpoints are unchanged.

Detailed NLP results: `nlp_metrics.json`. Per-episode results: `episodes.csv`.
