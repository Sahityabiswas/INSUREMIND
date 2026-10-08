# Voice Training and Evaluation

Run type: configured experiment.
Environment: voice_v2_observable_dialogue. Purchase threshold and reward weights unchanged.
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
| voice_ppo | 5.9363 | 0.5554 | 11.33% | 0.0167 |
| rule | 4.7697 | 0.3919 | 11.67% | 0.9700 |
| text_trained_ppo | 5.8260 | 0.3481 | 12.00% | 0.2600 |
| prior_voice_ppo | 5.9690 | 0.5165 | 11.33% | 0.0000 |

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

- Observable-dialogue mode verbalizes existing simulated objections/readiness. Compare policies within this version;
  do not attribute differences from the original voice report solely to retraining.

Detailed NLP results: `nlp_metrics.json`. Per-episode results: `episodes.csv`.
