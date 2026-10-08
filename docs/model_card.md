# Model Card

Purpose: academic study of adaptive insurance conversation strategy in a synthetic environment.
Models: multinomial Naive Bayes understanding, supervised linear softmax action policy, linear NumPy PPO.

Inputs: customer text for NLP, a 94-feature compact state for PPO. Outputs: categorical NLP predictions
with probabilities, one of 16 sales actions and its controlled strategy label, then a verified template or
optional local LLM response. Products and premiums are synthetic examples, not insurer quotations.

The default models are trained on generated templates. Simulator state is more directly observed than
live conversation state. Emotion and sales-stage accuracy are limited; generated data does not establish
deployment readiness or real customer conversion. Existing checkpoints must be regenerated after state
or taxonomy changes.

Use `results/reports/report.md` for measured metrics and `experiments/*/run.json` for seeds, configuration,
reward weights, training history and versions. Unsupported needs have no eligible recommendations.
The system honors explicit stop requests. The factuality guard covers numeric terms and a small set of
phrases; unrecognized semantic claims may pass. Review real product facts and responses before any real use.

Optional LLM evaluation requires `INSURANCE_LLM_MODEL` and a reachable Ollama endpoint. Failed or
unconfigured direct-LLM trials are excluded and listed as skipped, not silently replaced by a rule policy.

## Voice and Integrated Runtime

Pretrained Vosk supplies English transcripts and confidence estimates. Windows SAPI supplies generated
speech. Neither speech model is fine-tuned. The voice insurance NLP bundle and separate PPO policy are
trained on synthetic speech transcripts; the CLI and browser resolve their paths from `configs/voice.yaml`.
Current v2 artifacts are in `results/voice_v2/`; earlier text and voice checkpoints remain preserved.
Typed messages can also exercise the voice-trained policy in the browser, sharing memory with audio turns.

Voice state uses observable text cues and live defaults for unobserved latent scores. Explicit affirmative
interest allows commitment actions; negated/conditional interest does not. Recognized stop requests override
the ASR confidence retry gate. Low-confidence or silent audio otherwise leaves buyer memory unchanged.
This does not guarantee recognition of every real spoken stop request; missed stop turns are reported.

Hybrid mode uses deterministic replies for selected constrained tasks and Llama for eligible open-ended
wording. Planned template routing, LLM fallback and actual LLM output are labelled separately. The basic
response guard is not semantic, legal or regulatory verification. Invalid checked replies are not spoken.

The current evaluation uses one synthetic system voice at different rates, not independent human speakers.
Voice v2's observable simulator dialogue differs from the original experiment. Prior and new voice PPO
both obtain 11.33% simulated conversion in the v2 environment, so superior retraining performance has
not been established. The UI displays separate experiment provenance and does not equate synthetic
conversion with policy sales. Real microphone/accent/noise testing and human evaluation remain necessary.
