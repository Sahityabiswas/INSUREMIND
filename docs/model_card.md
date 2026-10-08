# Model Card

Purpose: academic study of adaptive insurance conversation strategy in a synthetic environment.
Models: fine-tuned MiniLM intent/multi-label objections, mapped RoBERTa text emotion, pretrained wav2vec2
acoustic emotion, supervised linear action policy and linear NumPy PPO. NB remains an explicit baseline.

Inputs: customer text for NLP, waveform for acoustic emotion, a 94-feature compact state for PPO. Outputs: NLP predictions
with probabilities, one of 16 sales actions and its controlled strategy label, then a verified template or
optional local LLM response. Products and premiums are synthetic examples, not insurer quotations.

MiniLM was fully fine-tuned on 234 distinct synthetic/weakly labeled sentences with a 37-sentence validation
split and 39-sentence test. Objection outputs use independent sigmoids. RoBERTa uses a heuristic 28-to-8
mapping tested on 32 authored examples; it was not fine-tuned here. Vocal emotion retains four original
labels, is uncalibrated, and has no labeled-human accuracy evaluation. spaCy is supported but its native
DLL is blocked on this host; visible regex fallback is active. See [full understanding card](understanding.md).

Training data are generated templates, not verified buyer labels. Simulator state is more directly observed than
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
speech. Neither speech model is fine-tuned. The separate PPO policy is trained on synthetic speech
transcripts with transformer understanding; the CLI and browser use `configs/voice_transformer.yaml`.
Current v3 artifacts are in `results/voice_v3_transformer/`; earlier text/v1/v2 checkpoints remain preserved.
The NB baseline uses `configs/voice.yaml`. Acoustic emotion cannot alter consent or PPO features.
Typed messages can also exercise the voice-trained policy in the browser, sharing memory with audio turns.

Voice state uses observable text cues and live defaults for unobserved latent scores. Explicit affirmative
interest allows commitment actions; negated/conditional interest does not. Recognized stop requests override
the ASR confidence retry gate. Low-confidence or silent audio otherwise leaves buyer memory unchanged.
This does not guarantee recognition of every real spoken stop request; missed stop turns are reported.

Hybrid mode uses deterministic replies for selected constrained tasks and Llama for eligible open-ended
wording. Planned template routing, LLM fallback and actual LLM output are labelled separately. The basic
response guard is not semantic, legal or regulatory verification. Invalid checked replies are not spoken.

The current evaluation uses one synthetic system voice at different rates, not independent human speakers.
Voice v3 includes transformer estimates and corrected explicit-stop masks. New voice PPO obtains 6.33%
simulated conversion and the prior voice checkpoint 8.00% in the same v3 run; superior retraining
performance has not been established. Historical v2 metrics differ and are not directly comparable.
The UI displays separate experiment provenance and does not equate synthetic
conversion with policy sales. Real microphone/accent/noise testing and human evaluation remain necessary.
