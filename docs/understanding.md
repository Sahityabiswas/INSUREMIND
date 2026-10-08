# Language and Acoustic Understanding

## What Changed

The live CLI, API and browser now default to the same transformer understanding module. Naive Bayes is
preserved as an explicitly selected baseline, not a hidden fallback. Missing transformer artifacts produce
an actionable error. Response wording remains a separate template/Ollama component: changing the
classifier does not fine-tune Llama or solve Llama's CPU latency.

| Task | Model or method | Local adaptation |
| --- | --- | --- |
| Intent | MiniLM-L6, 384-dimensional masked mean pooling, nine-way softmax head | Encoder layers and head fine-tuned |
| Objections | Same MiniLM encoder, ten independent sigmoid outputs | BCEWithLogitsLoss, multi-hot labels |
| Text emotion | INT8 RoBERTa GoEmotions, 28 original sigmoid scores | Heuristic mapping to eight categories, evaluated but not fine-tuned |
| Vocal emotion | wav2vec2-base-superb-er, float32 ONNX waveform inference | Pretrained only; four original IEMOCAP categories |
| Entities | spaCy EntityRuler plus regex/Decimal numeric rules | No trained statistical NER; explicit regex fallback if spaCy is unavailable |
| Dialogue stage | Observable dialogue state, explicit cues and workflow | Not inferred by an obsolete NB stage classifier in transformer mode |

Primary model sources and licenses:
- [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0.
- [RoBERTa GoEmotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) and the
  [author's ONNX export](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx), MIT.
- [Original wav2vec2 SUPERB ER](https://huggingface.co/superb/wav2vec2-base-superb-er) and
  [community ONNX export](https://huggingface.co/onnx-community/wav2vec2-base-superb-er-ONNX), Apache-2.0.
- [spaCy rule-based matching](https://spacy.io/usage/rule-based-matching).

`setup_understanding.py` pins immutable repository revisions and downloads only explicitly listed data
files. Each model directory contains a provenance manifest and file SHA-256 hashes. No model-provided
Python is executed. MiniLM checkpoints use safetensors, not remote pickle. Normal inference is local-only.

## Why Transcription Is Not Enough

Transcription preserves words, so it still supports semantic intent and objection detection. It loses
much of the pitch, intensity, timing and vocal quality available in the waveform. The updated voice path is:

```text
Explicit microphone click / supplied 16 kHz mono PCM WAV
    +--> Vosk --> transcript + confidence --> optional review/correction --> MiniLM + RoBERTa
    +--> wav2vec2 waveform classifier --> four acoustic scores + uncertainty
                                  |
                        shared ConversationSession
                                  |
       exact entities + explicit stop/consent guards + optional guided intake
                                  |
                 94-feature state --> masked PPO --> wording
                                  |
                    response checks --> local SAPI speech
```

The acoustic model receives samples, not the transcript. Its preprocessing follows its downloaded
configuration: 16 kHz float waveform scaled from PCM; this particular export has `do_normalize: false`.
At most the first ten seconds are analyzed; duration/truncation are reported. Silence and recordings
shorter than 0.5 seconds are not assigned an emotion. An estimate is marked uncertain below 0.65 top score
or a 0.15 top-two margin. These thresholds are heuristics, not validated confidence calibration.

Original labels are NEUTRAL, HAPPY, ANGRY and SAD. They are not falsely renamed into the eight project
categories. A sufficiently strong angry/sad estimate can request patient, brief, non-pressuring LLM wording.
Audio never changes PPO features, factual eligibility, financial recommendations, purchase readiness or
consent. It is not used to target vulnerable people. A recognized stop request skips emotion analysis and
ends the conversation. Acoustic failures are visible but do not discard usable recognized text.

The browser submits accepted speech directly and speaks the checked answer by default. **Review speech**
is an optional control for preview/correction; preview does not advance memory or PPO. When the user
corrects and confirms it, the original ASR transcript and the same acoustic estimate stay attached to the
turn. The main inspector shows **Vocal emotion** from the waveform for voice turns, or an explicit
uncertain/unavailable/skipped status. It never substitutes text emotion for missing audio analysis.
**Spoken intent** is semantic intent from recognized words, not an acoustic-intent model. The separate
expandable **Transcript emotion** section retains RoBERTa's mapped text estimate. Typed turns keep text
emotion in the main inspector and do not auto-play speech. **Spoken replies** can be switched off;
individual speaker buttons remain available. Raw recordings are deleted after processing; neither recordings nor live transcripts automatically
enter the training set. Human audio collection/training needs explicit consent and labeled data.

## Text Emotion Mapping

The maximum original score within each group is used, then the eight group scores are normalized.
These are mapped, uncalibrated scores, not verified emotions or calibrated probabilities.

| Project label | Original GoEmotions labels |
| --- | --- |
| POSITIVE | joy, excitement, optimism, love, amusement, admiration |
| NEUTRAL | neutral, curiosity, realization, surprise |
| CONCERNED | confusion, sadness, disappointment, caring |
| SKEPTICAL | disapproval |
| FRUSTRATED | annoyance |
| ANGRY | anger, disgust |
| ANXIOUS | fear, nervousness |
| SATISFIED | gratitude, relief, approval, pride |

Several original categories are deliberately unmapped. Skepticism is not equivalent to disapproval;
concern overlaps anxiety. These are known semantic weaknesses. A labeled insurance-specific emotion
dataset and further fine-tuning are necessary before treating these distinctions as dependable.

## Language Training and Evaluation

`train_understanding.py` trains all MiniLM encoder layers, an intent head and an objection head. The losses
are weighted cross-entropy and positive-weighted binary cross-entropy. AdamW uses encoder learning rate
3e-5, head learning rate 1e-3, batch size 16, weight decay 0.01 and gradient clipping at 1.0. Seed 42,
12 epochs, maximum 128 tokens, CPU with two threads. The best epoch is chosen on validation only.
Per-label objection thresholds are selected on validation from 0.3 through 0.7. Test labels do not select
epochs or thresholds. NONE is an empty objection set; it cannot coexist with another objection.

The original corpus has 6,888 training customer turns but just 42 distinct texts. All 42 also occur in the
original test split; 32 carry conflicting emotion labels. That old split does not establish unseen-language
generalization. It is preserved for historical reproducibility.

`data/annotations/understanding_v1.json` adds agent-authored synthetic examples with explicit provenance.
Compound objections are generated only from training sentences after splits are assigned. Distinct
legacy intent/objection sentences are included as weak labels, but inconsistent legacy emotion labels are
not used for new fine-tuning. New split sizes are 234 train, 37 validation and 39 test, with zero exact
normalized-text overlap. Related paraphrases and synthetic authorship remain important limitations.
These are not human or domain-expert annotations, and not independent real-customer evidence.

Recorded `results/understanding_v1/evaluation.json`:

| Task | Metric | Result |
| --- | --- | ---: |
| Intent | Accuracy / macro-F1 | 0.872 / 0.802 |
| Multi-label objections | Micro-F1 / macro-F1 | 0.818 / 0.804 |
| Multi-label objections | Exact set match | 0.718 |
| Eight-category emotion mapping | Macro-F1 | 0.787 |

The report includes per-objection support/F1, test predictions, emotion confusion counts and all emotion
examples. Test sets are small: 39 language cases and 32 emotion cases. There is no labeled human-audio
accuracy result. Generated neutral speech verifies the acoustic plumbing, not emotion accuracy.

## PPO Compatibility and Results

Full objection sets are exposed to the UI, trace and wording context. The existing 94-feature PPO state
uses only the primary objection so old checkpoints can still load. Encoding all labels or adding audio
features would require a new state version and new checkpoints. No such compatibility is implied here.

`configs/voice_transformer.yaml` defines a separate v3 run: MiniLM + mapped RoBERTa, real cached Vosk
hypotheses, three seeds (42/43/44), 20,000 PPO steps per seed, 100 evaluation episodes per seed/policy.
Four policies yield 1,200 evaluation conversations. New checkpoints are in `results/voice_v3_transformer/`.
The first configured seed is deployed, not a test-selected best seed. Old artifacts are untouched.

In this environment, new PPO mean reward is 5.0857 and simulated conversion is 6.33%; the prior voice
checkpoint obtains 5.1107 and 8.00%. Both have zero measured violation turns here. This does not establish
an improvement in sales or policy learning. Different environment versions cannot isolate classifier
effects; compare checkpoints within the same run. One synthetic speaker, finite repeated language,
unvalidated emotion mapping and simulator rewards are substantial limitations.

## Commands

All commands run from `D:\insurence seller\insurance_sales_agent` using `..\term_project\Scripts\python.exe`.
The existing installation and checkpoints require no download or training just to run.

```powershell
# UI; use an unused port
& '..\term_project\Scripts\python.exe' demo_server.py --port 8767
# Interactive text / voice
& '..\term_project\Scripts\python.exe' chat.py --age 35 --nlp-backend transformer --debug
& '..\term_project\Scripts\python.exe' voice.py --age 35 --nlp-backend transformer --debug
# Explicit historical baseline
& '..\term_project\Scripts\python.exe' chat.py --age 35 --nlp-backend nb
# Fresh language experiment, preserving the completed v1 run
& '..\term_project\Scripts\python.exe' train_understanding.py --output results/understanding_v2
# Small policy plumbing check, separate smoke artifacts, not a research benchmark
& '..\term_project\Scripts\python.exe' train_voice.py --config configs/voice_transformer.yaml --quick
# Full local/model integration checks; generated audio only
$env:VOICE_INTEGRATION='1'
$env:TRANSFORMER_INTEGRATION='1'
& '..\term_project\Scripts\python.exe' -m pytest -q
```

For a new full PPO experiment, copy the voice config to a new filename, set new output/dataset directories
and the intended `nlp_model_dir`, then pass `--config`. Completed runs are protected from accidental training
overwrite. `--evaluate-only` explicitly replaces evaluation reports, not checkpoints. Set `model_dir` in
`configs/understanding.yaml` when deploying a new language checkpoint, and keep the selected voice config
paired with the same language model. Use `INSURANCE_NLP_BACKEND=nb` for an explicitly labeled baseline run.

Windows Application Control currently blocks a spaCy vectors DLL. Phrase-boundary regex rules and exact
numeric extraction remain active and the UI reports `Regex rules (spaCy unavailable)`. The spaCy path remains
available on compatible hosts. Transformers 4.44.2 is pinned to avoid importing a blocked, optional
scikit-learn DLL in a newer release. No Windows security control was disabled. CPU Torch fine-tuning and
ONNX emotion inference were actually executed; this is not a fallback represented as a transformer.
