# Insurance Sales Conversational RL Agent
Spec: `../Insurance_Sales_RL_Agent_Project_Specification.md`
Architecture: `UNDERSTAND -> STATE -> PPO(action) -> STRATEGY -> VERIFY -> GENERATE -> SIMULATE -> REWARD -> LEARN`.
PPO chooses the action, the product engine controls eligible facts, and templates or a local LLM verbalize
the assigned strategy. This is an academic prototype with synthetic products and simulated customers.

## Setup (venv `term_project`)
```powershell
Set-Location 'D:\insurence seller\insurance_sales_agent'
& '..\term_project\Scripts\python.exe' -m pip install -r requirements.txt
& '..\term_project\Scripts\python.exe' -m pip install -r requirements-understanding.txt -r requirements-voice.txt
```
The existing `term_project` environment is reused. On a new machine create it with `python -m venv ../term_project`.

## Run the Project

The trained text/voice checkpoints and installed models on this machine are ready to use. Training is
not required each time you start the project. For the integrated browser workspace:

```powershell
Set-Location 'D:\insurence seller\insurance_sales_agent'
& '..\term_project\Scripts\python.exe' demo_server.py
```

Open [the local workspace](http://127.0.0.1:8765). Voice-trained + Hybrid is selected when the voice
artifacts are available. Type a buyer message or click the microphone, grant access, speak, and click
stop to get the agent's answer directly. **Spoken replies** is on by default for voice turns; typed turns
stay silent. Recording ends automatically at 10 seconds; the X discards it. Enable **Review speech**
only when you want to correct the transcript before confirming with Send. The inspector shows waveform
vocal emotion first for voice turns, with transcript emotion in separate expandable details. Intent is
inferred from recognized words. Disable **Spoken replies** for silent responses, or use a reply's speaker
button for manual playback. This uses the actual trained project, not a
separate mock agent. MiniLM + RoBERTa is the default classifier. The old Naive Bayes baseline and
text-trained policy remain selectable. Restart the server and start a new session after code updates.

## Transformer and Acoustic Understanding

MiniLM-L6 is fully fine-tuned for nine intents (softmax) and ten simultaneous objections (independent
sigmoids; an empty set means NONE). RoBERTa GoEmotions supplies text-emotion scores with an explicit,
heuristic mapping to the eight project categories. It is not an eight-category fine-tuned emotion model.
The waveform also goes to pretrained wav2vec2 emotion recognition before the temporary audio is deleted.
Its four labels and uncertainty stay separate from text emotion and never supply consent or PPO state.

Models are already installed and trained here. On a fresh machine:

```powershell
& '..\term_project\Scripts\python.exe' setup_understanding.py
& '..\term_project\Scripts\python.exe' train_understanding.py
& '..\term_project\Scripts\python.exe' train_voice.py --config configs/voice_transformer.yaml
```

Downloads and hashes live under `.runtime/understanding/` on the project drive. No cloud inference or
remote model code is used. Fine-tuned language artifacts: `results/understanding_v1/`. New PPO experiment:
`results/voice_v3_transformer/` (three seeds, 20,000 steps each). These are separate from all old artifacts.
Completed training outputs are protected: choose a new output directory/config for another run.

The independent-sentence synthetic check contains only 39 intent/objection cases and 32 emotion cases:
intent accuracy 87.2%, objection micro-F1 0.818, mapped emotion macro-F1 0.787. This is not human validation.
The new PPO run did not outperform the prior voice checkpoint. See the UI's separate language and policy
results, and [model/training details](docs/understanding.md), for limitations and exact commands.

Entity extraction uses spaCy EntityRuler plus exact-value regex rules when available. On this host Windows
Application Control blocks a spaCy DLL, so the visible regex-only fallback is active. PyTorch works;
Transformers 4.44.2 avoids an unused scikit-learn import that was also blocked. No security setting was changed.

For regression tests, retraining, or the text CLI:
```powershell
& '..\term_project\Scripts\python.exe' -m pytest tests -q
& '..\term_project\Scripts\python.exe' chat.py --age 35 --debug
```
Use `chat.py --policy rule` before PPO training. A one-shot structured response is available with
`chat.py --message "I already have a 5 lakh policy." --age 35`. Stop requests end the session.
An eligible age and identified need are required before the live session offers product options.

The historical `run_all.py` experiment trains 20,000 steps for each of three independent PPO seeds and retrains six ablations.
`--quick` is a small smoke run that overwrites artifacts and is not a full research result.
`configs/base.yaml` is the canonical training/data/reward configuration; `configs/evaluation.yaml` sets
evaluation seeds and episode counts. The separate data/PPO config files document the corresponding settings.

Outputs include `results/reports/report.md`, per-seed metrics and confidence intervals,
`results/trajectories/evaluation.jsonl`, `experiments/*/run.json`, checkpoints and reward-curve PNGs.
`results/checkpoints/ppo.npz` is the first seed's checkpoint used by the live text session.

## Components
- Synthetic `insurance_sales_dialogue_dataset v2`: 1,200 conversations, 20,396 turns, 3,000 profiles.
- Scenario/transcript grouping, provenance, validation, and derived objection/transition datasets.
- Intent, emotion, objection, need/entity, and sales-stage components.
- A normalized 94-feature state, action masks, customer simulator and multi-objective reward.
- Product eligibility by age/need/budget, numeric fact checks and repetition checks.
- Masked stochastic PPO training with GAE, clipped objective, Adam and supervised initialization.
- Random, rule, supervised and PPO baselines, independent seed runs and retrained ablations.
- Interactive text conversations, verified templates, optional Ollama generation and direct-LLM policy.
- Public NLP corpus readers, schemas, model/dataset cards and experiment logs.

## Core Voice Project

The voice implementation is shared with the browser workspace. `voice.py` runs local microphone
or WAV input through Vosk, the shared conversation engine, the trained voice PPO checkpoint and Windows
speech synthesis. Speech-model weights are pretrained. `train_understanding.py` fine-tunes MiniLM;
`train_voice.py` uses that model and transcribed synthetic speech to train PPO. Text checkpoints and
baseline reports remain unchanged.

```powershell
Set-Location 'D:\insurence seller\insurance_sales_agent'
& '..\term_project\Scripts\python.exe' -m pip install -r requirements-voice.txt
& '..\term_project\Scripts\python.exe' setup_voice.py --download-model
& '..\term_project\Scripts\python.exe' voice.py --age 35
```

Launch spoken conversation with the already installed local Llama model:

```powershell
& '..\term_project\Scripts\python.exe' run_llm.py --voice --age 35 --debug
```

The launcher defaults to hybrid generation for voice: discovery, money clarification and payment
limitations use deterministic replies; other eligible wording may use Llama. These planned templates
are not LLM failures. `--generator ollama` requests full LLM wording and can be slower. The voice LLM
socket timeout defaults to 20 seconds, not a guaranteed total-turn deadline; failure is labelled.

Press Enter before each bounded microphone recording. Vosk endpoint detection can stop it early;
`--fixed-window` on `voice.py` disables this. `/quit` or Ctrl+C stops. The microphone is closed
before generation and playback; this is turn-based conversation, not simultaneous speech or telephone calling.
Silence/low-confidence audio prompts a retry without changing buyer memory. Recognized stop requests
end the conversation. Input audio is temporary and is not automatically used for training.

The current configured experiment produces `results/voice_v3_transformer/report.md`, language and word-error metrics, per-episode policy
results, and separate checkpoints. Its speech data is synthetic system-voice audio, not evidence of human
buyer performance. Setup, architecture, training details and limitations: [Voice project guide](docs/voice.md).

## Demonstration UI
```powershell
Set-Location 'D:\insurence seller\insurance_sales_agent'
& '..\term_project\Scripts\python.exe' demo_server.py
```
Open [the local demo](http://127.0.0.1:8765). The server binds to the local machine only.
Use `--port 8767` if the default port is occupied. Install the understanding and voice requirements above on a new machine.

The UI includes buyer presets, separate text/voice-trained pipelines, PPO/rule policies, hybrid/LLM/template
generation, microphone or WAV input, checked-response playback, and a turn-by-turn decision inspector.
The inspector records generation route, LLM timing, classifier source, all objections, mapped text emotion,
independent vocal-emotion scores, exact entities, ASR transcript confidence and shared buyer memory.
The catalogue, separate text/voice evaluations and JSON transcript export use the real project artifacts.
Transformer voice paths resolve from `configs/voice_transformer.yaml`; the NB baseline uses `configs/voice.yaml`.
The CLI and UI share this selection. Evaluation provenance is taken from
the saved run, not inferred from current settings. The UI does not start training automatically.

Browser capture uses Web Audio to produce 16 kHz mono PCM WAV, processed by local Vosk, not a cloud
speech-recognition service. Recording requires a click and browser permission. Tracks close before
processing, on discard, on page hiding and on error. Uploaded WAVs must use the same PCM format, up to
30 seconds. Input files are temporary; transcripts remain in session memory and explicit JSON exports.
Speech playback uses Windows SAPI and speaks only a stored checked response. Audio is not used for training.
Profile changes apply to a new session. Explicit rejection closes the conversation. Template fallbacks
and NLP estimates are labelled; the product-claim guard is lexical, not a guarantee of factual accuracy.
The live dialogue layer also uses explicit request cues and transparent action constraints. It clarifies
unlabelled amounts instead of guessing a budget band, and explains unavailable payment quotes rather
than inventing a price. The inspector records the original policy proposal and any dialogue override.
A buyer-supplied number may be referenced when clarifying its meaning, but is not treated as a premium quote.
Explicit stop requests alone permit conversation closure. Requests to answer questions one by one activate
guided intake: need, age, coverage, existing policy and budget band, then review and a handoff summary.
The summary is not an issued policy, price quote, payment or completed sale.

The server starts the project-local Ollama executable if needed. Hardware detection can take over a
minute; model status updates automatically. `--no-start-ollama` leaves Ollama untouched. Local LLM
defaults to `llama3.2:3b`, configurable with `INSURANCE_LLM_MODEL`. Template mode needs no LLM.
Sessions stay in server memory for up to an hour of inactivity and are lost when the server restarts.
Keep this research demo on localhost; it does not provide production authentication or policy issuance.
Lucide 1.8.0 is vendored under `web/vendor/` with its license, so the UI needs no CDN or build step.

### Verification

```powershell
$env:VOICE_INTEGRATION='1'
$env:TRANSFORMER_INTEGRATION='1'
& '..\term_project\Scripts\python.exe' -m pytest tests -q
& '..\term_project\Scripts\python.exe' validate_voice.py --with-llm
& '..\term_project\Scripts\python.exe' microphone_check.py
```

The first two commands use generated audio, not a microphone. `microphone_check.py` is the separate
opt-in human check and waits for Enter before recording. Browser regression tests are in
`tests/demo_browser.cjs`; run with Playwright on `NODE_PATH` and `DEMO_URL` pointing to the running UI.
They use Chromium fake audio capture and muted playback, never the real microphone. The generated WAV
fixture defaults to `.runtime/speech/validation/clean-0.wav` (created by `validate_voice.py`), overridable
with `DEMO_VOICE_WAV`. See `AGENTS.md` for the project rule requiring UI/API/tests/docs to stay in sync.

## Optional LLM
For this machine, see [Windows setup and PATH troubleshooting](docs/ollama_setup.md).
After the local installation and model download, `python run_llm.py --age 35 --debug` starts the LLM chat.

Set `INSURANCE_LLM_MODEL` to the exact name of an already installed Ollama model and run
`chat.py --generator ollama`. `INSURANCE_LLM_URL` defaults to `http://localhost:11434/api/chat`.
An invalid strategy or unsupported response falls back to verified templates. The direct-LLM experiment
is reported as skipped when no model is configured. It is never replaced with a heuristic labelled as LLM.
The adviser prompt uses actual user/assistant history and a constrained JSON strategy schema. Input
context is compact, output is capped, and `INSURANCE_LLM_KEEP_ALIVE` defaults to `5m` to avoid reloading
the model between nearby turns. Rejection reasons and connection errors are visible in the inspector.
Response checks cover numeric claims, recent literal repetition, buyer-role cues and selected dialogue
requirements; they are not a full semantic or regulatory validation system. Exact premium payments
require a real, verified pricing source, which the synthetic catalogue does not contain.

## Research Limits
The default run uses only synthetic and simulated evidence. Public corpus metadata in `data/raw/` is
not a downloaded example. Actual local MultiDoGO, GoEmotions and MELD data can be normalized with
`python -m src.external_data`; see [experiment instructions](docs/experiments.md).
PPO remains implemented in NumPy. MiniLM uses the working PyTorch CPU runtime, and emotion inference uses
ONNX Runtime. Some unrelated/native dependency DLLs remain blocked as documented above.
This project does not establish real-world conversion performance.

Details: [methodology](docs/methodology.md), [dataset card](docs/dataset_card.md),
[model card](docs/model_card.md), [experiment instructions](docs/experiments.md),
[generated research report](results/reports/report.md).
