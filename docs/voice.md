# Core Voice Agent

## Scope

This is a project-level voice implementation shared by the CLI and browser workspace. It uses pretrained
speech recognition and synthesis, then trains and evaluates the insurance agent on recognized speech.
The product catalogue is still synthetic. The project does not issue policies, calculate genuine insurer
quotes, collect payments, or make telephone calls.

The existing text dataset, PPO checkpoint and benchmark results are preserved. Voice artifacts are stored
under `results/voice_v2/`; generated speech is under `data/processed/voice_v2/`. Original v1 artifacts in
`results/voice/` and `data/processed/voice/` are preserved. The ASR model is on the project
drive under `.runtime/speech/`. No additional Llama model is downloaded by voice setup.

## Architecture

```text
Microphone (explicit bounded recording) or local PCM WAV
  -> audio-format and silence checks
  -> pretrained Vosk speech recognizer
  -> transcript, word confidence and spoken-number normalization
  -> retry if unclear, or continue with recognized text
  -> shared ConversationSession: NLP, entities, memory, eligibility and action mask
  -> speech-trained PPO action + explicit conversation constraints
  -> existing local Llama or template response generator
  -> existing response checks
  -> Windows speech synthesis to WAV
  -> local speaker playback and next buyer turn
```

PPO selects the conversational action. It does not generate audio and it does not fine-tune Llama.
The speech layer only speaks the checked response. If response validation fails, it speaks a fixed
clarification instead. If synthesis fails, the text response remains available with an explicit error.

## Models and Dependencies

| Component | Current implementation | Trained by this workflow? |
| --- | --- | --- |
| Speech recognition | Vosk `vosk-model-small-en-us-0.15` | No; pretrained weights |
| Spoken numbers | `text2num`, preserving the original transcript | No |
| Intent/emotion/objection/stage | Existing Naive Bayes architecture, adapted with speech transcripts | Yes |
| Sales action policy | Existing 94-feature, 16-action NumPy PPO | Yes, separate voice checkpoints |
| Response wording | Existing Llama 3.2 3B through Ollama, or templates | No Llama fine-tuning |
| Speech synthesis | Installed Microsoft Zira Desktop through Windows SAPI | No; pretrained system voice |

The emotion component analyzes words. It does not infer emotion from pitch, tone, rhythm or other audio
features. This first voice version supports English and a Windows TTS backend.

Sources: [official Vosk models](https://alphacephei.com/vosk/models),
[Vosk waveform example](https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py),
[sounddevice installation](https://python-sounddevice.readthedocs.io/en/latest/installation.html),
[Microsoft speech output API](https://learn.microsoft.com/en-us/dotnet/api/system.speech.synthesis.speechsynthesizer.setoutputtowavefile).

## Setup

```powershell
Set-Location 'D:\insurence seller\insurance_sales_agent'
& '..\term_project\Scripts\python.exe' -m pip install -r requirements-voice.txt
& '..\term_project\Scripts\python.exe' setup_voice.py --download-model
```

`setup_voice.py` downloads from the official Vosk model site, validates archive paths, and records the
source URL, license and downloaded archive SHA-256. It does not download a model during ordinary inference.
The hash is locally recorded provenance, not verification against a publisher-signed checksum.

Windows must have an English desktop speech voice installed. `configs/voice.yaml` selects its name and
rate. A missing voice is an error, not a silent substitution with another speech engine. Installations on
other operating systems need another TTS adapter; the text pipeline remains usable without audio packages.

## Training Workflow

```powershell
& '..\term_project\Scripts\python.exe' -u train_voice.py
```

1. Select unique customer utterances within each existing conversation split, up to the configured cap.
2. Generate local speech for those utterances using an installed system voice.
3. Generate the finite set of customer-simulator utterances at train/validation/test speech rates.
4. Transcribe the real generated WAV files with Vosk. Save hypotheses, confidence, timings and hashes.
5. Fit new insurance NLP models using original training text plus transcribed training speech. Do not use
   validation or test labels for fitting.
6. Create a speech-conditioned environment. Recognized text goes through the same state preparation used
   by live conversations. Trust, engagement, satisfaction and intent scores use live defaults rather than
   exposing hidden simulator values to the policy.
7. Initialize PPO from the existing supervised action weights and train 20,000 steps per seed by default.
8. Evaluate the new voice policy, the unchanged text-trained policy, and the rule policy on matching
   held-out simulator seeds using the same test-rate ASR channel and adapted NLP.
9. Save checkpoints, training histories, speech/NLP metrics, per-episode policy metrics and a report.

The simulator retains hidden state to compute reward and transitions. Known age and profile are treated
as enrollment information. Budget and need must come from recognized utterances. Action masks come from
the observable conversation state. Explicit task constraints are included in the training action mask,
so PPO records probabilities for the action actually executed.

The transcribed simulator phrases are cached before training. A PPO step uses their actual ASR hypotheses;
it does not repeatedly synthesize/transcribe the same phrase during every rollout. Missing cache entries
raise an error. They are never replaced silently with perfect reference text.

The original simulator's finite language and action-based reactions remain limitations. Its rewards are
not human judgments of spoken persuasiveness, and the simulator does not listen to the agent's audio.
Low-confidence retries advance a simulator turn; live retries deliberately leave conversation memory
unchanged. That difference is recorded in the report.

### Reuse and Smoke Runs

```powershell
# Reuse and validate already generated audio; retrain only voice artifacts.
& '..\term_project\Scripts\python.exe' -u train_voice.py --reuse-data

# Only prepare/transcribe speech and calculate ASR metrics.
& '..\term_project\Scripts\python.exe' train_voice.py --prepare-only

# Small pipeline test in separate voice_smoke folders; not a research benchmark.
& '..\term_project\Scripts\python.exe' train_voice.py --quick
```

The default training seeds are 42, 43 and 44, with 100 held-out episodes per seed and policy. The deployed
voice checkpoint is the first configured seed, not the seed that scored best on the test set. Configuration
changes are made in `configs/voice.yaml`. PPO optimization and reward parameters come from `configs/base.yaml`.

## Run a Conversation

### Local Llama Wording

```powershell
& '..\term_project\Scripts\python.exe' run_llm.py --voice --age 35 --debug
```

This reuses the existing Ollama installation and model storage. It starts the local server when needed.
The existing launcher can take time to initialize hardware; it does not pull a new model automatically.
This now defaults to hybrid wording: deterministic discovery/clarification/payment/exit replies and
compact local Llama requests for eligible open-ended wording. `--generator ollama` selects full LLM
wording. The hybrid socket timeout defaults to 20 seconds (`INSURANCE_VOICE_LLM_TIMEOUT`), but model
loading and processing mean that this is not a strict total-turn deadline. LLM failures remain visible.

### Template Wording

```powershell
& '..\term_project\Scripts\python.exe' voice.py --age 35 --debug
```

Press Enter to start each recording, speak, and wait for the reply. Vosk endpoint detection stops
capture after an utterance, with an eight-second maximum by default.
Use `/quit` before recording or Ctrl+C to exit. `--seconds 5` changes the capture duration. Microphone
capture finishes before generation and playback, preventing the agent's own reply from becoming buyer
input. `--fixed-window` disables endpoint detection. This is turn-based voice, without barge-in,
wake words or full-duplex audio.

### Integrated Browser Workspace

```powershell
& '..\term_project\Scripts\python.exe' demo_server.py
```

Open `http://127.0.0.1:8765`. Voice-trained + Hybrid uses the same configured NLP/PPO artifacts as
`voice.py`, via `src/voice_runtime.py`. The text-trained pipeline remains selectable. Changing the
pipeline, buyer profile, policy or generator starts a new session after confirmation.

- The microphone button requests browser permission and starts explicit recording. Stop sends it;
  discard cancels. Capture ends at a ten-second maximum. Browser recording uses a bounded manual
  window, not the CLI's streaming Vosk endpoint detector.
- Web Audio's resampler converts captured audio to 16 kHz mono PCM WAV. `/audio` passes it through
  the existing `VoiceSession` and local Vosk. No browser cloud ASR is used.
- The WAV upload control accepts the same format, at most 30 seconds. Malformed uploads are rejected
  before ASR or conversation state changes. Low-confidence/silent input does not advance a turn.
- Typed and spoken inputs share the same session memory. Successful voice turns record the original
  and normalized transcript, ASR confidence and timings. A recognized stop closes the conversation.
- Spoken replies are opt-in. `/speech` synthesizes only an existing checked response through Windows
  SAPI; the caller cannot submit arbitrary synthesis text. Playback failures leave the text intact.
- Microphone tracks close on stop, cancel, failure, page hide and navigation. Temporary WAVs are
  removed after each API call. Session transcripts remain in memory until expiry/restart; exporting
  JSON is an explicit action. Raw audio is not retained or automatically used for training.
- Request IDs and input fingerprints prevent a retried upload from creating a second buyer turn.
  Per-session locks and a bounded speech worker reject overlapping work with an explicit busy error.
- The Results view separates text and voice experiments, shows saved provenance, seed spread, ASR
  word errors, violations and missed stops, and does not imply a real-world conversion rate.

`tests/test_ui_voice.py` covers the shared API contract and failure paths; `tests/demo_browser.cjs`
checks desktop/mobile layouts and synthetic browser capture with muted playback. Neither is evidence
that a real person's microphone, accent or room acoustics has passed validation.

Suggested first sequence: "I need health insurance for my family", "around two hundred fifty thousand",
"what will be my payment", and "please stop". Speech recognition may make errors; inspect the displayed
transcript. Unknown amounts are clarified instead of treated as verified premiums.

### WAV Input

```powershell
& '..\term_project\Scripts\python.exe' voice.py --age 35 --wav buyer.wav --output response.wav --no-play --debug
```

Input must be uncompressed, mono, 16-bit PCM at 16 kHz, no longer than 30 seconds. Unsupported formats are
rejected explicitly. Use a trusted audio converter for other formats. `--no-play` still generates response
audio but does not play it. `--output` explicitly preserves response WAVs; interactive files receive a
turn suffix. Otherwise temporary input/output audio is removed at the end of each turn.

### Device Selection

```powershell
& '..\term_project\Scripts\python.exe' voice.py --list-devices
& '..\term_project\Scripts\python.exe' voice.py --age 35 --input-device 1 --output-device 3
```

Device numbers are machine-specific. Allow microphone access in Windows privacy settings when necessary.
Neither training nor tests record the microphone. Native integration tests synthesize audio directly to
files and do not play it aloud.

## Evaluation and Evidence

`results/voice_v2/report.md` summarizes the current completed run. `speech_metrics.json` records corpus-level
word error rate (total word edits / total reference words), character error rate, exact matches and failed
transcriptions. WER can exceed 100% when there are many insertions. Punctuation and case are normalized;
this is not semantic equivalence or validation of extracted money amounts.

`nlp_metrics.json` compares original and adapted NLP predictions on held-out recognized text. Empty
hypotheses remain evaluation failures rather than being silently excluded. `evaluation.json` and
`episodes.csv` contain separate simulator results; they do not replace the earlier text benchmark.

### Current v2 Run: 8 October 2026

- 237 synthetic speech files, 79 per speech-rate split; test WER 4.67%.
- Three PPO seeds with 20,000 training steps each. Evaluation includes 1,200 conversations across
  new voice PPO, original text PPO, prior voice PPO and the rule baseline.
- Mean reward: new voice PPO 5.9363, prior voice PPO 5.9690, text PPO 5.8260 and rules 4.7697.
  Simulated conversion: 11.33%, 11.33%, 12.00% and 11.67%, respectively.
- Observable simulator utterances now expose existing objections/readiness. The shared live state
  recognizes affirmative interest and masks commitment without that signal. The prior voice policy
  matches the retrained policy's conversion in the new environment, so this is not evidence that
  retraining alone caused the increase from v1's zero conversion. Original reward weights and the
  simulator purchase threshold are unchanged. Violations and missed stop turns remain nonzero.
- `validate_voice.py --with-llm` records a tiny eight-phrase clean/white-noise test and processing
  timings. Four hybrid dialogue turns took about 1.55-1.80 seconds, all deterministic templates.
  The separate open-ended Llama attempt timed out and fell back after 22.13 seconds. Do not describe
  the deterministic timings as an LLM speed improvement. Timings exclude recording and playback.
- Re-evaluate saved policies without retraining with `train_voice.py --evaluate-only`.
- `microphone_check.py` is the remaining opt-in human test, not part of unattended automation.

### Historical v1 Run: 8 October 2026

The following describes the original environment and earlier code, not the current runtime:

- Generated and transcribed 222 synthetic audio files, 74 per speech-rate split.
- Trained three voice PPO checkpoints, each for 20,000 steps, and evaluated 900 conversations in total
  across the voice PPO, text-trained PPO and rule policies.
- Test-rate word error: 5.01%. This is a synthetic system-voice measurement, not human ASR accuracy.
- Mean simulator reward: voice PPO 6.4493, text-trained PPO 6.0613, rule policy 4.8287. All recorded
  simulated conversion rates were 0%. Higher reward therefore does not demonstrate successful selling.
- The rule policy's commitment branch expects a `COMMITMENT` stage, while the shared live stage rules
  do not currently produce that stage. This is a structural limitation of that live-state baseline.
  Understanding the learned policies' lack of conversion needs action/reward analysis rather than
  assuming speech recognition caused it.
- Completed an actual WAV -> Vosk -> trained PPO -> local Llama -> response WAV turn. Llama was the
  response source, with no fallback or response-check rejection. End-to-end processing took 52.6 seconds
  in this individual run. A template-wording run took 4.5 seconds. These are observations, not benchmarks.
- The 64-test suite passed with native integration enabled, including a four-turn audio conversation.
- Detected available microphone and speaker devices, but did not record a human microphone test.
- Verified the original text PPO checkpoint's SHA-256 was unchanged by the voice experiment.

Important limits:

- Speech is generated by one system voice. Different speaking rates are not independent human speakers.
- Conversation splits are inherited, but short synthetic utterance templates can recur across splits.
- Simulator utterances intentionally repeat with channel variations. This is not unseen-language evaluation.
- ASR word confidence is not calibrated probability of understanding; the 0.55 retry threshold is a heuristic.
- A recognized stop request overrides the retry gate, but an ASR system can still miss a spoken rejection.
- Human microphone tests, noise/accent variation, amount accuracy and stop-request recall require a separate
  consented dataset before claiming deployment readiness.
- No speech model, Llama model, real-customer reward model, or acoustic-emotion model is trained here.
- Real insurance selling remains outside the implemented quote/issuance/payment scope.

## Tests

```powershell
& '..\term_project\Scripts\python.exe' -m pytest tests -q

# Include actual Windows TTS -> WAV -> Vosk integration (no microphone recording).
$env:VOICE_INTEGRATION = '1'
& '..\term_project\Scripts\python.exe' -m pytest tests/test_voice.py -q
Remove-Item Env:VOICE_INTEGRATION
```

Coverage includes audio validation, low-confidence and silent input, stop requests, guarded spoken output,
spoken numbers, interrupted microphone cleanup, core state reuse, masked actions, PPO checkpoint round trips,
word-error calculations and native multi-file synthesis/recognition.
