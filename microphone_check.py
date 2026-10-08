"""Opt-in human microphone check; no recording before explicit Enter for each phrase."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from src.common import load_config
from src.nlp import extract_needs, live_cues, stop_requested
from src.speech import ROOT, NoSpeechError, SpeechError, VoskASR
from src.voice_training import speech_metrics, write_json

PHRASES = ["I need health insurance for my family.", "Around two hundred fifty thousand.",
           "I am not ready to proceed.", "Please stop. I am not interested."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/voice.yaml"))
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    cfg = load_config(args.config)
    asr = VoskASR(ROOT / cfg["asr"]["model_path"])
    folder = ROOT / ".runtime/speech"
    folder.mkdir(parents=True, exist_ok=True)
    output = ROOT / cfg["training"]["output_dir"] / "microphone_check.json"
    print("Read the displayed fictional phrases. Do not include personal information.")
    print("Each recording requires Enter. /quit or Ctrl+C cancels. Audio clips are deleted.")
    print(f"Recognized text and test results will be saved locally to {output}")
    rows = []
    try:
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            for index, reference in enumerate(PHRASES):
                print(f"\nPhrase {index + 1}: {reference}")
                if input("Press Enter when ready to speak, or /quit: ").strip().lower() == "/quit":
                    break
                print("Recording now...")
                path = Path(tmp) / "microphone.wav"
                asr.record_utterance(path, args.seconds, args.input_device)
                try:
                    transcript = asdict(asr.transcribe(path))
                except NoSpeechError as exc:
                    transcript = {"text": "", "normalized_text": "", "confidence": 0, "asr_error": str(exc)}
                print("Heard:", transcript["normalized_text"] or "[no speech recognized]")
                rows.append({"text": reference, "transcript": transcript})
    except (KeyboardInterrupt, EOFError):
        print("\nMicrophone check cancelled.")
    except SpeechError as exc:
        parser.exit(1, f"Microphone error: {exc}\n")
    if not rows:
        print("No recordings completed; no report saved.")
        return
    complete = len(rows) == len(PHRASES)
    transcript = lambda i: rows[i]["transcript"]["normalized_text"]
    checks = {"all_phrases_completed": complete,
              "need_recognized": "FAMILY_HEALTH" in extract_needs(transcript(0))["needs"],
              "amount_recognized": live_cues(transcript(1))["amount"] == 250000 if len(rows) > 1 else False,
              "negation_preserved": ("not" in transcript(2).split() and not live_cues(transcript(2))["ready_to_proceed"]) if len(rows) > 2 else False,
              "stop_recognized": stop_requested(transcript(3)) if complete else False,
              "all_confident": complete and all(row["transcript"]["confidence"] >= cfg["asr"]["min_confidence"] for row in rows)}
    write_json(output, {"source_type": "human_microphone", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "metrics": speech_metrics(rows), "checks": checks, "passed": all(checks.values()), "rows": rows})
    print("Check passed:" if all(checks.values()) else "Check needs review:", checks)
    print("Saved:", output)


if __name__ == "__main__":
    main()
