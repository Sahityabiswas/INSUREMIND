"""Core voice client: local microphone/WAV -> ASR -> trained agent -> local WAV/playback."""
import argparse
import json
from pathlib import Path
import tempfile

from src.voice_runtime import voice_artifacts, voice_conversation
from src.speech import ROOT, SpeechError, VoskASR, VoiceSession, WindowsTTS, play_audio, record_microphone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/voice.yaml"))
    parser.add_argument("--wav", type=Path, help="Process an existing 16 kHz mono PCM WAV instead of the microphone")
    parser.add_argument("--output", type=Path, help="Persist response audio; otherwise temporary files are removed")
    parser.add_argument("--generator", choices=["template", "ollama", "hybrid"], default="template")
    parser.add_argument("--policy", choices=["ppo", "rule"], default="ppo")
    parser.add_argument("--checkpoint", type=Path, help="Explicit alternative PPO checkpoint")
    parser.add_argument("--nlp-model", type=Path, help="Explicit alternative voice NLP bundle")
    parser.add_argument("--age", type=int)
    parser.add_argument("--budget", choices=["unknown", "low", "mid", "high"], default="unknown")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--fixed-window", action="store_true", help="Disable automatic end-of-utterance detection")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-play", action="store_true", help="Do not send output audio to speakers")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice
        print(sounddevice.query_devices())
        return
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    config, _, _, _ = voice_artifacts(args.config)
    try:
        session = voice_conversation(config_path=args.config, checkpoint=args.checkpoint,
                                     nlp_model=args.nlp_model, policy=args.policy, generator=args.generator,
                                     age=args.age, budget=args.budget)
    except RuntimeError as exc:
        parser.error(str(exc))
    cfg = config["asr"]
    voice = VoiceSession(session, VoskASR(ROOT / cfg["model_path"], cfg["max_audio_seconds"]),
                         WindowsTTS(config["tts"]["voice"], config["tts"]["rate"]), cfg["min_confidence"])
    temporary_root = ROOT / ".runtime/speech"
    temporary_root.mkdir(parents=True, exist_ok=True)
    print("Local voice research agent. Synthetic products; no real quote, issuance or payment.")
    print("Audio is processed locally. Microphone capture starts only after Enter. Ctrl+C or /quit stops.")
    attempt = 0
    try:
        while not session.closed:
            with tempfile.TemporaryDirectory(dir=temporary_root) as tmp:
                incoming = args.wav or Path(tmp) / "customer.wav"
                if not args.wav:
                    if input("Enter to record, /quit to exit: ").strip().lower() == "/quit":
                        break
                    print(f"Recording now (maximum {args.seconds:g} seconds)...")
                    if args.fixed_window:
                        record_microphone(incoming, args.seconds, args.input_device)
                    else:
                        voice.asr.record_utterance(incoming, args.seconds, args.input_device)
                attempt += 1
                if args.output:
                    suffix = f"-{attempt:03d}" if not args.wav else ""
                    outgoing = args.output.with_name(args.output.stem + suffix + ".wav")
                else:
                    outgoing = Path(tmp) / "agent.wav"
                result = voice.reply_audio(incoming, outgoing)
                print("Heard:", (result["transcript"] or {}).get("normalized_text", "[no speech]"))
                print("Agent:", result["spoken_text"])
                if result["tts_error"]:
                    print("Speech output unavailable:", result["tts_error"])
                if args.debug:
                    print(json.dumps(result, indent=2))
                if result["audio_path"] and not args.no_play:
                    play_audio(result["audio_path"], args.output_device)
                if args.wav:
                    break
    except (KeyboardInterrupt, EOFError):
        print("\nVoice session stopped.")
    except (SpeechError, OSError) as exc:
        parser.exit(1, f"Voice error: {exc}\n")


if __name__ == "__main__":
    main()
