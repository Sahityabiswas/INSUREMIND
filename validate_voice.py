"""Reproducible synthetic-audio robustness and latency checks; never records a microphone."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from src.common import load_config
from src.conversation import ConversationSession
from src.generation import OllamaClient, ResponseGenerator
from src.nlp import live_cues, stop_requested
from src.speech import ROOT, NoSpeechError, VoskASR, VoiceSession, WindowsTTS, read_wav, write_wav
from src.voice_training import load_predictor, speech_metrics, write_json

PHRASES = ["I need health insurance for my family.", "Around two hundred fifty thousand.",
           "What will be my payment?", "I am ready to proceed.", "I am not ready to proceed.",
           "Please stop.", "I am not interested.", "That is too expensive for my budget."]


def add_noise(pcm, snr_db, seed):
    samples = np.frombuffer(pcm, dtype="<i2").astype(float)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    noise = np.random.default_rng(seed).normal(size=len(samples))
    noise *= rms / (10 ** (snr_db / 20)) / max(float(np.sqrt(np.mean(noise ** 2))), 1e-12)
    mixed = samples + noise
    peak = max(float(np.max(np.abs(mixed))), 1)
    if peak > 32767:
        mixed *= 32767 / peak
    return np.round(mixed).astype("<i2").tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/voice.yaml"))
    parser.add_argument("--with-llm", action="store_true", help="Also call the installed local Llama server")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ROOT / cfg["training"]["output_dir"]
    folder = ROOT / ".runtime/speech/validation"
    folder.mkdir(parents=True, exist_ok=True)
    tts = WindowsTTS(cfg["tts"]["voice"], cfg["tts"]["rate"])
    asr = VoskASR(ROOT / cfg["asr"]["model_path"])
    predictor = load_predictor(output / "nlp_models.json")
    paths = [folder / f"clean-{i}.wav" for i in range(len(PHRASES))]
    tts.synthesize_many([{"text": text, "output": path} for text, path in zip(PHRASES, paths)])
    robustness, details = {}, []
    for label, snr in (("clean", None), ("noise_20db", 20), ("noise_10db", 10)):
        rows = []
        for i, (reference, clean) in enumerate(zip(PHRASES, paths)):
            audio = clean
            if snr is not None:
                audio = folder / f"{label}-{i}.wav"
                pcm, _ = read_wav(clean)
                write_wav(audio, add_noise(pcm, snr, i))
            try:
                transcript = asdict(asr.transcribe(audio))
            except NoSpeechError as exc:
                transcript = {"text": "", "normalized_text": "", "confidence": 0, "asr_error": str(exc)}
            row = {"text": reference, "transcript": transcript, "condition": label,
                   "stop_expected": stop_requested(reference), "stop_detected": stop_requested(transcript["normalized_text"]),
                   "ready_expected": live_cues(reference)["ready_to_proceed"],
                   "ready_detected": live_cues(transcript["normalized_text"])["ready_to_proceed"]}
            rows.append(row)
        expected_stops = sum(row["stop_expected"] for row in rows)
        robustness[label] = {**speech_metrics(rows),
            "stop_recall": sum(r["stop_expected"] and r["stop_detected"] for r in rows) / expected_stops,
            "stop_examples": expected_stops,
            "false_ready": sum(r["ready_detected"] and not r["ready_expected"] for r in rows)}
        details.extend(rows)
        print(label, robustness[label], flush=True)
    latency = []
    for provider in (["template", "hybrid"] if args.with_llm else ["template"]):
        conversation = ConversationSession("ppo", provider, age=35,
            checkpoint=output / "checkpoints/ppo_voice.npz", predictor=predictor)
        session = VoiceSession(conversation, asr, tts, cfg["asr"]["min_confidence"])
        for i in (0, 1, 2, 5):
            result = session.reply_audio(paths[i], folder / f"reply-{provider}-{i}.wav")
            latency.append({"provider": provider, "reference": PHRASES[i], **result})
            print(provider, result["timing"], (result["response"] or {}).get("generation_route"), flush=True)
    open_reply = None
    if args.with_llm:
        client = OllamaClient(os.environ.get("INSURANCE_LLM_MODEL", "llama3.2:3b"), profile="voice")
        start = time.perf_counter()
        open_reply = ResponseGenerator("hybrid", client).generate("HANDLE_OBJECTION",
            {"need": "FAMILY_HEALTH", "age": 35}, customer_text="I am worried about policy exclusions.")
        open_reply["wall_seconds"] = time.perf_counter() - start
        print("Open-ended generator:", open_reply["source"], open_reply["wall_seconds"], open_reply["llm_error"], flush=True)
    report = {"protocol": "Synthetic system-voice WAVs; no microphone/speaker playback. Timings exclude recording and ASR model loading.",
              "noise": "Deterministic additive white noise; not a substitute for accents, room noise or human speech.",
              "robustness": robustness, "recognitions": details, "latency": latency,
              "open_ended_generator_only": open_reply,
              "scripted_audio_checks_passed": all(r["accepted"] and r["audio_path"] and not r["tts_error"]
                  and r["response"]["validation"]["ok"] for r in latency)}
    write_json(output / "voice_validation.json", report)
    print("Saved:", output / "voice_validation.json", flush=True)
    if not report["scripted_audio_checks_passed"]:
        raise SystemExit("A scripted voice check failed; inspect the report")


if __name__ == "__main__":
    main()
