"""Local audio adapters and a voice session using the project's real conversation core."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000


class SpeechError(RuntimeError):
    pass


class NoSpeechError(SpeechError):
    pass


def read_wav(path, max_seconds=30):
    try:
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
                raise SpeechError("Audio must be uncompressed 16 kHz mono 16-bit PCM WAV")
            frames = audio.getnframes()
            if not 0 < frames <= SAMPLE_RATE * max_seconds:
                raise SpeechError(f"Audio must contain between 0 and {max_seconds} seconds")
            pcm = audio.readframes(frames)
            if len(pcm) != frames * 2:
                raise SpeechError("Truncated WAV audio")
            return pcm, frames / SAMPLE_RATE
    except (wave.Error, EOFError) as exc:
        raise SpeechError(f"Invalid WAV file: {exc}") from exc


def write_wav(path, pcm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm)


def normalize_spoken_text(text):
    from text_to_num import alpha2digit
    text = alpha2digit(text.strip(), "en", threshold=0)
    # Keep the numeric value and the original transcript; do not infer premium units.
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class Transcript:
    text: str
    normalized_text: str
    confidence: float
    audio_seconds: float
    asr_seconds: float
    model: str


class VoskASR:
    def __init__(self, model_path=None, max_seconds=30):
        path = Path(model_path or ROOT / ".runtime/speech/vosk-model-small-en-us-0.15").resolve()
        if not (path / "am/final.mdl").is_file():
            raise SpeechError(f"Speech model missing at {path}. Run python setup_voice.py --download-model")
        try:
            import vosk
            vosk.SetLogLevel(-1)
            self.model = vosk.Model(str(path))
        except (ImportError, OSError) as exc:
            raise SpeechError(f"Cannot load Vosk. Install requirements-voice.txt: {exc}") from exc
        self.model_name, self.max_seconds = path.name, max_seconds

    def record_utterance(self, path, max_seconds=8, device=None):
        """Use Vosk's speech endpoint detector, with an explicit maximum capture window."""
        import sounddevice as sd
        from vosk import KaldiRecognizer
        if not 1 <= max_seconds <= min(30, self.max_seconds):
            raise ValueError("Recording duration must be within the configured audio limit")
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        chunks, frames = [], 0
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device,
                                   blocksize=1600) as stream:
                while frames < int(max_seconds * SAMPLE_RATE):
                    size = min(1600, int(max_seconds * SAMPLE_RATE) - frames)
                    data, overflow = stream.read(size)
                    if overflow:
                        raise SpeechError("Microphone audio overflowed; please retry with other applications closed")
                    chunk = bytes(data)
                    chunks.append(chunk)
                    frames += size
                    if recognizer.AcceptWaveform(chunk) and json.loads(recognizer.Result()).get("text"):
                        break
        except Exception as exc:
            raise SpeechError(f"Microphone capture failed: {exc}") from exc
        write_wav(path, b"".join(chunks))
        return frames / SAMPLE_RATE

    def transcribe(self, path):
        from vosk import KaldiRecognizer
        pcm, duration = read_wav(path, self.max_seconds)
        if np.max(np.abs(np.frombuffer(pcm, dtype="<i2").astype(float))) < 32:
            raise NoSpeechError("No audible speech detected")
        start = time.perf_counter()
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        recognizer.SetWords(True)
        parts, words = [], []
        for offset in range(0, len(pcm), 8000):
            if recognizer.AcceptWaveform(pcm[offset:offset + 8000]):
                result = json.loads(recognizer.Result())
                parts.append(result.get("text", ""))
                words.extend(result.get("result", []))
        result = json.loads(recognizer.FinalResult())
        parts.append(result.get("text", ""))
        words.extend(result.get("result", []))
        text = " ".join(part for part in parts if part).strip()
        if not text:
            raise NoSpeechError("The recognizer could not identify speech; please repeat")
        return Transcript(text, normalize_spoken_text(text),
                          float(np.mean([w["conf"] for w in words])) if words else 0.0,
                          duration, time.perf_counter() - start, self.model_name)


class WindowsTTS:
    def __init__(self, voice="Microsoft Zira Desktop", rate=0):
        if os.name != "nt":
            raise SpeechError("The installed TTS backend uses Windows SAPI")
        if not -10 <= rate <= 10:
            raise ValueError("TTS rate must be between -10 and 10")
        self.voice, self.rate = voice, rate

    def synthesize_many(self, jobs):
        payload = []
        for job in jobs:
            if not isinstance(job["text"], str) or not job["text"].strip() or len(job["text"]) > 8000:
                raise SpeechError("TTS needs 1-8000 characters of plain text")
            output = Path(job["output"]).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            payload.append({"text": job["text"], "output": str(output),
                            "voice": job.get("voice", self.voice), "rate": job.get("rate", self.rate)})
        if not payload:
            return
        folder = ROOT / ".runtime/speech"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            manifest = Path(tmp) / "jobs.json"
            manifest.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
            powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
            try:
                completed = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-File",
                    str(Path(__file__).with_name("speech_sapi.ps1")), "-Jobs", str(manifest)],
                    capture_output=True, text=True, timeout=max(60, len(payload) * 15),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SpeechError(f"Windows synthesis unavailable: {exc}") from exc
            if completed.returncode:
                raise SpeechError(f"Windows synthesis failed: {completed.stderr.strip()}")
        for job in payload:
            read_wav(job["output"], max_seconds=300)

    def synthesize(self, text, output):
        self.synthesize_many([{"text": text, "output": str(output)}])
        return str(Path(output).resolve())


def record_microphone(path, seconds=8, device=None):
    """Explicit, bounded capture. Caller starts it only after a user command."""
    import sounddevice as sd
    if not 1 <= seconds <= 30:
        raise ValueError("Recording duration must be between 1 and 30 seconds")
    try:
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                       dtype="int16", device=device, blocking=True)
        write_wav(path, audio.astype("<i2").tobytes())
    except Exception as exc:
        raise SpeechError(f"Microphone capture failed: {exc}") from exc
    finally:
        sd.stop()


def play_audio(path, device=None):
    import sounddevice as sd
    pcm, _ = read_wav(path, max_seconds=300)
    try:
        sd.play(np.frombuffer(pcm, dtype="<i2"), SAMPLE_RATE, device=device, blocking=True)
    except Exception as exc:
        raise SpeechError(f"Audio playback failed: {exc}") from exc
    finally:
        sd.stop()


def recognize_audio(asr, path, min_confidence=0.55):
    from .nlp import stop_requested
    try:
        transcript = asr.transcribe(path)
        accepted = transcript.confidence >= min_confidence or stop_requested(transcript.normalized_text)
        return {"accepted": accepted, "transcript": asdict(transcript),
                "spoken_text": None if accepted else "I could not hear that clearly. Please repeat or use the text interface."}
    except NoSpeechError:
        return {"accepted": False, "transcript": None,
                "spoken_text": "I did not catch any speech. Please repeat or use the text interface."}


class VoiceSession:
    def __init__(self, conversation, asr, tts, min_confidence=0.55):
        if not 0 <= min_confidence <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        self.conversation, self.asr, self.tts = conversation, asr, tts
        self.min_confidence = min_confidence

    def reply_audio(self, audio_path, output_path):
        if self.conversation.closed:
            raise RuntimeError("This conversation is closed")
        if Path(audio_path).resolve() == Path(output_path).resolve():
            raise ValueError("Input audio and response audio must have different paths")
        start = time.perf_counter()
        recognized = recognize_audio(self.asr, audio_path, self.min_confidence)
        accepted, transcript = recognized["accepted"], recognized["transcript"]
        response, text = None, recognized["spoken_text"]
        if accepted:
            response = self.conversation.reply(transcript["normalized_text"])
            text = response["text"] if response["validation"]["ok"] else "I cannot verify that response. Please use the text interface for clarification."
        generation_seconds = time.perf_counter() - start
        tts_start = time.perf_counter()
        audio, tts_error = None, None
        try:
            if self.tts is not None:
                audio = self.tts.synthesize(text, output_path)
        except (SpeechError, OSError) as exc:
            tts_error = str(exc)
        return {"accepted": accepted, "transcript": transcript,
                "response": response, "spoken_text": text, "audio_path": audio,
                "tts_error": tts_error, "closed": self.conversation.closed,
                "timing": {"asr_and_response_seconds": generation_seconds,
                           "tts_seconds": time.perf_counter() - tts_start,
                           "total_seconds": time.perf_counter() - start}}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
