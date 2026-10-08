"""Waveform-based emotion estimates, kept separate from language and consent."""
from functools import lru_cache
import json
import threading
import time

import numpy as np

from .speech import SAMPLE_RATE, read_wav
from .understanding import ASSETS, config, ort_session

LABELS = {"neu": "NEUTRAL", "hap": "HAPPY", "ang": "ANGRY", "sad": "SAD"}
_LOAD_LOCK = threading.Lock()


class AcousticEmotion:
    def __init__(self):
        folder = ASSETS / "acoustic"
        self.session = ort_session(folder / "onnx/model.onnx")
        self.labels = json.loads((folder / "config.json").read_text())["id2label"]
        self.preprocessing = json.loads((folder / "preprocessor_config.json").read_text())
        self.lock = threading.Lock()

    def predict(self, pcm, duration):
        cfg = config()["acoustic"]
        started = time.perf_counter()
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        samples = samples[:int(cfg["max_seconds"] * SAMPLE_RATE)]
        if self.preprocessing["do_normalize"]:
            samples = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
        available = {"input_values": samples[None, :], "attention_mask": np.ones((1, len(samples)), dtype=np.int64)}
        inputs = {node.name: available[node.name] for node in self.session.get_inputs()}
        with self.lock:
            logits = self.session.run(None, inputs)[0][0]
        exp = np.exp(logits - np.max(logits))
        probabilities = exp / exp.sum()
        scores = {LABELS[self.labels[str(i)]]: float(p) for i, p in enumerate(probabilities)}
        ranking = sorted(scores, key=scores.get, reverse=True)
        uncertain = scores[ranking[0]] < cfg["confidence_threshold"] or scores[ranking[0]] - scores[ranking[1]] < cfg["margin_threshold"]
        return {"status": "uncertain" if uncertain else "estimated", "label": ranking[0],
                "confidence": scores[ranking[0]], "probabilities": scores,
                "model": "wav2vec2-base-superb-er", "source": "audio_waveform",
                "analyzed_seconds": len(samples) / SAMPLE_RATE, "audio_seconds": duration,
                "truncated": duration > cfg["max_seconds"], "inference_seconds": time.perf_counter() - started,
                "calibrated": False, "domain_validated": False, "used_for_policy": False,
                "scope": "Four IEMOCAP labels; not the eight INSUREMIND categories. Advisory only."}


@lru_cache(maxsize=1)
def _model():
    return AcousticEmotion()


def analyze_audio(path):
    cfg = config()["acoustic"]
    if not cfg["enabled"]:
        return {"status": "disabled", "source": "audio_waveform"}
    try:
        pcm, duration = read_wav(path)
        level = np.sqrt(np.mean((np.frombuffer(pcm, dtype="<i2").astype(float) / 32768.0) ** 2))
        if duration < cfg["min_seconds"] or level < 0.001:
            return {"status": "insufficient_audio", "source": "audio_waveform", "audio_seconds": duration}
        with _LOAD_LOCK:
            model = _model()
        return model.predict(pcm, duration)
    except Exception as exc:
        # Acoustic failure must be visible but must not discard a usable transcript or stop request.
        return {"status": "unavailable", "source": "audio_waveform", "error": str(exc),
                "used_for_policy": False}
