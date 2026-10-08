"""Local transformer inference. Missing models fail explicitly, never silently use NB."""
from functools import lru_cache
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from .common import EMOTIONS, INTENTS, OBJECTIONS, load_config

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / ".runtime/understanding"
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/huggingface"))
_LOAD_LOCK = threading.RLock()

# This is an explicit heuristic label mapping, not an eight-label fine-tuned model.
EMOTION_MAPPING = {
    "POSITIVE": ["joy", "excitement", "optimism", "love", "amusement", "admiration"],
    "NEUTRAL": ["neutral", "curiosity", "realization", "surprise"],
    "CONCERNED": ["confusion", "sadness", "disappointment", "caring"],
    "SKEPTICAL": ["disapproval"],
    "FRUSTRATED": ["annoyance"],
    "ANGRY": ["anger", "disgust"],
    "ANXIOUS": ["fear", "nervousness"],
    "SATISFIED": ["gratitude", "relief", "approval", "pride"],
}


def config():
    return load_config(ROOT / "configs/understanding.yaml")


def selected_backend():
    backend = os.environ.get("INSURANCE_NLP_BACKEND", config()["backend"])
    if backend not in ("transformer", "nb"):
        raise ValueError("INSURANCE_NLP_BACKEND must be transformer or nb")
    return backend


def status():
    cfg = config()
    path = ROOT / cfg["model_dir"]
    ready = all(p.is_file() for p in (path / "model.safetensors", path / "metadata.json",
                                    ASSETS / "roberta/onnx/model_quantized.onnx"))
    return {"backend": selected_backend(), "transformer_ready": ready,
            "model_dir": cfg["model_dir"], "intent_model": "MiniLM fine-tuned encoder",
            "objection_output": "independent sigmoid, multi-label",
            "text_emotion_model": "RoBERTa GoEmotions, heuristic 28-to-8 mapping",
            "acoustic_ready": (ASSETS / "acoustic/onnx/model.onnx").is_file(),
            "acoustic_enabled": cfg["acoustic"]["enabled"],
            "acoustic_model": "wav2vec2 SUPERB ER, four original labels",
            "emotion_validated_on_real_buyers": False,
            "ppo_compatibility": "94-feature policy uses the primary objection; audio is advisory only"}


def ort_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = config()["threads"]
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def map_emotions(original):
    scores = {label: max((original.get(k, 0.0) for k in names), default=0.0)
              for label, names in EMOTION_MAPPING.items()}
    total = sum(scores.values())
    return {label: score / total if total else float(label == "NEUTRAL") for label, score in scores.items()}


class TextEmotion:
    def __init__(self):
        from tokenizers import Tokenizer
        folder = ASSETS / "roberta"
        self.session = ort_session(folder / "onnx/model_quantized.onnx")
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=config()["max_tokens"])
        self.labels = json.loads((folder / "config.json").read_text())["id2label"]

    def __call__(self, text):
        tokens = self.tokenizer.encode(text)
        inputs = {"input_ids": np.asarray([tokens.ids], dtype=np.int64),
                  "attention_mask": np.asarray([tokens.attention_mask], dtype=np.int64)}
        scores = sigmoid(self.session.run(None, inputs)[0][0])
        original = {self.labels[str(i)]: float(x) for i, x in enumerate(scores)}
        mapped = map_emotions(original)
        label = max(mapped, key=mapped.get)
        return {"emotion": label, "emotion_conf": mapped[label], "emotion_probs": mapped,
                "text_emotion": {"model": "RoBERTa GoEmotions (INT8)", "original_scores": original,
                    "mapping": "heuristic_28_to_8_v1", "uncertain": mapped[label] < 0.55,
                    "calibrated": False, "domain_validated": False}}


class TransformerPredictor:
    def __init__(self, model_dir=None):
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoTokenizer
            from .transformer_model import InsuranceEncoder
        except (ImportError, OSError) as exc:
            raise RuntimeError("Transformer runtime unavailable. Install requirements-understanding.txt "
                               "or explicitly choose the NB baseline. " + str(exc)) from exc
        cfg = config()
        torch.set_num_threads(cfg["threads"])
        folder = Path(model_dir or ROOT / cfg["model_dir"])
        try:
            self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.model = InsuranceEncoder(folder / "encoder")
            self.model.load_state_dict(load_file(str(folder / "model.safetensors")))
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(folder / "encoder", local_files_only=True)
            self.emotion = TextEmotion()
        except (OSError, ValueError) as exc:
            raise RuntimeError("Transformer assets unavailable. Run setup_understanding.py then train_understanding.py; "
                               "or explicitly select the NB baseline.") from exc
        self.lock = threading.RLock()
        self.max_tokens = cfg["max_tokens"]

    def __call__(self, text, ctx=""):
        import torch
        from .nlp import stop_requested
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
            intent, objections = self.model(**tokens)
            ip = torch.softmax(intent[0], dim=0).tolist()
            op = torch.sigmoid(objections[0]).tolist()
            emotion = self.emotion(text)
        intent_label = INTENTS[int(np.argmax(ip))]
        scores = dict(zip(OBJECTIONS[1:], op))
        thresholds = self.metadata["objection_thresholds"]
        labels = sorted((label for label in scores if scores[label] >= thresholds[label]),
                        key=scores.get, reverse=True)
        primary = labels[0] if labels else "NONE"
        result = {"intent": intent_label, "intent_conf": max(ip), "intent_probs": dict(zip(INTENTS, ip)),
                  "objection": primary, "objection_conf": scores.get(primary, 1 - max(op)),
                  "objections": labels, "objection_probs": scores,
                  "stage": "OBJECTION_HANDLING" if labels else "DISCOVERY", **emotion,
                  "classifier": {"backend": "transformer", "model": "MiniLM-L6 fine-tuned",
                      "version": self.metadata["version"], "multilabel": True,
                      "seconds": time.perf_counter() - started, "calibrated": False}}
        if stop_requested(text):
            result.update(intent="REJECTION", intent_conf=1.0, objection="NOT_READY", objection_conf=1.0,
                          objections=list(dict.fromkeys(["NOT_READY", *labels])), stage="REJECTION")
        return result


@lru_cache(maxsize=2)
def _predictor(model_dir):
    return TransformerPredictor(model_dir)


def get_predictor(model_dir=None):
    with _LOAD_LOCK:
        return _predictor(str(Path(model_dir or ROOT / config()["model_dir"]).resolve()))
