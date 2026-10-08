"""Fine-tune MiniLM, evaluate multi-label objections and the eight-category emotion mapping."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from src.common import EMOTIONS, INTENTS, OBJECTIONS
from src.nlp import _scores, load_rows

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/huggingface"))
DATA = ROOT / "data/annotations/understanding_v1.json"


def normalized(text):
    return " ".join(text.lower().split()).rstrip(".!?")


def corpus():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    splits = {s: [] for s in ("train", "val", "test")}
    for group in data["groups"]:
        for split in splits:
            splits[split].extend({"text": t, "intent": group["intent"], "objections": group["objections"],
                                  "source": "authored_synthetic"} for t in group[split])
    for split, rows in data["compound"].items():
        splits[split].extend({**row, "source": "authored_synthetic_compound"} for row in rows)
    # Compound augmentation only uses training sentences; validation/test are never augmented into training.
    objectors = [g for g in data["groups"] if len(g["objections"]) == 1 and g["intent"] != "REJECTION"]
    for a, b in itertools.combinations(objectors, 2):
        for k in (0, 1):
            splits["train"].append({"text": a["train"][k].rstrip(".") + ", and " + b["train"][k].lower(),
                                    "intent": "OBJECTION", "objections": a["objections"] + b["objections"],
                                    "source": "synthetic_train_only_composition"})
    original = load_rows("train")
    seen = {normalized(r["text"]) for rows in splits.values() for r in rows}
    for row in original:
        key = normalized(row["text"])
        if key not in seen:
            seen.add(key)
            splits["train"].append({"text": row["text"], "intent": row["customer_intent"],
                "objections": [] if row["objection_type"] == "NONE" else [row["objection_type"]],
                "source": "legacy_synthetic_weak_label"})
    split_keys = {s: {normalized(r["text"]) for r in rows} for s, rows in splits.items()}
    for a, b in itertools.combinations(splits, 2):
        if split_keys[a] & split_keys[b]:
            raise ValueError(f"Duplicate text leakage: {a}/{b}")
    conflicts = defaultdict(set)
    for row in original:
        conflicts[normalized(row["text"])].add(row["customer_emotion"])
    audit = {"legacy_train_rows": len(original), "legacy_unique_texts": len(conflicts),
             "legacy_texts_with_conflicting_emotions": sum(len(v) > 1 for v in conflicts.values()),
             "legacy_split_text_overlap": len({normalized(r['text']) for r in original} &
                                              {normalized(r['text']) for r in load_rows('test')}),
             "new_split_sizes": {s: len(r) for s, r in splits.items()}, "new_exact_text_overlap": 0,
             "labels": "Synthetic, not human-validated; legacy emotion labels excluded from fine-tuning"}
    return data, splits, audit


def multilabel_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp, fp, fn = (truth & predicted).sum(0), (~truth & predicted).sum(0), (truth & ~predicted).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {"micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
            "macro_f1": float(f1.mean()), "exact_match": float((truth == predicted).all(1).mean()),
            "per_label": {label: {"f1": float(f), "support": int(n)}
                          for label, f, n in zip(OBJECTIONS[1:], f1, truth.sum(0))}, "examples": len(truth)}


def evaluate(model, tokenizer, rows):
    import torch
    model.eval()
    intents, objections = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            tokens = tokenizer([r["text"] for r in rows[start:start + 16]], padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            i, o = model(**tokens)
            intents.extend(i.argmax(1).tolist())
            objections.extend(torch.sigmoid(o).tolist())
    return intents, np.asarray(objections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/understanding_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.threads <= 16:
        parser.error("epochs must be 1-100 and threads 1-16")
    output = ROOT / args.output
    if output.exists() and any(output.iterdir()):
        parser.error("Output already contains an experiment; use a new --output directory.")
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    from src.transformer_model import InsuranceEncoder
    from src.understanding import ASSETS, TextEmotion, EMOTION_MAPPING
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data, splits, audit = corpus()
    print(json.dumps(audit, indent=2), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ASSETS / "minilm", local_files_only=True)
    model = InsuranceEncoder(ASSETS / "minilm")
    texts = [r["text"] for r in splits["train"]]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    y_i = torch.tensor([INTENTS.index(r["intent"]) for r in splits["train"]])
    y_o = torch.tensor([[float(c in r["objections"]) for c in OBJECTIONS[1:]] for r in splits["train"]])
    pos_weight = ((len(texts) - y_o.sum(0)) / y_o.sum(0).clamp_min(1)).clamp(max=8)
    intent_weight = (len(texts) / (len(INTENTS) * torch.bincount(y_i, minlength=len(INTENTS)).clamp_min(1))).sqrt()
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 3e-5},
        {"params": itertools.chain(model.intent.parameters(), model.objections.parameters()), "lr": 1e-3}],
        weight_decay=0.01)
    best, best_state, history = -1, None, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(texts)).split(16):
            optimizer.zero_grad(set_to_none=True)
            i, o = model(**{k: v[batch] for k, v in tokens.items()})
            loss = torch.nn.functional.cross_entropy(i, y_i[batch], weight=intent_weight) + \
                   torch.nn.functional.binary_cross_entropy_with_logits(o, y_o[batch], pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        vi, vo = evaluate(model, tokenizer, splits["val"])
        vm = _scores([r["intent"] for r in splits["val"]], [INTENTS[i] for i in vi], INTENTS)
        vt = [[c in r["objections"] for c in OBJECTIONS[1:]] for r in splits["val"]]
        vf = multilabel_metrics(vt, vo >= .5)
        score = (vm["macro_f1"] + vf["macro_f1"]) / 2
        entry = {"epoch": epoch + 1, "loss": total / len(texts), "val_intent_macro_f1": vm["macro_f1"],
                 "val_objection_macro_f1": vf["macro_f1"]}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if score > best:
            best, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, vo = evaluate(model, tokenizer, splits["val"])
    vt = np.asarray([[c in r["objections"] for c in OBJECTIONS[1:]] for r in splits["val"]])
    thresholds = []
    for col in range(len(OBJECTIONS) - 1):
        candidates = []
        for threshold in (.3, .4, .5, .6, .7):
            pred = vo[:, col] >= threshold
            tp = (pred & vt[:, col]).sum()
            f1 = 2 * tp / max(pred.sum() + vt[:, col].sum(), 1)
            candidates.append((f1, -abs(threshold - .5), threshold))
        thresholds.append(max(candidates)[2])
    ti, to = evaluate(model, tokenizer, splits["test"])
    tt = [[c in r["objections"] for c in OBJECTIONS[1:]] for r in splits["test"]]
    report = {"intent": _scores([r["intent"] for r in splits["test"]], [INTENTS[i] for i in ti], INTENTS),
              "objections": multilabel_metrics(tt, to >= thresholds), "data_audit": audit,
              "limitations": ["Small agent-authored synthetic language test, not independent human validation",
                              "Multi-label thresholds selected on validation only; probabilities not calibrated",
                              "Emotion mapping is heuristic, RoBERTa encoder not fine-tuned on insurance data",
                              "No labeled emotional human audio is available for acoustic accuracy evaluation",
                              "This language evaluation does not measure PPO performance; see the separate policy experiment"]}
    emotion_model = TextEmotion()
    truth, predicted, details = [], [], []
    for label, sentences in data["emotion_evaluation"].items():
        for text in sentences:
            result = emotion_model(text)
            truth.append(label)
            predicted.append(result["emotion"])
            details.append({"text": text, "expected": label, "predicted": result["emotion"],
                            "score": result["emotion_conf"]})
    report["text_emotion_mapping"] = {**_scores(truth, predicted, EMOTIONS), "details": details,
        "confusion": {a: {b: sum(x == a and y == b for x, y in zip(truth, predicted)) for b in EMOTIONS} for a in EMOTIONS}}
    report["test_predictions"] = [{"text": r["text"], "expected_intent": r["intent"], "intent": INTENTS[i],
        "expected_objections": r["objections"], "objections": [c for c, p, t in zip(OBJECTIONS[1:], scores, thresholds) if p >= t]}
        for r, i, scores in zip(splits["test"], ti, to)]
    output.mkdir(parents=True)
    model.encoder.save_pretrained(output / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(output / "encoder")
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(output / "model.safetensors"))
    metadata = {"version": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoder_training": "full fine-tuning, all MiniLM layers", "seed": args.seed,
        "epochs": args.epochs, "best_epoch": best_epoch, "training_seconds": time.perf_counter() - started,
        "objection_loss": "BCEWithLogitsLoss", "intent_loss": "CrossEntropyLoss",
        "objection_thresholds": dict(zip(OBJECTIONS[1:], thresholds)), "emotion_mapping": EMOTION_MAPPING,
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(), "data_provenance": data["provenance"],
        "pretrained": {name: json.loads((ASSETS / name / "provenance.json").read_text()) for name in ("minilm", "roberta")}}
    for name, obj in (("metadata.json", metadata), ("evaluation.json", report), ("history.json", history), ("corpus.json", splits)):
        (output / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("test_predictions", "text_emotion_mapping")}, indent=2), flush=True)
    print("Emotion mapping:", report["text_emotion_mapping"]["macro_f1"], "Saved:", output, flush=True)


if __name__ == "__main__":
    main()
