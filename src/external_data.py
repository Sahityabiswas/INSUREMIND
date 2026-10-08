"""Normalize licensed local public NLP corpora without fabricating sales labels."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .common import INTENTS, EMOTIONS
from .data_pipeline import EXTERNAL

GO_EMOTIONS = ("admiration amusement anger annoyance approval caring confusion curiosity desire disappointment "
               "disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism "
               "pride realization relief remorse sadness surprise neutral").split()
EMOTION_MAP = {"anger": "ANGRY", "annoyance": "FRUSTRATED", "disapproval": "SKEPTICAL",
               "fear": "ANXIOUS", "nervousness": "ANXIOUS", "sadness": "CONCERNED", "grief": "CONCERNED",
               "disappointment": "CONCERNED", "confusion": "CONCERNED", "disgust": "FRUSTRATED",
               "neutral": "NEUTRAL", "relief": "SATISFIED", "gratitude": "SATISFIED"}
for emotion in ("admiration", "amusement", "approval", "caring", "excitement", "joy", "love", "optimism", "pride"):
    EMOTION_MAP[emotion] = "POSITIVE"


def normalize(path, dataset, split="train", intent_map=None):
    if dataset not in ("multidogo", "goemotions", "meld"):
        raise ValueError("Supported NLP readers: multidogo, goemotions, meld")
    if split not in ("train", "val", "test"):
        raise ValueError("Use the upstream split: train, val or test")
    metadata = {name: (url, license_name) for name, url, license_name, _ in EXTERNAL}
    source, license_name = metadata[dataset]
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t") if dataset == "goemotions" else csv.DictReader(f, delimiter="\t" if dataset == "multidogo" else ",")
        for index, row in enumerate(reader):
            if dataset == "goemotions":
                if len(row) != 3:
                    raise ValueError(f"Malformed GoEmotions row {index}")
                text, ids, record_id = row
                labels = [GO_EMOTIONS[int(i)] for i in ids.split(",")]
                mapped = {EMOTION_MAP.get(label, "NEUTRAL") for label in labels}
                emotion = next(iter(mapped)) if len(mapped) == 1 else None
                native_labels, conversation_id = labels, record_id
                intent = None
            elif dataset == "multidogo":
                text, conversation_id = row["utterance"], row["conversationId"]
                native_labels = row.get("intent", "").split("<div>")
                mapped = {intent_map[label] for label in native_labels if label in (intent_map or {})}
                intent = next(iter(mapped)) if len(mapped) == 1 else None
                emotion = None
            else:
                text, conversation_id = row["Utterance"], str(row["Dialogue_ID"])
                native_labels = [row["Emotion"].lower()]
                emotion = EMOTION_MAP.get(native_labels[0], "NEUTRAL")
                intent = None
            if intent is not None and intent not in INTENTS or emotion is not None and emotion not in EMOTIONS:
                raise ValueError("Label map contains unknown project labels")
            if not text.strip():
                continue
            records.append({"record_id": f"{dataset}:{split}:{index}", "conversation_id": f"{dataset}:{conversation_id}",
                            "text": text.strip(), "source_type": "public", "source_dataset": dataset,
                            "license_reference": license_name, "source_url": source, "split": split,
                            "native_labels": native_labels, "label_source": "derived_mapping",
                            "customer_intent": intent, "customer_emotion": emotion,
                            "mapping_version": "nlp_transfer_v1"})
    return records


def import_corpus(path, dataset, split="train", intent_map=None, output_dir=None):
    records = normalize(path, dataset, split, intent_map)
    output = Path(output_dir or Path(__file__).resolve().parents[1] / "data" / "interim" / "external")
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{dataset}_{split}.jsonl"
    existing = []
    for other in output.glob(f"{dataset}_*.jsonl"):
        if other == target:
            continue
        existing.extend(json.loads(line) for line in other.read_text(encoding="utf-8").splitlines())
    other_ids = {r["conversation_id"] for r in existing}
    if any(r["conversation_id"] in other_ids for r in records):
        raise ValueError("Upstream conversation appears in another split")
    seen, unique = set(), []
    for r in records:
        key = (r["conversation_id"], r["text"], tuple(r["native_labels"]))
        if key not in seen:
            unique.append(r)
            seen.add(key)
    with target.open("w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with target.with_suffix(".manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "split": split, "rows": len(unique), "input_sha256": digest,
                   "original_path": str(Path(path).resolve()), "mapping_version": "nlp_transfer_v1",
                   "intent_mapping": intent_map or {}}, f, indent=2)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["multidogo", "goemotions", "meld"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--intent-map", help="JSON mapping from native MultiDoGO intents to project labels")
    args = parser.parse_args()
    mapping = json.loads(Path(args.intent_map).read_text()) if args.intent_map else None
    print(import_corpus(args.input, args.dataset, args.split, mapping))


if __name__ == "__main__":
    main()
