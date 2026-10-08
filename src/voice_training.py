"""Speech-conditioned data, NLP adaptation and PPO environment; no speech-weight training."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re

import gymnasium as gym
import numpy as np

from . import nlp
from .common import ACTIONS, NEEDS
from .conversation import ConversationSession
from .sim_env import InsuranceEnv
from .speech import ROOT, NoSpeechError, SpeechError, file_sha256, read_wav

TASKS = {"intent": "customer_intent", "emotion": "customer_emotion",
         "objection": "objection_type", "stage": "sales_stage"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def simulator_utterances(observable_dialogue=False):
    phrases = ["I'd like to discuss insurance options.", "That makes sense, tell me more.",
            "Hmm, I need to think about it.", "No, that's not what I want.",
            "Please stop. I am not interested."] + [
                f"I need {need.lower().replace('_', ' ')} with a {budget} budget."
                for need in NEEDS for budget in ("low", "mid", "high")]
    if observable_dialogue:
        phrases += ["That is too expensive for my budget.", "I need more information about the exclusions.",
                    "I already have an insurance policy.", "I understand the terms and I am ready to proceed.",
                    "That makes sense. How would the next step work?"]
    return phrases


def prepare_corpus(folder, asr, tts, limit=80, rates=None, observable_dialogue=False, reuse_manifest=None):
    folder = Path(folder)
    rates = rates or {"train": 0, "val": -1, "test": 1}
    if limit <= 0 or set(rates) != {"train", "val", "test"}:
        raise ValueError("Positive example limit and train/val/test speech rates are required")
    rows, jobs, reused = [], {}, {}
    if reuse_manifest and Path(reuse_manifest).is_file():
        for old in read_jsonl(reuse_manifest):
            path = Path(old["wav"])
            if old["asr_model"] == asr.model_name and path.is_file() and file_sha256(path) == old["audio_sha256"]:
                reused[(old["text"], old["voice"], old["speech_rate"])] = old
    for split in ("train", "val", "test"):
        source = ROOT / f"data/processed/insurance_sales/{split}.jsonl"
        candidates = [r for r in read_jsonl(source) if r["speaker"] == "customer"]
        random.Random(42).shuffle(candidates)
        seen, selected = set(), []
        for row in candidates:
            if row["text"] not in seen:
                selected.append(row)
                seen.add(row["text"])
            if len(selected) >= limit:
                break
        entries = [{"kind": "nlp", "labels": row, "text": row["text"],
                    "source_sha256": file_sha256(source)} for row in selected]
        entries += [{"kind": "simulator", "text": text} for text in simulator_utterances(observable_dialogue)]
        for entry in entries:
            key = hashlib.sha256(json.dumps([entry["text"], tts.voice, rates[split]], ensure_ascii=True).encode()).hexdigest()[:24]
            path = (folder / "audio" / f"{key}.wav").resolve()
            old = reused.get((entry["text"], tts.voice, rates[split]))
            if old:
                path = Path(old["wav"])
            row = {**entry, "id": f"{split}-{entry['kind']}-{key}", "split": split,
                   "wav": str(path), "speech_rate": rates[split], "voice": tts.voice,
                   "source_type": "synthetic_tts", "speaker_type": "pretrained_system_voice",
                   "asr_model": asr.model_name}
            rows.append(row)
            jobs[key] = {"text": entry["text"], "rate": rates[split], "output": path}
    pending = []
    for job in jobs.values():
        try:
            read_wav(job["output"], max_seconds=300)
        except (OSError, SpeechError):
            pending.append(job)
    print(f"Synthesizing {len(pending)} local speech files", flush=True)
    tts.synthesize_many(pending)
    cache = {old["wav"]: old["transcript"] for old in reused.values()}
    for index, row in enumerate(rows):
        if row["wav"] not in cache:
            try:
                cache[row["wav"]] = {**asdict(asr.transcribe(row["wav"])), "asr_error": None}
            except NoSpeechError as exc:
                cache[row["wav"]] = {"text": "", "normalized_text": "", "confidence": 0.0,
                                     "asr_seconds": None, "audio_seconds": None, "asr_error": str(exc)}
        row["transcript"] = cache[row["wav"]]
        row["audio_sha256"] = file_sha256(row["wav"])
        if (index + 1) % 25 == 0:
            print(f"Transcribed {index + 1}/{len(rows)}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_json(folder / "provenance.json", {"source": "local synthetic text -> Windows TTS -> Vosk",
        "speech_weights_updated": False, "rates": rates, "rows": len(rows),
        "manifest_sha256": file_sha256(manifest),
        "limitations": ["One synthetic speaker; not a human-speech benchmark",
                        "NLP examples retain the original conversation split; utterance templates can repeat",
                        "Simulator phrases repeat across splits with different speech rates",
                        "No personal microphone recordings are included automatically"]})
    return rows


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def normalized_words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def speech_metrics(rows):
    word_errors = words = char_errors = chars = exact = 0
    for row in rows:
        reference = normalized_words(row["text"])
        hypothesis = normalized_words(row["transcript"]["text"])
        word_errors += edit_distance(reference, hypothesis)
        words += len(reference)
        left, right = " ".join(reference), " ".join(hypothesis)
        char_errors += edit_distance(left, right)
        chars += len(left)
        exact += reference == hypothesis
    return {"examples": len(rows), "word_errors": word_errors, "reference_words": words,
            "wer": word_errors / words if words else None, "cer": char_errors / chars if chars else None,
            "exact_match": exact / len(rows) if rows else None,
            "failed_transcriptions": sum(bool(r["transcript"].get("asr_error")) for r in rows)}


def load_predictor(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or set(data["models"]) != set(TASKS):
        raise ValueError("Incompatible voice NLP bundle")
    models = {}
    for name, fields in data["models"].items():
        model = nlp.NB()
        for field in ("classes", "vocab", "logprob", "logprior"):
            setattr(model, field, fields[field])
        model.ci = {label: i for i, label in enumerate(model.classes)}
        models[name] = model

    @lru_cache(maxsize=4096)
    def predict(text):
        return nlp.predict(text, models=models)
    return predict


def train_nlp(rows, output):
    training = [r for r in nlp.load_rows("train")]
    training += [{**row["labels"], "text": row["transcript"]["normalized_text"]}
                 for row in rows if row["kind"] == "nlp" and row["split"] == "train"
                 and row["transcript"]["normalized_text"]]
    models = {}
    for name, label in TASKS.items():
        model = nlp.NB().fit([r["text"] for r in training], [r[label] for r in training])
        models[name] = {field: getattr(model, field) for field in ("classes", "vocab", "logprob", "logprior")}
    path = Path(output) / "nlp_models.json"
    write_json(path, {"version": 1, "training_examples": len(training), "models": models})
    adapted = load_predictor(path)
    metrics = {}
    for split in ("val", "test"):
        examples = [row for row in rows if row["kind"] == "nlp" and row["split"] == split]
        if not examples:
            raise ValueError(f"No held-out NLP examples in {split}")
        metrics[split] = {}
        for name, predictor in (("text_nlp", nlp.predict), ("speech_adapted_nlp", adapted)):
            guesses = [predictor(row["transcript"]["normalized_text"]) for row in examples]
            metrics[split][name] = {task: nlp._scores([r["labels"][col] for r in examples],
                [guess[task] for guess in guesses], models[task]["classes"]) for task, col in TASKS.items()}
    write_json(Path(output) / "nlp_metrics.json", metrics)
    return adapted, metrics


class TranscriptChannel:
    def __init__(self, rows, split, observable_dialogue=False):
        self.lookup = {row["text"]: row["transcript"] for row in rows
                       if row["kind"] == "simulator" and row["split"] == split}
        missing = set(simulator_utterances(observable_dialogue)) - self.lookup.keys()
        if missing:
            raise ValueError(f"Missing {len(missing)} simulator speech transcripts for {split}")

    def __call__(self, text):
        if text not in self.lookup:
            raise ValueError(f"Uncached simulator utterance: {text!r}. Regenerate the speech dataset.")
        return self.lookup[text]


class _GeneratedResponse:
    result = None

    def generate(self, *args, **kwargs):
        if self.result is None:
            raise RuntimeError("No conversation response was prepared")
        return self.result


class SpeechInsuranceEnv(gym.Env):
    """PPO sees ASR-derived live state, while the simulator retains hidden reward state."""
    metadata = {"render_modes": []}

    def __init__(self, channel, predictor, reward_weights=None, max_turns=20, min_confidence=0.55,
                 observable_dialogue=False):
        self.forward = _GeneratedResponse()
        self.env = InsuranceEnv(reward_weights=reward_weights, max_turns=max_turns, generator=self.forward,
                                observable_dialogue=observable_dialogue)
        self.action_space, self.observation_space = self.env.action_space, self.env.observation_space
        self.channel, self.predictor, self.min_confidence = channel, predictor, min_confidence

    def _prepare(self, info):
        self.transcript = self.channel(info["customer_text"])
        text = self.transcript["normalized_text"]
        self.retry = not text or (self.transcript["confidence"] < self.min_confidence and not nlp.stop_requested(text))
        self.prepared = self.session.prepare_turn(text or "[unintelligible audio]")
        mask = list(self.prepared["mask"])
        if self.retry:
            # A bounded simulator retry advances its turn; live retries do not change buyer memory.
            action = "ASK_CLARIFYING_QUESTION"
            mask = [name == action for name in ACTIONS]
            self.prepared["task"] = None
        elif self.prepared["task"] in ("clarify_amount", "explain_payment"):
            action = "AFFORDABILITY_FRAMING" if self.prepared["task"] == "explain_payment" and self.prepared["available"] else "ASK_CLARIFYING_QUESTION"
            mask = [name == action for name in ACTIONS]
        self.prepared["mask"] = mask
        visible = {**self.prepared["info"], "action_mask": mask, "asr_retry": self.retry,
                   "asr_confidence": self.transcript["confidence"]}
        self.visible = visible
        return self.prepared["observation"], visible

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _, info = self.env.reset(seed=seed, options=options)
        self.session = ConversationSession("rule", profile=self.env.sim.profile, age=self.env.sim.age,
                                           predictor=self.predictor)
        self.initial_objection = self.env.sim.s["objection"] != "NONE"
        self.done = False
        return self._prepare(info)

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset the voice environment before another episode")
        result = self.session.complete_turn(self.prepared, action_id=action)
        self.forward.result = result
        _, reward, terminated, truncated, truth = self.env.step(ACTIONS.index(result["action"]))
        self.done = terminated or truncated
        if terminated:
            observation, visible = self.prepared["observation"], self.visible
        else:
            observation, visible = self._prepare(truth)
        info = {**truth, **visible, "simulated_purchase": truth["purchased"],
                "simulated_satisfaction": truth["satisfaction"],
                "missed_stop": bool(truth["stop_requested"] and not nlp.stop_requested(self.transcript["normalized_text"]))}
        return observation, reward, terminated, truncated, info


def evaluate_policy(env, policy, episodes, seed, trajectory_sink=None):
    rows = []
    for episode in range(episodes):
        obs, info = env.reset(seed=100000 + seed * 10000 + episode)
        reward = turns = violations = retries = missed_stop = commitments = interest = repairs = 0
        done = False
        while not done:
            retries += bool(info["asr_retry"])
            action = policy(obs, info["action_mask"], info)
            if not info["action_mask"][action]:
                repairs += 1
                action = next(i for i, ok in enumerate(info["action_mask"]) if ok)
            interest += info["sales_stage"] == "COMMITMENT"
            commitments += ACTIONS[action] == "ASK_FOR_COMMITMENT"
            before = {"heard": env.transcript["normalized_text"], "stage": info["sales_stage"],
                      "objection": info["objection"], "action": ACTIONS[action]}
            obs, r, terminated, truncated, info = env.step(action)
            if trajectory_sink is not None:
                trajectory_sink.append({"episode": episode, "seed": seed, "turn": turns, **before,
                    "reward": r, "simulated_purchase": info["simulated_purchase"],
                    "violation": info["safety_violation"], "premature": info["premature"],
                    "unsuitable": info["unsuitable"], "pressure": info["pressure"],
                    "unsupported": info["unsupported"], "response": info["seller_response"]})
            reward += r
            turns += 1
            violations += bool(info["safety_violation"])
            missed_stop += bool(info["missed_stop"])
            done = terminated or truncated
        rows.append({"episode": episode, "seed": seed, "reward": reward, "turns": turns,
                     "conversion": int(info["simulated_purchase"]), "satisfaction": info["simulated_satisfaction"],
                     "interest_turns": interest, "commitment_attempts": commitments, "masked_action_repairs": repairs,
                     "violations": violations, "asr_retries": retries, "missed_stop_turns": missed_stop})
    summary = {key: float(np.mean([row[key] for row in rows])) for key in
               ("reward", "turns", "conversion", "satisfaction", "violations", "asr_retries", "missed_stop_turns",
                "interest_turns", "commitment_attempts", "masked_action_repairs")}
    return summary, rows
