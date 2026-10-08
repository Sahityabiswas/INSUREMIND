import json
import os
from pathlib import Path

import numpy as np
import pytest

from src import entities
from src.common import EMOTIONS, OBJECTIONS
from src.conversation import ConversationSession
from src.speech import Transcript, recognize_audio, write_wav
from src.understanding import map_emotions
from train_understanding import corpus, multilabel_metrics, normalized

NATIVE = pytest.mark.skipif(os.environ.get("TRANSFORMER_INTEGRATION") != "1", reason="Opt-in downloaded transformer models")


def test_no_duplicate_sentence_leakage_and_true_multilabel_rows():
    _, splits, audit = corpus()
    keys = {s: {normalized(r["text"]) for r in rows} for s, rows in splits.items()}
    assert not keys["train"] & (keys["val"] | keys["test"])
    assert not keys["val"] & keys["test"]
    assert any(len(r["objections"]) > 1 for r in splits["train"])
    assert any(len(r["objections"]) > 1 for r in splits["test"])
    assert all("NONE" not in r["objections"] for rows in splits.values() for r in rows)
    assert audit["legacy_unique_texts"] < audit["legacy_train_rows"]


def test_multilabel_metrics_do_not_reduce_to_primary():
    truth = [[1, 1] + [0] * 8]
    assert multilabel_metrics(truth, [[1, 0] + [0] * 8])["exact_match"] == 0
    assert multilabel_metrics(truth, truth)["exact_match"] == 1


@pytest.mark.parametrize("original,label", [({"anger": .9}, "ANGRY"), ({"nervousness": .8}, "ANXIOUS"),
                                           ({"gratitude": .8}, "SATISFIED"), ({}, "NEUTRAL")])
def test_emotion_mapping_is_explicit_and_complete(original, label):
    scores = map_emotions(original)
    assert set(scores) == set(EMOTIONS)
    assert max(scores, key=scores.get) == label
    assert sum(scores.values()) == pytest.approx(1)


@pytest.mark.parametrize("text,expected", [
    ("I am 35. My premium budget is INR 25,000 per year and I want coverage of 10 lakh.",
     (35, [(25000, "premium_budget", "annual"), (1000000, "coverage", "unknown")])),
    ("I am 35 years old and my budget is 2000 per month.", (35, [(2000, "premium_budget", "monthly")])),
    ("Coverage is 1.5 crore", (None, [(15000000, "coverage", "unknown")])),
    ("My annual budget is 25000", (None, [(25000, "premium_budget", "annual")])),
    ("The number is 250000", (None, [])),
    ("I am 25000", (None, [])),
])
def test_exact_numeric_rules_and_units(text, expected):
    result = entities.extract(text)
    assert result["age"] == expected[0]
    assert [(a["value"], a["kind"], a["period"]) for a in result["amounts"]] == expected[1]


def test_regex_fallback_is_visible_and_boundary_safe(monkeypatch):
    monkeypatch.setattr(entities, "language", lambda: (None, "native runtime unavailable"))
    result = entities.extract("I want individual health insurance, not a healthy lifestyle tip.")
    assert result["needs"] == ["INDIVIDUAL_HEALTH"]
    assert result["extractor"] == "Regex rules (spaCy unavailable)"
    assert result["extractor_error"]
    assert entities.extract("a healthy lifestyle")['needs'] == []


def test_exact_coverage_affects_catalogue_without_inventing_premium_band():
    session = ConversationSession("rule", age=35, nlp_backend="nb")
    result = session.reply("I need family health insurance with coverage of 15 lakh.")
    assert result["state"]["coverage_target"] == 1500000
    assert result["state"]["budget"] == "unknown"
    assert not result["facts"]
    session.reply("My premium budget is 2000 per month.")
    later = session.reply("I am willing to share my information.")
    assert later["state"]["premium_budget"]["value"] == 2000
    assert later["state"]["premium_budget"]["period"] == "monthly"


def test_audio_signal_never_supplies_consent_or_changes_policy_vector():
    from src.state_products import encode
    audio = {"status": "estimated", "label": "HAPPY", "confidence": .99}
    first = ConversationSession("rule", nlp_backend="nb").prepare_turn("hello", acoustic=audio)
    second = ConversationSession("rule", nlp_backend="nb").prepare_turn("hello")
    assert encode(first["state"]) == encode(second["state"])
    assert first["state"]["sales_stage"] != "COMMITMENT"
    stop = ConversationSession("rule", nlp_backend="nb").reply("Please stop", acoustic=audio)
    assert stop["closed"] and stop["action"] == "RESPECT_REJECTION"


def test_waveform_reaches_acoustic_model_and_silence_does_not(tmp_path, monkeypatch):
    from src import acoustic_emotion
    calls = []
    class Model:
        def predict(self, pcm, duration):
            calls.append((pcm, duration))
            return {"status": "uncertain", "source": "audio_waveform"}
    monkeypatch.setattr(acoustic_emotion, "_model", lambda: Model())
    path = tmp_path / "audio.wav"
    pcm = (np.sin(np.arange(16000) / 20) * 10000).astype("<i2").tobytes()
    write_wav(path, pcm)
    assert acoustic_emotion.analyze_audio(path)["source"] == "audio_waveform"
    assert calls == [(pcm, 1.0)]
    write_wav(path, b"\0\0" * 16000)
    assert acoustic_emotion.analyze_audio(path)["status"] == "insufficient_audio"
    assert len(calls) == 1


def test_acoustic_failure_is_visible_without_losing_transcription(tmp_path, monkeypatch):
    from src import acoustic_emotion
    def broken():
        raise RuntimeError("test acoustic failure")
    monkeypatch.setattr(acoustic_emotion, "_model", broken)
    path = tmp_path / "audio.wav"
    write_wav(path, np.full(16000, 500, dtype="<i2").tobytes())
    class ASR:
        def transcribe(self, path):
            return Transcript("hello", "hello", .9, 1, .1, "test")
    result = recognize_audio(ASR(), path)
    assert result["accepted"]
    assert result["acoustic_emotion"]["status"] == "unavailable"
    assert "test acoustic failure" in result["acoustic_emotion"]["error"]


def test_review_keeps_audio_estimate_with_corrected_transcript(tmp_path, monkeypatch):
    from demo_server import SessionStore
    from src import acoustic_emotion
    audio = {"status": "uncertain", "label": "SAD", "source": "audio_waveform"}
    monkeypatch.setattr(acoustic_emotion, "analyze_audio", lambda path: audio)
    class ASR:
        def transcribe(self, path):
            return Transcript("bad words", "bad words", .6, 1, .1, "test")
    store = SessionStore()
    store.asr = ASR()
    session = store.create({"pipeline": "voice", "policy": "rule", "generator": "template", "nlp_backend": "nb"})
    path = tmp_path / "audio.wav"
    write_wav(path, b"\0\0" * 16000)
    preview = store.reply_audio(session["id"], path.read_bytes(), "draft", review=True)
    assert preview["acoustic_emotion"] == audio
    assert store.get(session["id"]).engine.turn == 0
    entry = store.reply(session["id"], {"message": "I need family health insurance", "request_id": "confirmed", "transcript_request_id": "draft"})
    assert entry["voice"]["acoustic_emotion"] == audio
    assert entry["voice"]["corrected"]
    assert entry["response"]["state"]["acoustic_emotion"] == audio


@NATIVE
def test_real_transformer_sigmoid_emotion_and_stop_contract():
    from src.understanding import get_predictor
    predictor = get_predictor()
    result = predictor("The premium is too expensive and I don't trust this insurer.")
    assert result["classifier"]["backend"] == "transformer"
    assert {"PRICE_TOO_HIGH", "DO_NOT_TRUST_INSURER"} <= set(result["objections"])
    assert set(result["objection_probs"]) == set(OBJECTIONS[1:])
    assert set(result["emotion_probs"]) == set(EMOTIONS)
    assert len(result["text_emotion"]["original_scores"]) == 28
    assert predictor("yes I am happy to share information ask one by one")["intent"] != "REJECTION"
    assert predictor("Please stop contacting me")["intent"] == "REJECTION"


@NATIVE
def test_real_audio_emotion_does_not_need_a_transcript():
    from src.acoustic_emotion import analyze_audio
    root = Path(__file__).resolve().parents[1]
    result = analyze_audio(root / ".runtime/speech/validation/clean-0.wav")
    assert result["status"] in ("estimated", "uncertain"), result
    assert result["source"] == "audio_waveform"
    assert set(result["probabilities"]) == {"NEUTRAL", "HAPPY", "ANGRY", "SAD"}
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert not result["used_for_policy"]
