import os
import wave

import numpy as np
import pytest

from src.common import ACTIONS
from src.conversation import ConversationSession
from src.speech import (NoSpeechError, SpeechError, Transcript, VoiceSession,
                        normalize_spoken_text, read_wav, write_wav)
from src.voice_training import (SpeechInsuranceEnv, TranscriptChannel, edit_distance,
                                simulator_utterances, speech_metrics)


def predict(text):
    from src.nlp import stop_requested
    return {"intent": "REJECTION" if stop_requested(text) else "PRODUCT_INQUIRY",
            "emotion": "NEUTRAL", "emotion_probs": {}, "objection": "NONE", "objection_conf": 0.0}


class FakeASR:
    def __init__(self, text="I need health insurance for my family.", confidence=0.95):
        self.text, self.confidence = text, confidence

    def transcribe(self, path):
        return Transcript(self.text, self.text, self.confidence, 2.0, 0.1, "test-stub")


class FakeTTS:
    def __init__(self):
        self.text = None

    def synthesize(self, text, output):
        self.text = text
        write_wav(output, b"\0\0" * 16000)
        return str(output)


def test_wave_validation_and_silence_format(tmp_path):
    path = tmp_path / "test.wav"
    write_wav(path, b"\0\0" * 16000)
    assert read_wav(path)[1] == 1
    with pytest.raises(SpeechError, match="seconds"):
        read_wav(path, 0.5)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 16000)
    with pytest.raises(SpeechError, match="mono"):
        read_wav(path)


def test_voice_uses_core_and_same_guard(tmp_path):
    conversation = ConversationSession("rule", age=35, predictor=predict)
    tts = FakeTTS()
    voice = VoiceSession(conversation, FakeASR(), tts)
    first = voice.reply_audio(tmp_path / "input.wav", tmp_path / "reply.wav")
    assert first["accepted"] and conversation.turn == 1
    assert first["response"]["action"] in ("ASK_CLARIFYING_QUESTION", "DISCOVER_COVERAGE_GAP")
    voice.asr = FakeASR("what will be my payment")
    second = voice.reply_audio(tmp_path / "input.wav", tmp_path / "reply.wav")
    assert second["response"]["decision_reason"] == "no_verified_premium_quote"
    assert "quote" in tts.text and second["response"]["validation"]["ok"]


def test_uncertain_audio_does_not_mutate_conversation(tmp_path):
    conversation = ConversationSession("rule", predictor=predict)
    voice = VoiceSession(conversation, FakeASR(confidence=0.2), FakeTTS())
    result = voice.reply_audio(tmp_path / "in.wav", tmp_path / "out.wav")
    assert not result["accepted"] and result["response"] is None
    assert conversation.turn == 0 and len(conversation.memory.h) == 0 and conversation.need is None


def test_stop_request_wins_even_on_first_turn_low_confidence(tmp_path):
    conversation = ConversationSession("rule", predictor=predict)
    voice = VoiceSession(conversation, FakeASR("Please stop.", 0.1), FakeTTS())
    result = voice.reply_audio(tmp_path / "in.wav", tmp_path / "out.wav")
    assert result["closed"] and result["response"]["action"] == "RESPECT_REJECTION"
    with pytest.raises(RuntimeError, match="closed"):
        voice.reply_audio(tmp_path / "in.wav", tmp_path / "out.wav")


def test_silence_retry_and_tts_failure(tmp_path):
    class Silent:
        def transcribe(self, path):
            raise NoSpeechError("silence")
    class Broken:
        def synthesize(self, text, output):
            raise SpeechError("speaker unavailable")
    conversation = ConversationSession("rule", predictor=predict)
    voice = VoiceSession(conversation, Silent(), Broken())
    result = voice.reply_audio(tmp_path / "in.wav", tmp_path / "out.wav")
    assert not result["accepted"] and result["audio_path"] is None
    assert result["tts_error"] and conversation.turn == 0


def test_failed_fact_check_is_not_spoken(tmp_path):
    class Unsafe:
        closed = False
        def reply(self, text, acoustic=None):
            return {"text": "guaranteed claim", "validation": {"ok": False}}
    tts = FakeTTS()
    voice = VoiceSession(Unsafe(), FakeASR(), tts)
    voice.reply_audio(tmp_path / "in.wav", tmp_path / "out.wav")
    assert "guaranteed claim" not in tts.text


def test_distinct_audio_paths(tmp_path):
    voice = VoiceSession(ConversationSession("rule"), FakeASR(), FakeTTS())
    with pytest.raises(ValueError, match="different"):
        voice.reply_audio(tmp_path / "same.wav", tmp_path / "same.wav")


def test_prepared_turn_owner_stale_and_invalid_action():
    first, second = ConversationSession("rule", predictor=predict), ConversationSession("rule", predictor=predict)
    prepared = first.prepare_turn("hello")
    with pytest.raises(RuntimeError, match="belongs"):
        second.complete_turn(prepared)
    with pytest.raises(ValueError, match="not allowed"):
        first.complete_turn(prepared, ACTIONS.index("ASK_FOR_COMMITMENT"))
    first.complete_turn(prepared)
    with pytest.raises(RuntimeError, match="stale"):
        first.complete_turn(prepared)


@pytest.mark.parametrize("text,expected", [("around two hundred fifty thousand", "around 250000"),
    ("I am thirty five", "I am 35"), ("please stop", "please stop")])
def test_spoken_numbers(text, expected):
    pytest.importorskip("text_to_num")
    assert normalize_spoken_text(text) == expected


def test_edit_metrics_use_word_weighted_denominator():
    assert edit_distance(["one", "two"], ["one", "three"]) == 1
    assert edit_distance(["one"], []) == 1
    rows = [{"text": "one two", "transcript": {"text": "one three"}},
            {"text": "one two three four", "transcript": {"text": "one two three four"}}]
    result = speech_metrics(rows)
    assert result["wer"] == pytest.approx(1 / 6) and result["exact_match"] == 0.5


def channel_rows():
    return [{"kind": "simulator", "split": "train", "text": text,
             "transcript": {"text": text, "normalized_text": text, "confidence": 0.9}}
            for text in simulator_utterances()]


def test_channel_never_substitutes_reference_for_missing_audio():
    with pytest.raises(ValueError, match="Missing"):
        TranscriptChannel([], "test")
    channel = TranscriptChannel(channel_rows(), "train")
    with pytest.raises(ValueError, match="Uncached"):
        channel("new phrase")


def test_speech_environment_uses_observed_not_hidden_state():
    env = SpeechInsuranceEnv(TranscriptChannel(channel_rows(), "train"), predict, max_turns=3)
    obs, info = env.reset(seed=8, options={"profile": {"trust": 0.9, "purchase_intent": 0.9}})
    assert obs.shape == (94,) and env.observation_space.contains(obs)
    assert info["state"]["trust"] == 0.5 and info["state"]["purchase_intent"] == 0.3
    for _ in range(3):
        action = ACTIONS.index("ASK_CLARIFYING_QUESTION")
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward) and env.observation_space.contains(obs)
        if terminated or truncated:
            break
    assert truncated
    with pytest.raises(RuntimeError):
        env.step(action)


def test_speech_ppo_trains_and_roundtrips(tmp_path):
    from src import ppo_numpy
    env = SpeechInsuranceEnv(TranscriptChannel(channel_rows(), "train"), predict, max_turns=3)
    model, reward = ppo_numpy.train(env, total_steps=24, n_steps=8, batch_size=4, n_epochs=1)
    assert model.history[-1]["steps"] == 24 and np.isfinite(reward)
    path = str(tmp_path / "voice.npz")
    ppo_numpy.save(model, path)
    loaded = ppo_numpy.load(path, 94, 16)
    assert np.array_equal(model.W, loaded.W)


def test_recording_releases_device_on_interrupt(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.speech import record_microphone
    stopped = []
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(rec=interrupt, stop=lambda: stopped.append(True)))
    with pytest.raises(KeyboardInterrupt):
        record_microphone(tmp_path / "never.wav")
    assert stopped == [True]


@pytest.mark.skipif(os.name != "nt" or os.environ.get("VOICE_INTEGRATION") != "1",
                    reason="Opt-in native Windows TTS and installed Vosk model")
def test_native_batch_synthesis_and_actual_recognition(tmp_path):
    from src.speech import WindowsTTS, VoskASR
    from src.nlp import stop_requested
    jobs = [{"text": "I need health insurance for my family.", "output": tmp_path / "one.wav"},
            {"text": "Please stop. I am not interested.", "output": tmp_path / "two.wav"}]
    WindowsTTS().synthesize_many(jobs)
    asr = VoskASR()
    first, second = [asr.transcribe(job["output"]) for job in jobs]
    assert "insurance" in first.text and first.audio_seconds > 0
    assert stop_requested(second.normalized_text)
    assert first.model != "test-stub"


@pytest.mark.skipif(os.name != "nt" or os.environ.get("VOICE_INTEGRATION") != "1",
                    reason="Opt-in native speech and trained voice checkpoints")
def test_trained_four_turn_audio_conversation(tmp_path):
    from src.speech import ROOT, WindowsTTS, VoskASR
    from src.voice_training import load_predictor
    from src.common import load_config
    root = ROOT / load_config("configs/voice.yaml")["training"]["output_dir"]
    if not (root / "checkpoints/ppo_voice.npz").is_file():
        pytest.skip("Run train_voice.py first")
    tts = WindowsTTS()
    texts = ["I need health insurance for my family.", "Around two hundred fifty thousand.",
             "What will be my payment?", "Please stop. I am not interested."]
    jobs = [{"text": text, "output": tmp_path / f"input-{i}.wav"} for i, text in enumerate(texts)]
    tts.synthesize_many(jobs)
    conversation = ConversationSession("ppo", age=35, checkpoint=root / "checkpoints/ppo_voice.npz",
                                       predictor=load_predictor(root / "nlp_models.json"))
    session = VoiceSession(conversation, VoskASR(), tts)
    replies = [session.reply_audio(job["output"], tmp_path / f"reply-{i}.wav") for i, job in enumerate(jobs)]
    assert all(r["accepted"] and r["audio_path"] and not r["tts_error"] for r in replies)
    assert all(r["response"]["validation"]["ok"] for r in replies)
    assert replies[1]["response"]["state"]["amount_context"]["value"] == 250000
    assert replies[1]["response"]["decision_reason"] == "ambiguous_amount"
    assert replies[2]["response"]["decision_reason"] == "no_verified_premium_quote"
    assert replies[3]["closed"] and conversation.turn == 4
