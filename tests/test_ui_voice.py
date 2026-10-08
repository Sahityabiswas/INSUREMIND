"""The browser API must use the trained voice path and preserve CLI safety behavior."""
import io
import json
import os
from pathlib import Path
import wave
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import demo_server
from demo_server import APIError, SessionStore, recorded_results, runtime_status
from src.speech import NoSpeechError, Transcript
from src.voice_runtime import voice_artifacts
from tests.test_demo import demo_url, request


def wav_bytes(frames=1600):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * frames)
    return buffer.getvalue()


class Recognizer:
    def __init__(self, text="I need health insurance for my family.", confidence=.95):
        self.text, self.confidence, self.calls = text, confidence, 0

    def transcribe(self, path):
        self.calls += 1
        assert Path(path).is_file()
        if self.text is None:
            raise NoSpeechError("silence")
        return Transcript(self.text, self.text, self.confidence, .1, .01, "test-asr")


def test_ui_uses_shared_voice_artifacts_and_hybrid_defaults():
    store = SessionStore()
    session = store.create({"pipeline": "voice", "policy": "ppo"})
    record = store.get(session["id"])
    assert session["settings"]["generator"] == "hybrid"
    assert record.engine.generator.client.timeout == 20
    _, _, checkpoint, _ = voice_artifacts()
    assert runtime_status()["voice"]["checkpoint"] == checkpoint.relative_to(demo_server.ROOT).as_posix()
    entry = store.reply(session["id"], {"message": "I need health insurance for my family.", "request_id": "text"})
    assert entry["response"]["generation_route"] == "deterministic_task"
    assert not entry["response"]["llm_attempted"]
    assert entry["input_mode"] == "text"


def test_voice_replay_conflicts_and_shared_text_memory():
    store = SessionStore()
    store.asr = Recognizer()
    session = store.create({"pipeline": "voice", "generator": "template"})
    result = store.reply_audio(session["id"], wav_bytes(), "first")
    assert result["accepted"] and result["entry"]["input_mode"] == "voice"
    assert "audio_path" not in result
    assert result["entry"]["voice"]["transcript"]["confidence"] == .95
    assert store.reply_audio(session["id"], wav_bytes(), "first") == result
    assert store.asr.calls == 1
    with pytest.raises(APIError, match="another input"):
        store.reply_audio(session["id"], wav_bytes(1800), "first")
    with pytest.raises(APIError, match="another input"):
        store.reply(session["id"], {"message": "hello", "request_id": "first"})
    follow = store.reply(session["id"], {"message": "around 250000", "request_id": "second"})
    assert follow["turn"] == 2
    assert follow["response"]["state"]["need"] == "FAMILY_HEALTH"
    assert follow["response"]["decision_reason"] == "ambiguous_amount"


@pytest.mark.parametrize("heard", [None, "I would like insurance"])
def test_unusable_audio_does_not_advance_buyer_state(heard):
    store = SessionStore()
    store.asr = Recognizer(heard, .1)
    session = store.create({"pipeline": "voice", "generator": "template"})
    result = store.reply_audio(session["id"], wav_bytes(), "unclear")
    assert not result["accepted"] and result["entry"] is None
    record = store.get(session["id"])
    assert record.engine.turn == 0 and not record.history and record.engine.need is None
    assert store.reply_audio(session["id"], wav_bytes(), "unclear") == result
    assert store.asr.calls == 1


def test_low_confidence_stop_closes_and_can_replay():
    store = SessionStore()
    store.asr = Recognizer("Please stop.", .1)
    session = store.create({"pipeline": "voice", "generator": "hybrid"})
    result = store.reply_audio(session["id"], wav_bytes(), "stop")
    assert result["accepted"] and result["entry"]["response"]["closed"]
    assert result["entry"]["response"]["action"] == "RESPECT_REJECTION"
    assert store.reply_audio(session["id"], wav_bytes(), "stop") == result
    with pytest.raises(APIError, match="ended"):
        store.reply_audio(session["id"], wav_bytes(), "after-stop")


def test_invalid_audio_and_text_pipeline_rejected_before_asr():
    store = SessionStore()
    store.asr = Recognizer()
    session = store.create({"pipeline": "voice", "generator": "template"})
    with pytest.raises(APIError) as malformed:
        store.reply_audio(session["id"], b"invalid wave" * 5, "invalid")
    assert malformed.value.status == 400
    assert store.asr.calls == 0 and not store.get(session["id"]).history
    text = store.create({"pipeline": "text", "generator": "template"})
    with pytest.raises(APIError, match="voice-trained"):
        store.reply_audio(text["id"], wav_bytes(), "wrong-pipeline")
    with store.speech_lock:
        with pytest.raises(APIError, match="busy"):
            store.reply_audio(session["id"], wav_bytes(), "busy")


def test_playback_only_uses_a_stored_checked_response(monkeypatch):
    store = SessionStore()
    session = store.create({"generator": "template", "policy": "rule"})
    entry = store.reply(session["id"], {"message": "Please stop.", "request_id": "one"})
    seen = []
    def synthesize(self, text, path):
        seen.append(text)
        path.write_bytes(wav_bytes())
        return str(path)
    monkeypatch.setattr("src.speech.WindowsTTS.synthesize", synthesize)
    assert store.speak(session["id"], 1) == wav_bytes()
    assert seen == [entry["response"]["text"]]
    entry["response"]["validation"]["ok"] = False
    store.speak(session["id"], 1)
    assert "cannot verify" in seen[-1]
    for turn in (0, True, "1", 2):
        with pytest.raises(APIError):
            store.speak(session["id"], turn)


def test_api_speech_assets_and_recorded_results(demo_url):
    status = request(demo_url + "/api/status")
    assert status["voice"]["nlp_ready"] and status["voice"]["ppo_ready"]
    data = request(demo_url + "/api/results?pipeline=voice")
    config, output, _, _ = voice_artifacts()
    original = json.loads((output / "evaluation.json").read_text())
    assert data["aggregate"] == original["aggregate"]
    assert data["training"]["environment_version"] == config["training"]["environment_version"]
    for path in ("/audio.js", "/audio-worklet.js"):
        with urlopen(demo_url + path) as response:
            assert response.read() and "blob:" in response.headers["Content-Security-Policy"]
            assert response.headers["Permissions-Policy"] == "microphone=(self)"
    with pytest.raises(HTTPError) as invalid:
        request(demo_url + "/api/results?pipeline=unknown")
    assert invalid.value.code == 400


def test_audio_http_validation(demo_url):
    session = request(demo_url + "/api/sessions", {"pipeline": "voice", "generator": "template"})
    endpoint = f"{demo_url}/api/sessions/{session['id']}/audio"
    for body, headers, code in [(wav_bytes(), {"Content-Type": "audio/wav"}, 400),
                                (b"broken" * 20, {"Content-Type": "audio/wav", "X-Request-ID": "bad"}, 400),
                                (wav_bytes(), {"Content-Type": "audio/webm"}, 415),
                                (wav_bytes(), {"Content-Type": "audio/wav", "Origin": "https://example.com"}, 403)]:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(endpoint, body, headers), timeout=10)
        assert error.value.code == code
    assert not request(f"{demo_url}/api/sessions/{session['id']}")["history"]


def test_missing_results_and_artifacts_have_actionable_errors(monkeypatch, tmp_path):
    config, _, _, _ = voice_artifacts()
    monkeypatch.setattr(demo_server, "voice_artifacts", lambda: (config, tmp_path, tmp_path / "missing.npz", tmp_path / "missing.json"))
    with pytest.raises(APIError, match="train_voice.py") as error:
        recorded_results("voice")
    assert error.value.status == 404
    def missing(**kwargs):
        raise RuntimeError("Voice training artifacts missing. Run python train_voice.py first.")
    monkeypatch.setattr(demo_server, "voice_conversation", missing)
    with pytest.raises(APIError, match="train_voice.py") as error:
        SessionStore().create({"pipeline": "voice"})
    assert error.value.status == 503


@pytest.mark.skipif(os.environ.get("VOICE_INTEGRATION") != "1", reason="Opt-in native speech test")
def test_native_browser_audio_contract(demo_url, tmp_path):
    from src.speech import WindowsTTS, read_wav
    incoming = tmp_path / "buyer.wav"
    WindowsTTS().synthesize("I need health insurance for my family.", incoming)
    session = request(demo_url + "/api/sessions", {"pipeline": "voice", "generator": "hybrid"})
    endpoint = f"{demo_url}/api/sessions/{session['id']}"
    headers = {"Content-Type": "audio/wav", "X-Request-ID": "native"}
    with urlopen(Request(endpoint + "/audio", incoming.read_bytes(), headers), timeout=30) as response:
        result = json.load(response)
    assert result["accepted"] and result["entry"]["response"]["state"]["need"] == "FAMILY_HEALTH"
    assert result["entry"]["response"]["generation_route"] == "deterministic_task"
    with urlopen(Request(endpoint + "/speech", b'{"turn": 1}', {"Content-Type": "application/json"}), timeout=30) as response:
        output = tmp_path / "agent.wav"
        output.write_bytes(response.read())
    assert read_wav(output)[1] > 0
