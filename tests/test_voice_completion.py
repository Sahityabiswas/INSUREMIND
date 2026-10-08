import io
import json
import sys
from types import SimpleNamespace

import pytest

from src.common import ACTIONS
from src.conversation import ConversationSession
from src.generation import OllamaClient, ResponseGenerator
from src.nlp import live_cues
from src.sim_env import CustomerSim
from src.speech import VoskASR, read_wav
from src.voice_training import SpeechInsuranceEnv, TranscriptChannel, simulator_utterances


def neutral(text):
    return {"intent": "PRODUCT_INQUIRY", "emotion": "NEUTRAL", "objection": "NONE", "objection_conf": 0.0}


@pytest.mark.parametrize("text", ["I am ready to proceed", "I would like to apply", "We want to buy", "Let's proceed"])
def test_explicit_affirmative_interest(text):
    assert live_cues(text)["ready_to_proceed"]


@pytest.mark.parametrize("text", ["I am not ready to proceed", "I am ready to proceed if it is cheaper",
    "Maybe I would like to apply", "I don't want to buy", "I want to buy but I am not sure"])
def test_negated_and_conditional_interest_is_not_consent(text):
    assert not live_cues(text)["ready_to_proceed"]


def test_commitment_requires_buyer_interest_not_elapsed_turns():
    session = ConversationSession("rule", age=35, predictor=neutral)
    for text in ("I need health insurance for my family", "low budget", "Explain the options"):
        session.reply(text)
    ordinary = session.prepare_turn("That makes sense, tell me more")
    assert not ordinary["mask"][ACTIONS.index("ASK_FOR_COMMITMENT")]
    session.complete_turn(ordinary)
    ready = session.reply("I understand the terms and I am ready to proceed")
    assert ready["state"]["sales_stage"] == "COMMITMENT"
    assert ready["action"] == "ASK_FOR_COMMITMENT"
    assert ready["understanding"]["intent"] == "PURCHASE_INTEREST"
    stop = session.reply("Please stop. I am not interested.")
    assert stop["closed"] and stop["action"] == "RESPECT_REJECTION"


def test_unresolved_objection_overrides_interest():
    session = ConversationSession("rule", age=35, predictor=neutral)
    for text in ("I need family health insurance", "low budget", "Explain the options"):
        session.reply(text)
    result = session.reply("I am ready to proceed but this is too expensive")
    assert result["state"]["sales_stage"] == "OBJECTION_HANDLING"
    assert result["action"] != "ASK_FOR_COMMITMENT"


def test_observable_dialogue_does_not_change_hidden_transition_rules():
    old, new = CustomerSim(12), CustomerSim(12, observable_dialogue=True)
    profile = {"personality": "READY_TO_BUY", "primary_need": "FAMILY_HEALTH", "age": 35,
               "budget": "low", "purchase_intent": 0.95, "objection": "NONE", "patience": 20}
    assert old.reset(profile) == new.reset(profile)
    differences = 0
    for action in ("DISCOVER_NEEDS", "EXPLAIN_PRODUCT", "BUILD_TRUST", "ASK_FOR_COMMITMENT"):
        a, b = old.step(action), new.step(action)
        assert old.s == new.s and a[1:] == b[1:]
        differences += a[0] != b[0]
    assert differences > 0


def test_readiness_must_arrive_through_recognized_words():
    rows = [{"kind": "simulator", "split": "test", "text": text,
             "transcript": {"text": text, "normalized_text": text, "confidence": 0.99}}
            for text in simulator_utterances(True)]
    for row in rows:
        if "ready to proceed" in row["text"]:
            row["transcript"]["normalized_text"] = "I need more time to think about it"
    env = SpeechInsuranceEnv(TranscriptChannel(rows, "test", True), neutral, observable_dialogue=True)
    env.reset(seed=12, options={"profile": {"personality": "READY_TO_BUY", "primary_need": "FAMILY_HEALTH",
        "age": 35, "budget": "low", "purchase_intent": 1.0, "objection": "NONE", "patience": 20}})
    env.step(ACTIONS.index("DISCOVER_NEEDS"))
    # The first recognized insurance need now completes discovery before product education.
    env.step(ACTIONS.index("ASK_CLARIFYING_QUESTION"))
    env.step(ACTIONS.index("EXPLAIN_PRODUCT"))
    _, _, _, _, info = env.step(ACTIONS.index("BUILD_TRUST"))
    assert env.env.sim.s["purchase_intent"] > 0.8
    assert info["sales_stage"] != "COMMITMENT"
    assert not info["action_mask"][ACTIONS.index("ASK_FOR_COMMITMENT")]


@pytest.mark.parametrize("task", ["initial_discovery", "clarify_amount", "explain_payment"])
def test_hybrid_routes_deterministic_tasks_without_contacting_llm(task):
    class NeverCall:
        def complete(self, *args):
            raise AssertionError("No model call should be made")
    result = ResponseGenerator("hybrid", NeverCall()).generate("ASK_CLARIFYING_QUESTION", {"dialogue_task": task})
    assert not result["llm_attempted"] and result["source"] == "template"
    assert result["generation_route"] == "deterministic_task" and result["validation"]["ok"]


def test_hybrid_open_response_uses_llm_and_timeout_is_explicit():
    class Client:
        last_metrics = {"eval_count": 12}
        def complete(self, system, payload):
            return {"strategy": payload["strategy"], "text": "We can review the documented exclusions together."}
    ok = ResponseGenerator("hybrid", Client()).generate("HANDLE_OBJECTION", {}, customer_text="Explain exclusions")
    assert ok["source"] == "ollama" and ok["llm_attempted"] and ok["llm_metrics"]["eval_count"] == 12
    class Timeout:
        def complete(self, *args):
            raise TimeoutError("test timeout")
    failed = ResponseGenerator("hybrid", Timeout()).generate("HANDLE_OBJECTION", {})
    assert failed["source"] == "template" and failed["generation_route"] == "llm_fallback"
    assert failed["llm_error"] == "TimeoutError" and failed["validation"]["ok"]


def test_compact_profile_and_server_timings(monkeypatch):
    from src import generation
    bodies = []
    def open_request(request, timeout):
        assert timeout == 20
        bodies.append(json.loads(request.data))
        return io.StringIO(json.dumps({"done_reason": "stop", "load_duration": 123,
            "eval_count": 8, "message": {"content": '{"strategy":"OBJECTION_HANDLING","text":"Let us review the exclusions."}'}}))
    monkeypatch.delenv("INSURANCE_VOICE_LLM_TIMEOUT", raising=False)
    monkeypatch.setattr(generation, "urlopen", open_request)
    client = OllamaClient("local-model", profile="voice")
    client.complete("adviser", {"strategy": "OBJECTION_HANDLING", "dialogue": [
        {"customer": "old", "agent": "old response"}, {"customer": "new", "agent": "new response"}], "customer_text": "question"})
    assert len(bodies[0]["messages"]) == 4
    assert bodies[0]["options"]["num_predict"] == 96
    assert bodies[0]["options"]["num_ctx"] == 1536
    assert client.last_metrics == {"load_duration": 123, "eval_count": 8}


def test_microphone_endpoint_closes_stream_early(tmp_path, monkeypatch):
    closed, reads = [], []
    class Stream:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
        def read(self, frames):
            reads.append(frames)
            return b"\0\0" * frames, False
    class Recognizer:
        def AcceptWaveform(self, chunk):
            return len(reads) >= 2
        def Result(self):
            return '{"text":"hello"}'
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(RawInputStream=lambda **kwargs: Stream()))
    monkeypatch.setitem(sys.modules, "vosk", SimpleNamespace(KaldiRecognizer=lambda *args: Recognizer()))
    asr = VoskASR.__new__(VoskASR)
    asr.model, asr.max_seconds = object(), 30
    path = tmp_path / "endpoint.wav"
    assert asr.record_utterance(path, max_seconds=8) == pytest.approx(0.2)
    assert read_wav(path)[1] == pytest.approx(0.2) and closed == [True]
