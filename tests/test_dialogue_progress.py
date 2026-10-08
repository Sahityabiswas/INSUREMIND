import pytest
import os

from src.common import ACTIONS
from src.conversation import ConversationSession
from src.generation import ResponseGenerator
from src.sim_env import action_mask
from src.voice_runtime import voice_conversation
from demo_server import APIError, SessionStore
from tests.test_ui_voice import Recognizer, wav_bytes

REPORTED = ["i am looking for cavalier saudis are my family",
            "i am looking for health insurance in my family",
            "yes i am happy to share of my information ask 1 by 1"]


@pytest.mark.parametrize("pipeline", ["text", "voice"])
@pytest.mark.parametrize("policy", ["rule", "ppo"])
def test_reported_sequence_continues_through_summary_and_explicit_stop(pipeline, policy):
    make = voice_conversation if pipeline == "voice" else ConversationSession
    session = make(policy=policy, generator="hybrid", age=35, budget="low", profile="FAMILY_ORIENTED")
    replies = [session.reply(text) for text in REPORTED]
    assert all(not reply["closed"] and not reply["llm_attempted"] and reply["validation"]["ok"] for reply in replies)
    assert replies[0]["state"]["dialogue_task"] == "clarify_need"
    assert replies[1]["state"]["dialogue_task"] == "initial_discovery"
    assert replies[2]["state"]["dialogue_task"] == "discovery_coverage"
    assert replies[2]["understanding"]["intent"] != "REJECTION"
    assert replies[2]["text"].count("?") == 1
    coverage = session.reply("500000")
    assert coverage["state"]["amount_context"]["kind"] == "coverage"
    assert coverage["state"]["dialogue_task"] == "discovery_existing"
    existing = session.reply("no")
    assert not existing["closed"] and existing["state"]["existing_coverage"] == "none"
    assert existing["state"]["workflow"]["phase"] == "ready_for_review"
    assert [p["product_id"] for p in existing["facts"]] == ["HLTH-5L"]
    summary = session.reply("yes please")
    assert summary["state"]["dialogue_task"] == "handoff_summary"
    assert "cannot issue a policy" in summary["text"]
    assert not summary["closed"] and not summary["state"]["workflow"]["policy_issued"]
    stop = session.reply("Please stop.")
    assert stop["closed"] and stop["closed_reason"] == "buyer_stop"


def test_rejection_action_is_never_allowed_without_stop_stage():
    for stage in ("GREETING", "DISCOVERY", "EVALUATION", "COMMITMENT", "OBJECTION_HANDLING"):
        assert not action_mask(stage, 5, True)[ACTIONS.index("RESPECT_REJECTION")]
    assert action_mask("REJECTION", 5, True) == [action == "RESPECT_REJECTION" for action in ACTIONS]


def test_unreliable_classifier_and_policy_cannot_close_a_cooperative_conversation():
    def bad_nlp(text):
        return {"intent": "REJECTION", "emotion": "ANGRY", "objection": "NOT_READY", "objection_conf": 1.0}
    session = ConversationSession("rule", predictor=bad_nlp, age=35)
    session.rule = lambda *args: ACTIONS.index("RESPECT_REJECTION")
    for text in ["I need family health insurance", "ask one by one", "500000", "no", "low", "yes please"]:
        response = session.reply(text)
        assert not response["closed"] and response["action"] != "RESPECT_REJECTION"
    assert session.reply("Please stop.")["closed"]


def test_missing_fields_asked_in_order_and_ambiguous_answers_do_not_advance():
    session = ConversationSession("rule", generator="template")
    first = session.reply("I need health insurance for my family, ask one by one")
    assert first["state"]["dialogue_task"] == "discovery_age"
    retry = session.reply("I did not understand")
    assert retry["state"]["dialogue_task"] == "discovery_age"
    assert retry["text"] == first["text"] and retry["validation"]["ok"]
    age = session.reply("35")
    assert age["state"]["age"] == 35 and age["state"]["dialogue_task"] == "discovery_coverage"
    assert age["state"]["amount_context"]["value"] is None
    amount = session.reply("5 lakh")
    assert amount["state"]["coverage_target"] == 500000
    assert amount["state"]["dialogue_task"] == "discovery_existing"
    assert session.reply("no")["state"]["dialogue_task"] == "discovery_budget"
    result = session.reply("medium")
    assert result["state"]["budget"] == "mid"
    assert result["state"]["workflow"]["missing_fields"] == []


def test_coverage_target_is_not_ignored_and_quote_question_does_not_complete_intake():
    session = ConversationSession("rule", generator="template", age=35, budget="low")
    session.reply("I need family health insurance, ask one by one")
    price = session.reply("What will be my payment?")
    assert price["state"]["dialogue_task"] == "explain_payment"
    assert "cannot calculate" in price["text"]
    session.reply("10 lakh")
    result = session.reply("no")
    assert result["facts"] == [] and "No synthetic option matches" in result["text"]
    assert not result["closed"]


@pytest.mark.parametrize("provider", ["ollama", "hybrid", "template"])
def test_structured_questions_never_wait_for_llm(provider):
    class NeverCall:
        def complete(self, *args):
            raise AssertionError("Guided fields must not wait for LLM wording")
    result = ResponseGenerator(provider, NeverCall()).generate("ASK_CLARIFYING_QUESTION", {"dialogue_task": "discovery_coverage"})
    assert result["text"].count("?") == 1 and not result["llm_attempted"]
    assert result["validation"]["ok"]


def test_transcript_review_does_not_mutate_memory_and_correction_is_audited():
    store = SessionStore()
    store.asr = Recognizer(REPORTED[0], .3)
    session = store.create({"pipeline": "voice", "generator": "hybrid"})
    preview = store.reply_audio(session["id"], wav_bytes(), "preview", review=True)
    record = store.get(session["id"])
    assert preview["review"] and not preview["accepted"] and record.engine.turn == 0 and not record.history
    assert store.reply_audio(session["id"], wav_bytes(), "preview", review=True) == preview
    assert store.asr.calls == 1
    corrected = store.reply(session["id"], {"message": REPORTED[1], "request_id": "confirm", "transcript_request_id": "preview"})
    assert corrected["voice"]["corrected"] and corrected["voice"]["reviewed"]
    assert corrected["voice"]["transcript"]["normalized_text"] == REPORTED[0]
    assert corrected["response"]["state"]["need"] == "FAMILY_HEALTH"
    assert not corrected["response"]["closed"]
    with pytest.raises(APIError, match="another input"):
        store.reply_audio(session["id"], wav_bytes(), "preview", review=False)


def test_review_reference_is_session_scoped():
    store = SessionStore()
    session = store.create({"generator": "template"})
    with pytest.raises(APIError, match="unavailable"):
        store.reply(session["id"], {"message": "hello", "request_id": "bad", "transcript_request_id": "other-session"})
    assert not store.get(session["id"]).history


@pytest.mark.skipif(os.name != "nt" or os.environ.get("VOICE_INTEGRATION") != "1", reason="Opt-in generated speech integration")
def test_native_guided_audio_reaches_summary_without_false_rejection(tmp_path):
    from src.speech import WindowsTTS, VoskASR, VoiceSession
    phrases = ["I need health insurance for my family.",
               "Yes, I am happy to share my information. Ask one by one.",
               "Five hundred thousand.", "No.", "Yes please."]
    jobs = [{"text": text, "output": tmp_path / f"buyer-{i}.wav"} for i, text in enumerate(phrases)]
    WindowsTTS().synthesize_many(jobs)
    session = voice_conversation(policy="ppo", generator="hybrid", age=35, budget="low")
    voice = VoiceSession(session, VoskASR(), None)
    replies = [voice.reply_audio(job["output"], tmp_path / f"reply-{i}.wav") for i, job in enumerate(jobs)]
    heard = [(r["transcript"] or {}).get("normalized_text") for r in replies]
    assert all(r["accepted"] and not r["closed"] for r in replies), heard
    assert all(r["response"]["validation"]["ok"] and not r["response"]["llm_attempted"] for r in replies), heard
    assert replies[1]["response"]["state"]["dialogue_task"] == "discovery_coverage", heard
    assert replies[-1]["response"]["state"]["workflow"]["phase"] == "summary_ready", heard
