import io
import json

import pytest

from src.conversation import ConversationSession
from src.generation import OllamaClient, ResponseGenerator, validate_response
from src.nlp import live_cues


@pytest.mark.parametrize("policy", ["ppo", "rule"])
def test_reported_three_turn_sequence(policy):
    session = ConversationSession(policy, "template", profile="FAMILY_ORIENTED", age=35, budget="low")
    first = session.reply("I need health insurance for my family.")
    assert first["understanding"]["intent"] == "PRODUCT_INQUIRY"
    assert first["understanding"]["objection"] == "NONE"
    assert first["action"] in ("ASK_CLARIFYING_QUESTION", "DISCOVER_COVERAGE_GAP")
    assert first["decision_reason"] == "initial_discovery_mask"
    assert "?" in first["text"] and first["validation"]["ok"]
    amount = session.reply("around 250000")
    assert amount["action"] == "ASK_CLARIFYING_QUESTION"
    assert amount["decision_source"] == "conversation_guard"
    assert amount["state"]["amount_context"] == {"value": 250000, "kind": "unknown", "period": "unknown"}
    assert amount["state"]["budget"] == "low"
    assert "coverage" in amount["text"] and "budget" in amount["text"]
    payment = session.reply("what will be my payment")
    assert payment["decision_reason"] == "no_verified_premium_quote"
    assert payment["understanding"]["objection"] == "NONE"
    assert "cannot calculate" in payment["text"]
    assert "quote" in payment["text"] and "rupee" in payment["text"]
    assert payment["validation"]["ok"] and payment["text"] != amount["text"]
    assert not payment["closed"]


def test_amount_clarification_does_not_guess_a_budget_band():
    session = ConversationSession("rule", "template", age=35, budget="low")
    session.reply("I need family health insurance.")
    session.reply("about 2.5 lakh")
    clarified = session.reply("coverage")
    assert clarified["state"]["amount_context"]["kind"] == "coverage"
    assert clarified["state"]["amount_context"]["value"] == 250000
    assert clarified["state"]["budget"] == "low"
    stop = session.reply("Please stop. I am not interested.")
    assert stop["action"] == "RESPECT_REJECTION" and stop["closed"]


def test_real_family_approval_is_not_suppressed():
    session = ConversationSession("rule", "template", age=35, budget="low")
    response = session.reply("I need family health insurance but must discuss with my family.")
    assert response["understanding"]["objection"] == "NEED_FAMILY_APPROVAL"
    assert response["action"] == "FOLLOW_UP"


@pytest.mark.parametrize("text,amount", [("around 250000", 250000), ("about 2.5 lakh", 250000),
                                         ("INR 2,50,000", 250000), ("I am 35", None), ("what is my payment", None)])
def test_live_amount_cues(text, amount):
    assert live_cues(text)["amount"] == amount


def test_buyer_role_and_echo_are_rejected_with_diagnostics():
    class BuyerClient:
        def complete(self, system, payload):
            assert "insurance adviser, not the buyer" in system
            return {"strategy": payload["strategy"], "text": "I need health insurance for my family. Can you help me find a plan that fits our budget?"}

    response = ResponseGenerator("ollama", BuyerClient()).generate("ASK_CLARIFYING_QUESTION", {},
        customer_text="I need health insurance for my family.")
    assert response["source"] == "template" and response["llm_rejected"]
    assert "buyer_role" in response["llm_rejection_reasons"]
    assert "customer_echo" in response["llm_rejection_reasons"]
    assert response["validation"]["ok"]
    assert validate_response("I can help you review the documented terms.", [])["ok"]


def test_fallback_does_not_repeat_recent_product_response():
    generator = ResponseGenerator("template")
    history = []
    for _ in range(15):
        response = generator.generate("PERSONALIZE", {"need": "FAMILY_HEALTH", "budget": "low", "age": 35}, history)
        assert response["text"] not in history[-5:]
        assert response["validation"]["ok"]
        history.append(response["text"])


def test_payment_guard_rejects_missing_price_limitation():
    class BadPriceClient:
        def complete(self, system, payload):
            return {"strategy": payload["strategy"], "text": "The premium is low and you should proceed."}

    response = ResponseGenerator("ollama", BadPriceClient()).generate("AFFORDABILITY_FRAMING",
        {"need": "FAMILY_HEALTH", "budget": "low", "age": 35, "dialogue_task": "explain_payment"})
    assert "missing_price_limitation" in response["llm_rejection_reasons"]
    assert "cannot calculate" in response["text"] and response["validation"]["ok"]
    numeric = validate_response("No verified quote exists, but your payment is 65 per month.",
        [{"eligibility": "age 18-65"}], task="explain_payment")
    assert "unverified_payment_amount" in numeric["task_violations"] and not numeric["ok"]


def test_supplied_amount_is_allowed_only_for_clarification_not_a_quote():
    for value in ("250000", "250,000", "2,50,000"):
        check = validate_response(f"Does your amount of {value} mean coverage or your premium budget?", [],
            task="clarify_amount", clarification_amount=250000)
        assert check["ok"]
    for quote_text in ("Your premium is 250000.", "Your payment is about \u20b9250000.", "You will pay INR 250000."):
        invented = validate_response(quote_text + " Is that coverage or your budget?", [],
            task="clarify_amount", clarification_amount=250000)
        assert "assumed_payment_quote" in invented["task_violations"] and not invented["ok"]
    wrong_value = validate_response("Does your amount of 350000 mean coverage or premium budget?", [],
        task="clarify_amount", clarification_amount=250000)
    assert not wrong_value["ok"] and wrong_value["unsupported"]
    quote = validate_response("Your monthly payment is 250000.", [], task="explain_payment", clarification_amount=250000)
    assert not quote["ok"]


def test_ollama_uses_roles_and_schema_without_breaking_direct_policy(monkeypatch):
    from src import generation
    requests = []

    def fake_open(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        content = {"strategy": "CLARIFYING_QUESTION", "text": "What coverage do you need?"} if isinstance(body["format"], dict) else {"action": "DISCOVER_NEEDS"}
        return io.StringIO(json.dumps({"message": {"content": json.dumps(content)}, "done_reason": "stop"}))

    monkeypatch.setattr(generation, "urlopen", fake_open)
    client = OllamaClient("llama3.2:3b")
    client.complete("You are an adviser", {"strategy": "CLARIFYING_QUESTION", "customer_text": "around 250000",
        "dialogue": [{"customer": "I need insurance", "agent": "How much cover?"}], "context": {"need": "FAMILY_HEALTH"}})
    first = requests[0]
    assert [message["role"] for message in first["messages"]] == ["system", "user", "assistant", "user"]
    assert first["messages"][-1]["content"] == "around 250000"
    assert first["format"]["properties"]["strategy"]["enum"] == ["CLARIFYING_QUESTION"]
    assert first["options"]["num_predict"] == 160 and first["options"]["num_ctx"] == 2048
    result = client.complete("Choose an action", {"allowed_actions": ["DISCOVER_NEEDS"]})
    assert requests[1]["format"] == "json" and result["action"] == "DISCOVER_NEEDS"
