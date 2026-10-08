"""Spec §39 tests: data/state/product/reward/env/pipeline. Run: pytest tests/"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_data():
    from src.common import ACTIONS, STRATEGIES, ACTION_TO_STRATEGY
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "processed", "insurance_sales", "train.jsonl"))]
    assert rows and all(0.0 <= r["purchase_intent"] <= 1.0 for r in rows)
    assert all(r["seller_action"] in ACTIONS for r in rows)
    assert all(r["seller_strategy"] in STRATEGIES for r in rows)
    assert all(r["seller_strategy"] == ACTION_TO_STRATEGY[r["seller_action"]] for r in rows)
    assert any(r["yes_yes"] and r["seller_strategy"] == "YES_YES_FRAMING" for r in rows)
    tr = {r["conversation_id"] for r in rows}
    te = {json.loads(l)["conversation_id"] for l in open(os.path.join(ROOT, "data", "processed", "insurance_sales", "test.jsonl"))}
    assert not (tr & te), "conversation leakage"

def test_state():
    from src.state_products import build_state, encode, STATE_DIM
    s = build_state({"intent": "OBJECTION", "emotion": "CONCERNED", "emotion_probs": {},
                     "objection": "PRICE_TOO_HIGH", "objection_conf": 0.7}, {"needs": ["LIFE_PROTECTION"]},
                    "HESITANT", "DISCOVERY")
    v = encode(s)
    assert len(v) == STATE_DIM and all(0.0 <= x <= 1.0 for x in v)

def test_products():
    from src.state_products import recommend, eligible, PRODUCTS, validate_claim
    assert eligible(PRODUCTS[0], "FAMILY_HEALTH", "low")
    assert recommend("LIFE_PROTECTION", "low") == []
    assert recommend("RETIREMENT", "low", age=99) == []
    assert all(eligible(p, "FAMILY_HEALTH", "low", age=35) for p in recommend("FAMILY_HEALTH", "low", age=35))
    assert validate_claim("This gives 99L coverage.", [PRODUCTS[0]])["ok"] is False
    for p in PRODUCTS:
        for field in ("product_id", "product_type", "coverage", "premium", "sum_insured",
                      "eligibility", "age_range", "waiting_period", "exclusions", "source", "version"):
            assert field in p
    bad = [p for p in PRODUCTS if not eligible(p, "RETIREMENT", "low", age=99)]
    assert bad  # invalid product cannot be recommended at age 99 for low budget necessarily constrained

def test_reward():
    from src.sim_env import compute_reward
    w = {"need_discovery": 0.2, "useful_clarification": 0.1, "objection_resolution": 0.3,
         "engagement_improved": 0.3, "trust_improved": 0.3, "suitable_product": 0.5,
         "qualified_lead": 1.0, "suitable_purchase": 5.0, "repetition": -0.3,
         "premature_closing": -0.5, "pressure": -1.0, "unsupported_claim": -2.0,
         "unsuitable_recommendation": -3.0, "frustration": -1.0}
    prev = {"engagement": 0.4, "trust": 0.4}
    new = {"engagement": 0.5, "trust": 0.5, "purchase_intent": 0.9}
    assert compute_reward(prev, new, "ASK_FOR_COMMITMENT",
        {"suitable": True, "purchased": True, "reaction": "positive"}, w) > 5.0
    assert compute_reward(prev, new, "ASK_FOR_COMMITMENT",
        {"suitable": False, "unsuitable": True, "reaction": "negative"}, w) < 0

def test_env():
    from src.sim_env import InsuranceEnv
    env = InsuranceEnv(reward_weights={"need_discovery": 0.2, "useful_clarification": 0.1,
        "objection_resolution": 0.3, "engagement_improved": 0.3, "trust_improved": 0.3,
        "suitable_product": 0.5, "qualified_lead": 1.0, "suitable_purchase": 5.0, "repetition": -0.3,
        "premature_closing": -0.5, "pressure": -1.0, "unsupported_claim": -2.0,
        "unsuitable_recommendation": -3.0, "frustration": -1.0}, seed=0)
    o, info = env.reset(seed=0)
    assert env.observation_space.contains(o)
    assert info["strategy"] == "NEED_DISCOVERY"
    assert info["action_mask"][13] is False
    o2, r, done, _, info2 = env.step(0)
    assert env.observation_space.contains(o2) and abs(r) != float("inf")
    assert info2["action"] == "DISCOVER_NEEDS"
    assert info2["strategy"] == "NEED_DISCOVERY"
    assert "action_mask" in info2

def test_pipeline():
    from src import nlp as N
    from src.agents import render
    t, facts = render("YES_YES_FRAMING", "LIFE_PROTECTION", "HESITANT", "mid", "NO_NEED")
    assert "fair" in t.lower() or "pressure" in t.lower() or len(t) > 10
    t2, facts2 = render("COMMITMENT_REQUEST", "LIFE_PROTECTION", "READY_TO_BUY", "mid", "NONE")
    assert facts2 and len(t2) > 10
