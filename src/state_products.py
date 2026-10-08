"""Phase 4-5: state builder/encoder/memory + product/rule engine (independently testable)."""
import os, json
from collections import deque
from .common import INTENTS, EMOTIONS, NEEDS, OBJECTIONS, STAGES, PROFILES, ACTIONS, ACTION_TO_STRATEGY, onehot

PRODUCTS = [
    {"product_id": "HLTH-5L", "product_type": "FAMILY_HEALTH", "coverage": "hospitalization 5L",
     "premium": "low", "sum_insured": "5L", "eligibility": "age 18-65", "age_range": [18, 65],
     "waiting_period": "30 days", "deductible": "none", "exclusions": ["cosmetic", "pre-existing<2y"],
     "benefits": ["cashless"], "limits": "5L/year", "conditions": "standard", "source": "synthetic", "version": "v1"},
    {"product_id": "HLTH-10L", "product_type": "FAMILY_HEALTH", "coverage": "hospitalization 10L",
     "premium": "mid", "sum_insured": "10L", "eligibility": "age 18-65", "age_range": [18, 65],
     "waiting_period": "30 days", "deductible": "none", "exclusions": ["cosmetic"],
     "benefits": ["cashless", "top-up"], "limits": "10L/year", "conditions": "standard", "source": "synthetic", "version": "v1"},
    {"product_id": "LIFE-25L", "product_type": "LIFE_PROTECTION", "coverage": "term life 25L",
     "premium": "mid", "sum_insured": "25L", "eligibility": "age 18-60", "age_range": [18, 60],
     "waiting_period": "none", "deductible": "none", "exclusions": ["suicide<1y"],
     "benefits": ["nominee payout"], "limits": "25L", "conditions": "medical-declaration", "source": "synthetic", "version": "v1"},
    {"product_id": "LIFE-50L", "product_type": "LIFE_PROTECTION", "coverage": "term life 50L",
     "premium": "high", "sum_insured": "50L", "eligibility": "age 18-60", "age_range": [18, 60],
     "waiting_period": "none", "deductible": "none", "exclusions": ["suicide<1y"],
     "benefits": ["nominee payout"], "limits": "50L", "conditions": "medical-test", "source": "synthetic", "version": "v1"},
    {"product_id": "RET-STD", "product_type": "RETIREMENT", "coverage": "pension annuity",
     "premium": "mid", "sum_insured": "12L", "eligibility": "age 25-60", "age_range": [25, 60],
     "waiting_period": "5 years", "deductible": "none", "exclusions": ["early-withdrawal-fee"],
     "benefits": ["annuity"], "limits": "as-per-plan", "conditions": "standard", "source": "synthetic", "version": "v1"},
    {"product_id": "ACC-10L", "product_type": "ACCIDENT_PROTECTION", "coverage": "accident 10L",
     "premium": "low", "sum_insured": "10L", "eligibility": "age 18-70", "age_range": [18, 70],
     "waiting_period": "none", "deductible": "none", "exclusions": ["hazardous-sports"],
     "benefits": ["disability-cover"], "limits": "10L", "conditions": "standard", "source": "synthetic", "version": "v1"},
    {"product_id": "CI-15L", "product_type": "CRITICAL_ILLNESS", "coverage": "critical illness 15L",
     "premium": "mid", "sum_insured": "15L", "eligibility": "age 18-65", "age_range": [18, 65],
     "waiting_period": "90 days", "deductible": "none", "exclusions": ["pre-existing"],
     "benefits": ["lump-sum"], "limits": "15L", "conditions": "diagnosis-proof", "source": "synthetic", "version": "v1"},
    {"product_id": "CHLD-EDU", "product_type": "CHILD_PROTECTION", "coverage": "child education 10L",
     "premium": "mid", "sum_insured": "10L", "eligibility": "age 21-55", "age_range": [21, 55],
     "waiting_period": "none", "deductible": "none", "exclusions": [],
     "benefits": ["education-payout"], "limits": "10L", "conditions": "standard", "source": "synthetic", "version": "v1"},
]
BUDGET_OK = {"low": ["low"], "mid": ["low", "mid"], "high": ["low", "mid", "high"], "unknown": ["low", "mid", "high"]}

class Memory:
    def __init__(self, k=5): self.h = deque(maxlen=k)
    def add(self, s): self.h.append(s)
    def summary(self): return {"turns": len(self.h), "last_reaction": self.h[-1].get("reaction", "none") if self.h else "none"}

def build_state(nlp_out, entities, profile, stage, trust=0.5, eng=0.5, sat=0.5,
                purchase=0.3, prev_action="DISCOVER_NEEDS", prev_reaction="none", turn=0, budget="unknown",
                existing="unknown", gap="unknown"):
    return {"intent": nlp_out.get("intent", "OBJECTION"), "emotion": nlp_out.get("emotion", "NEUTRAL"),
        "emotion_probs": nlp_out.get("emotion_probs", {}), "need": (entities.get("needs") or ["GENERAL_FINANCIAL_PROTECTION"])[0],
        "budget": budget, "existing_coverage": existing, "coverage_gap": gap,
        "objection": nlp_out.get("objection", "NONE"), "severity": nlp_out.get("objection_conf", 0.0),
        "purchase_intent": purchase, "profile": profile, "sales_stage": stage, "trust": trust,
        "engagement": eng, "satisfaction": sat, "previous_strategy": ACTION_TO_STRATEGY.get(prev_action, "NEED_DISCOVERY"),
        "previous_action": prev_action, "previous_reaction": prev_reaction, "turn_count": turn}

def encode(s):
    s = {**build_state({}, {}, "HESITANT", "GREETING"), **s}
    def idx(v, lst): return lst.index(v) if v in lst else 0
    def norm(v):
        import math
        try:
            v = float(v)
            return min(max(v, 0.0), 1.0) if math.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0
    vec = (onehot(idx(s["intent"], INTENTS), len(INTENTS)) + onehot(idx(s["emotion"], EMOTIONS), len(EMOTIONS))
        + onehot(idx(s["need"], NEEDS), len(NEEDS)) + onehot(idx(s["objection"], OBJECTIONS), len(OBJECTIONS))
        + onehot(idx(s["sales_stage"], STAGES), len(STAGES)) + onehot(idx(s["profile"], PROFILES), len(PROFILES))
        + onehot(idx(s["previous_action"], ACTIONS), len(ACTIONS)))
    bmap = {"low": 0.2, "mid": 0.5, "high": 0.8, "unknown": 0.5}
    vec += [norm(s["purchase_intent"]), norm(s["trust"]), norm(s["engagement"]), norm(s["satisfaction"]), norm(s["severity"]),
            norm(float(s["turn_count"] or 0) / 20.0), bmap.get(s["budget"], 0.5),
            1.0 if s["coverage_gap"] == "yes" else 0.0]
    probs = [norm(s["emotion_probs"].get(e, 0)) for e in EMOTIONS]
    total = sum(probs)
    vec += [p / total if total else float(e == s["emotion"]) for p, e in zip(probs, EMOTIONS)]
    vec += [float(s["existing_coverage"] == c) for c in ("none", "partial-5L", "full-10L")]
    vec += [float(s["previous_reaction"] == r) for r in ("positive", "neutral", "negative")]
    return vec

STATE_DIM = 94

def ablate_observation(observation, disabled=()):
    """Remove features consistently during both policy training and evaluation."""
    import numpy as np
    out = np.asarray(observation, dtype=np.float32).copy()
    if "emotion" in disabled:
        out[9:17] = 0
        out[80:88] = 0
    if "objection" in disabled:
        out[26:37] = 0
        out[76] = 0
    return out

# ---- product / rule engine ----
def eligible(product, need, budget, age=35):
    lo, hi = product["age_range"]
    suitable = (need in (product["product_type"], "GENERAL_FINANCIAL_PROTECTION")
                or product["product_type"] == "FAMILY_HEALTH" and "HEALTH" in need)
    return suitable and product["premium"] in BUDGET_OK.get(budget, BUDGET_OK["unknown"]) and lo <= age <= hi

def recommend(need, budget="unknown", age=35, minimum_cover=None):
    def enough_cover(product):
        if minimum_cover is None:
            return True
        # Catalogue sums insured are structured lakh values, never premium quotes.
        value = product["sum_insured"]
        return value.endswith("L") and float(value[:-1]) * 100000 >= minimum_cover
    return [dict(p) for p in PRODUCTS if eligible(p, need, budget, age) and enough_cover(p)]

def validate_claim(text, facts):
    """Conservative lexical guard; open-ended semantic claims require human review."""
    import re
    pattern = r"\d[\d,]*(?:\.\d+)?\s*(?:lakh|lakhs|l|k|days?|years?|months?|%|rupees?|inr)?\b"
    def tokens(value):
        return {re.sub(r"[\s,]+", "", n.lower()).replace("lakhs", "l").replace("lakh", "l")
                for n in re.findall(pattern, value, re.I)}
    allowed = tokens(json.dumps(facts))
    bad = sorted(tokens(text) - allowed)
    for phrase in ("guaranteed approval", "guaranteed claim", "fully covered", "no exclusions", "cheapest", "best policy"):
        if phrase in text.lower():
            bad.append(phrase)
    return {"ok": not bad, "unsupported": bad}
