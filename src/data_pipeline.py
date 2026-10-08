"""Phase 1-2: external-dataset stubs + synthetic insurance dataset build/validate/split.
Uses real schemas/licenses for MultiDoGO, CaSiNo, Deal-or-No-Deal, GoEmotions, MELD,
EmpatheticDialogues, Banking77 but generates local representative samples (no heavy downloads).
Custom dataset: insurance_sales_dialogue_dataset v1 (~1200 convs, ~12k turns)."""
import os, json, random, csv, hashlib, re
from collections import Counter
from .common import (
    INTENTS,
    EMOTIONS,
    OBJECTIONS,
    NEEDS,
    PROFILES,
    STAGES,
    ACTIONS,
    STRATEGIES,
    ACTION_TO_STRATEGY,
    set_seed,
)

EXTERNAL = [
    ("multidogo", "https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset",
     "CDLA-Permissive (see upstream LICENSE.txt)", "insurance-domain dialogue structure/intent/slots"),
    ("casino", "https://github.com/kushalchawla/CaSiNo", "CC-BY-4.0", "negotiation strategy knowledge"),
    ("deal_or_no_deal", "https://github.com/facebookresearch/end-to-end-negotiator",
     "CC-BY-NC (see upstream LICENSE)", "strategic offer/accept negotiation"),
    ("goemotions", "https://github.com/google-research/google-research/tree/master/goemotions",
     "Apache-2.0", "text emotion labels"),
    ("meld", "https://github.com/declare-lab/MELD", "GPL-3.0 repository; check corpus/media terms separately", "conversational emotion w/ context"),
    ("empathetic_dialogues", "https://github.com/facebookresearch/EmpatheticDialogues", "CC-BY-NC",
     "emotionally sensitive dialogue"),
    ("banking77", "https://github.com/PolyAI-LDN/task-specific-datasets", "CC-BY-4.0", "general intent enrichment"),
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUSTOMER_TEMPLATES = {
    "PRODUCT_INQUIRY": ["What does the {need} plan cover?", "Tell me about your {need} policy options."],
    "PRICE_INQUIRY": ["What is the premium for this?", "This looks expensive, what will I pay monthly?"],
    "COVERAGE_INQUIRY": ["What is excluded from coverage?", "Is hospitalization fully covered?"],
    "EXISTING_COVERAGE": ["I already have a 5 lakh policy.", "My current insurer covers me partly."],
    "OBJECTION": ["I don't want to pay for insurance I don't need.", "The premium is too high for me."],
    "COMPARISON": ["How is this better than my current plan?", "Show me a comparison of options."],
    "PURCHASE_INTEREST": ["Okay, show me the options.", "I think I want to go ahead with this."],
    "FOLLOW_UP": ["Can I discuss with my family and get back?", "Please call me next week."],
    "REJECTION": ["No, I am not interested, please stop.", "Don't call me again about this."],
}
OBJECTION_FOR_INTENT = {"OBJECTION": None, "EXISTING_COVERAGE": "ALREADY_INSURED",
    "PRICE_INQUIRY": "PRICE_TOO_HIGH", "COMPARISON": "COMPARING_OPTIONS",
    "REJECTION": "NOT_READY", "FOLLOW_UP": "NEED_FAMILY_APPROVAL"}
EMOTION_FOR_OBJECTION = {"PRICE_TOO_HIGH": "CONCERNED", "ALREADY_INSURED": "SKEPTICAL",
    "DO_NOT_TRUST_INSURER": "SKEPTICAL", "BAD_PAST_EXPERIENCE": "FRUSTRATED",
    "NEED_MORE_INFORMATION": "ANXIOUS", "COVERAGE_NOT_CLEAR": "CONCERNED"}
AGENT_TEMPLATES = {
    "DISCOVER_NEEDS": "Could you share who this cover is mainly for and what worries you most?",
    "ASK_CLARIFYING_QUESTION": "Just to clarify, what matters more to you — lower premium or wider coverage?",
    "DISCOVER_COVERAGE_GAP": "Let's look at what your current coverage already provides and check for any real gap.",
    "EXPLAIN_PRODUCT": "This plan covers {need} with sum insured {si}, waiting period {wp}.",
    "VALUE_FRAMING": "For a small monthly amount you secure a much larger safety net for your family.",
    "RISK_EXPLANATION": "Without top-up cover, one hospitalization could wipe out savings; this fills that gap.",
    "AFFORDABILITY_FRAMING": "Within your budget we have a {si} option at a lower premium slab.",
    "HANDLE_OBJECTION": "I understand your concern about {obj}; let me address it directly.",
    "YES_YES_FRAMING": "That's completely fair — you shouldn't pay for cover you don't need. Let's first verify the actual gap, then decide together.",
    "BUILD_TRUST": "No pressure at all — I'll share only verifiable terms, exclusions and waiting periods.",
    "PERSONALIZE": "Given your {profile} profile and {need} need, this option fits best.",
    "COMPARE_OPTIONS": "Comparing factually: option A has {si} cover, option B is cheaper with narrower scope.",
    "OFFER_ALTERNATIVE": "If that premium is high, here is a lower-cover alternative within budget.",
    "ASK_FOR_COMMITMENT": "Would you like to proceed with a qualified lead summary for this option?",
    "FOLLOW_UP": "Shall I follow up next week after you've discussed with family?",
    "RESPECT_REJECTION": "Understood, I'll close here. Thank you for your time.",
}
RULE_NEXT = {  # oracle-ish policy used to generate plausible supervised labels
    "PRICE_TOO_HIGH": "AFFORDABILITY_FRAMING", "ALREADY_INSURED": "DISCOVER_COVERAGE_GAP",
    "NO_NEED": "DISCOVER_NEEDS", "DO_NOT_TRUST_INSURER": "BUILD_TRUST",
    "NEED_MORE_INFORMATION": "EXPLAIN_PRODUCT", "COMPARING_OPTIONS": "COMPARE_OPTIONS",
    "BAD_PAST_EXPERIENCE": "BUILD_TRUST", "COVERAGE_NOT_CLEAR": "EXPLAIN_PRODUCT",
    "NEED_FAMILY_APPROVAL": "FOLLOW_UP", "NOT_READY": "RESPECT_REJECTION", "NONE": "PERSONALIZE"}

def write_external_stubs():
    for name, url, lic, purpose in EXTERNAL:
        d = os.path.join(ROOT, "data", "raw", name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(f"# {name}\nSource: {url}\nLicense: {lic}\nPurpose: {purpose}\n"
                    "Metadata only. No public examples downloaded or used by the offline experiment.\n")
        with open(os.path.join(d, "sample.jsonl"), "w") as f:
            f.write(json.dumps({"source_dataset": name, "license_reference": lic,
                "source_type": "metadata", "note": "Not a dialogue or training example; see README for corpus URL"}) + "\n")

def gen_conversation(cid, rng):
    need = rng.choice(NEEDS); profile = rng.choice(PROFILES)
    budget = rng.choice(["low", "mid", "high"])
    existing = rng.choice(["none", "partial-5L", "full-10L"])
    gap = "yes" if existing in ("none", "partial-5L") else rng.choice(["yes", "no"])
    n_turns = rng.randint(8, 12)
    scenario_id = "|".join([need, profile, budget, existing, gap])
    intent_flow = ["PRODUCT_INQUIRY", "PRICE_INQUIRY", "EXISTING_COVERAGE", "OBJECTION",
                   "COMPARISON", "PURCHASE_INTEREST" if gap == "yes" else "FOLLOW_UP"]
    turns, purchase = [], 0.3
    for t in range(n_turns):
        stage = STAGES[min(t * len(STAGES) // n_turns, len(STAGES) - 1)]
        if t == n_turns - 1:
            intent = "REJECTION" if rng.random() < 0.12 else ("PURCHASE_INTEREST" if gap == "yes" and rng.random() < 0.6 else "FOLLOW_UP")
        else:
            intent = intent_flow[min(t, len(intent_flow) - 1)] if rng.random() < 0.7 else rng.choice(INTENTS)
        obj = OBJECTION_FOR_INTENT.get(intent, "NONE")
        if intent == "OBJECTION":
            obj = rng.choice([o for o in OBJECTIONS if o != "NONE"])
        emotion = EMOTION_FOR_OBJECTION.get(obj, rng.choice(["NEUTRAL", "POSITIVE", "CONCERNED", "ANXIOUS"]))
        if intent in ("PURCHASE_INTEREST",): emotion = rng.choice(["POSITIVE", "SATISFIED"])
        if intent == "REJECTION": emotion = rng.choice(["ANGRY", "FRUSTRATED"])
        action = RULE_NEXT.get(obj, "PERSONALIZE")
        if stage == "GREETING": action = rng.choice(["DISCOVER_NEEDS", "ASK_CLARIFYING_QUESTION"])
        if intent == "PURCHASE_INTEREST" and stage in ("EVALUATION", "COMMITMENT", "PURCHASE"):
            action = "ASK_FOR_COMMITMENT"
        if intent == "REJECTION": action = "RESPECT_REJECTION"
        if obj == "DO_NOT_TRUST_INSURER" and rng.random() < 0.5: action = "YES_YES_FRAMING"
        strategy = ACTION_TO_STRATEGY[action]
        objection_text = {"PRICE_TOO_HIGH": "The premium is too high for me.",
            "ALREADY_INSURED": "I already have insurance; would this duplicate it?",
            "NO_NEED": "I don't need more insurance.", "DO_NOT_TRUST_INSURER": "I don't trust this insurer.",
            "NEED_MORE_INFORMATION": "I need more information before deciding.",
            "COMPARING_OPTIONS": "I want to compare options first.",
            "BAD_PAST_EXPERIENCE": "I had a bad experience with an insurer in the past.",
            "COVERAGE_NOT_CLEAR": "The exclusions and coverage are not clear.",
            "NEED_FAMILY_APPROVAL": "I need my family's approval first.", "NOT_READY": "I'm not ready to decide."}
        ctext = (objection_text[obj] if intent == "OBJECTION" else
                 rng.choice(CUSTOMER_TEMPLATES[intent]).format(need=need.lower().replace("_", " ")))
        from .generation import ResponseGenerator
        atext = ResponseGenerator().generate(action, {"need": need, "budget": budget, "profile": profile})["text"]
        reaction = rng.choice(["positive", "neutral"]) if action in (
            "AFFORDABILITY_FRAMING", "BUILD_TRUST", "YES_YES_FRAMING", "PERSONALIZE") else rng.choice(["positive", "neutral", "negative"])
        before_purchase = purchase
        purchase = max(0.0, min(1.0, purchase + (0.08 if reaction == "positive" else (-0.08 if reaction == "negative" else 0.0))))
        turns.append({"conversation_id": cid, "turn_id": t, "speaker": "customer", "text": ctext,
            "source_type": "synthetic", "source_dataset": "scenario-generator",
            "license_reference": "synthetic-internal", "customer_intent": intent, "customer_emotion": emotion,
            "customer_profile_id": profile, "customer_need": need, "budget": budget,
            "existing_coverage": existing, "coverage_gap": gap, "objection_type": obj,
            "objection_severity": round(rng.uniform(0.5, 0.9) if obj != "NONE" else 0.0, 2),
            "sales_stage": stage, "seller_action": action, "seller_strategy": strategy,
            "yes_yes": strategy == "YES_YES_FRAMING",
            "yes_yes_stage": "REFRAME" if strategy == "YES_YES_FRAMING" else "",
            "customer_reaction": reaction, "objection_status": "resolved" if reaction == "positive" and obj != "NONE" else ("open" if obj != "NONE" else "none"),
            "purchase_intent": round(purchase, 2), "purchase_intent_change": round(purchase - before_purchase, 2),
            "outcome": "pending"})
        turns.append({"conversation_id": cid, "turn_id": t, "speaker": "agent", "text": atext,
            "source_type": "synthetic", "source_dataset": "scenario-generator",
            "license_reference": "synthetic-internal", "customer_intent": intent, "customer_emotion": emotion,
            "customer_profile_id": profile, "customer_need": need, "budget": budget,
            "existing_coverage": existing, "coverage_gap": gap, "objection_type": obj,
            "objection_severity": turns[-1]["objection_severity"] if turns else 0.0,
            "sales_stage": stage, "seller_action": action, "seller_strategy": strategy,
            "yes_yes": strategy == "YES_YES_FRAMING",
            "yes_yes_stage": "REFRAME" if strategy == "YES_YES_FRAMING" else "",
            "customer_reaction": reaction, "objection_status": turns[-1]["objection_status"],
            "purchase_intent": round(purchase, 2), "purchase_intent_change": round(purchase - before_purchase, 2),
            "outcome": "pending"})
        for row in turns[-2:]:
            row.update(scenario_id=scenario_id, dataset_version="v2", annotation_source="synthetic_rules",
                       outcome_source="synthetic_rule", labels_observed=False)
        if intent == "REJECTION":
            break
    outcome = "rejected" if intent == "REJECTION" else "purchase" if purchase > 0.7 else "follow_up" if purchase > 0.45 else "rejected"
    for row in turns[-2:]:
        row["outcome"] = outcome
    return turns

def build_dataset(n=1200, seed=42):
    set_seed(seed)
    rng = random.Random(seed)
    rows = []
    for cid in range(n):
        rows.extend(gen_conversation(f"conv_{cid:04d}", rng))
    return rows

def validate(rows):
    errs = []
    seen = set()
    for r in rows:
        required = ("conversation_id", "turn_id", "speaker", "text", "source_type", "seller_action",
                    "seller_strategy", "customer_intent", "customer_emotion", "objection_type", "sales_stage",
                    "purchase_intent", "objection_severity")
        missing = [k for k in required if k not in r]
        if missing:
            errs.append(f"missing fields: {missing}")
            continue
        key = (r["conversation_id"], r["turn_id"], r["speaker"])
        if key in seen: errs.append(f"dup {key}")
        seen.add(key)
        for k, tax in [("seller_action", ACTIONS), ("seller_strategy", STRATEGIES), ("customer_intent", INTENTS),
                       ("customer_emotion", EMOTIONS), ("objection_type", OBJECTIONS), ("sales_stage", STAGES)]:
            if r[k] not in tax: errs.append(f"bad {k}={r[k]}")
        if r["speaker"] not in ("customer", "agent") or not isinstance(r["text"], str) or not r["text"].strip():
            errs.append("invalid speaker/text")
        if r["source_type"] not in ("public", "manual_annotation", "synthetic", "simulated", "derived"):
            errs.append("invalid provenance")
        if ACTION_TO_STRATEGY.get(r["seller_action"]) != r["seller_strategy"]:
            errs.append("action/strategy mismatch")
        for field in ("purchase_intent", "objection_severity"):
            if not isinstance(r[field], (float, int)) or not 0.0 <= r[field] <= 1.0:
                errs.append(f"bad {field}")
    return errs

def save_and_split(rows, seed=42, train_ratio=0.7, val_ratio=0.15, n_profiles=3000):
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios")
    proc = os.path.join(ROOT, "data", "processed", "insurance_sales")
    os.makedirs(proc, exist_ok=True)
    cids = sorted({r["conversation_id"] for r in rows})
    rng = random.Random(seed)
    from collections import defaultdict
    conversations = defaultdict(list)
    for r in rows:
        conversations[r["conversation_id"]].append(r)
    # Union scenario variants and identical normalized transcripts into indivisible groups.
    parent = {cid: cid for cid in cids}
    def find(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid
    signatures, scenarios, exact_duplicates = {}, {}, 0
    for cid, conv in conversations.items():
        signature = hashlib.sha256("|".join(re.sub(r"\W+", " ", r["text"].lower()).strip()
                                           for r in conv).encode()).hexdigest()
        scenario = conv[0].get("scenario_id", cid)
        for lookup, key in ((signatures, signature), (scenarios, scenario)):
            if key in lookup:
                parent[find(cid)] = find(lookup[key])
                if lookup is signatures:
                    exact_duplicates += 1
            lookup[key] = cid
    groups = defaultdict(set)
    for cid in cids:
        groups[find(cid)].add(cid)
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    n = len(cids)
    tr, va = int(train_ratio * len(group_ids)), int((train_ratio + val_ratio) * len(group_ids))
    splits = {"train": set(), "val": set(), "test": set()}
    for name, ids in (("train", group_ids[:tr]), ("val", group_ids[tr:va]), ("test", group_ids[va:])):
        for group in ids:
            splits[name].update(groups[group])
    for name, ids in splits.items():
        with open(os.path.join(proc, f"{name}.jsonl"), "w") as f:
            for r in rows:
                if r["conversation_id"] in ids:
                    f.write(json.dumps(r) + "\n")
    df_n = len(rows)
    stats = {"conversations": n, "turns": df_n,
             "source_mix": dict(Counter(r["source_type"] for r in rows)),
             "source_ratios": {k: v / max(df_n, 1) for k, v in Counter(r["source_type"] for r in rows).items()},
             "splits": {k: len(v) for k, v in splits.items()}, "split_unit": "scenario_and_normalized_transcript",
             "scenario_groups": len(group_ids), "duplicate_transcripts_grouped": exact_duplicates,
             "repeated_turn_texts": df_n - len({r["text"] for r in rows}),
             "near_duplicate_policy": "All variants sharing need/profile/budget/coverage/gap are kept in one split",
             "version": "v2", "profiles": n_profiles}
    with open(os.path.join(proc, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # derived objection + profile datasets (spec §10-11)
    obj = [r for r in rows if r["speaker"] == "customer" and r["objection_type"] != "NONE"]
    responses = {(r["conversation_id"], r["turn_id"]): r["text"] for r in rows if r["speaker"] == "agent"}
    with open(os.path.join(ROOT, "data", "processed", "objections.jsonl"), "w") as f:
        for i, r in enumerate(obj[:5000]):
            f.write(json.dumps({"objection_id": i, "conversation_id": r["conversation_id"],
                "objection_type": r["objection_type"], "customer_emotion": r["customer_emotion"],
                "customer_profile": r["customer_profile_id"], "seller_response": responses[(r["conversation_id"], r["turn_id"])],
                "seller_strategy": r["seller_strategy"], "customer_reaction": r["customer_reaction"],
                "response_quality": "synthetic_unrated", "source_type": "derived", "derived_from": "synthetic",
                "objection_resolved": r["objection_status"] == "resolved"}) + "\n")
    with open(os.path.join(ROOT, "data", "processed", "profiles.jsonl"), "w") as f:
        for i in range(n_profiles):
            age = rng.randint(18, 75)
            f.write(json.dumps({"customer_id": f"cust_{i:04d}", "synthetic": True,
                "source_type": "synthetic", "age": age,
                "age_group": "18-30" if age <= 30 else "31-45" if age <= 45 else "46-60" if age <= 60 else "61-75",
                "family_size": rng.randint(1, 6), "income_band": rng.choice(["low", "mid", "high"]),
                "budget": rng.choice(["low", "mid", "high"]),
                "existing_coverage": rng.choice(["none", "partial-5L", "full-10L"]),
                "insurance_experience": rng.choice(["new", "experienced", "negative"]),
                "risk_attitude": rng.choice(["low", "mid", "high"]), "patience": rng.randint(4, 12),
                "primary_need": NEEDS[i % len(NEEDS)], "personality": PROFILES[i % len(PROFILES)],
                "initial_trust": round(0.3 + (i % 5) * 0.1, 2),
                "initial_purchase_intent": 0.3}) + "\n")
    return stats

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1200); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    write_external_stubs()
    rows = build_dataset(a.n, a.seed)
    errs = validate(rows)
    assert not errs, errs[:5]
    print(save_and_split(rows, a.seed))

if __name__ == "__main__":
    main()
