"""spaCy phrase entities and explicit, unit-preserving numeric rules. No statistical NER."""
from decimal import Decimal
from functools import lru_cache
import re

PHRASES = {
    "FAMILY_HEALTH": ["health", "health insurance", "medical insurance", "hospitalization", "hospital cover"],
    "LIFE_PROTECTION": ["life insurance", "life protection", "life policy"],
    "CHILD_PROTECTION": ["child protection", "children's protection", "child insurance"],
    "RETIREMENT": ["retirement", "pension"],
    "INCOME_PROTECTION": ["income protection", "income insurance"],
    "CRITICAL_ILLNESS": ["critical illness"],
    "ACCIDENT_PROTECTION": ["accident insurance", "accident protection", "personal accident"],
}
NUMBER = r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>crores?|lakhs?|thousand|l|k)?\b"
AMOUNT = re.compile(NUMBER, re.I)
CUES = re.compile(r"\b(?P<premium_budget>premium|budget|payment)\b|\b(?P<coverage>coverage|cover|sum insured)\b", re.I)


@lru_cache(maxsize=1)
def language():
    try:
        import spacy
        pipeline = spacy.blank("en")
        ruler = pipeline.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
        ruler.add_patterns([{"label": "INSURANCE_NEED", "pattern": phrase, "id": label}
                            for label, phrases in PHRASES.items() for phrase in phrases])
        return pipeline, None
    except (ImportError, OSError) as exc:
        return None, str(exc)


def numeric_value(match):
    value = Decimal(match.group("value").replace(",", ""))
    unit = (match.group("unit") or "").lower()
    value *= 10000000 if unit.startswith("crore") else 100000 if unit.startswith("l") else 1000 if unit in ("k", "thousand") else 1
    return float(value) if value != value.to_integral_value() else int(value)


def extract(text):
    pipeline, error = language()
    if pipeline is not None:
        doc = pipeline(text)
        spans = [{"text": e.text, "label": e.label_, "value": e.ent_id_, "start": e.start_char, "end": e.end_char} for e in doc.ents]
    else:
        # The same phrase vocabulary is usable on hosts that disallow spaCy's native DLLs.
        candidates = [{"text": m.group(), "label": "INSURANCE_NEED", "value": label, "start": m.start(), "end": m.end()}
                      for label, phrases in PHRASES.items() for phrase in phrases
                      for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", text, re.I)]
        spans = []
        for candidate in sorted(candidates, key=lambda c: (-(c["end"] - c["start"]), c["start"])):
            if not any(candidate["start"] < s["end"] and s["start"] < candidate["end"] for s in spans):
                spans.append(candidate)
        spans.sort(key=lambda s: s["start"])
    needs = list(dict.fromkeys(ent["value"] for ent in spans))
    if "FAMILY_HEALTH" in needs and re.search(r"\bindividual\b", text, re.I) and not re.search(r"\bfamily\b", text, re.I):
        needs[needs.index("FAMILY_HEALTH")] = "INDIVIDUAL_HEALTH"
    age_match = re.search(r"\b(?:age(?:d)?(?: is)?\s*|i am\s+|i'm\s+)(\d{1,3})(?!\d|[,.]\d)\b|\b(\d{1,3})\s*(?:years? old|year-old)\b", text, re.I)
    age = int(next(g for g in age_match.groups() if g is not None)) if age_match else None
    if age is not None and not 0 <= age <= 120:
        age = None
    cue_matches = list(CUES.finditer(text))
    amounts = []
    for match in AMOUNT.finditer(text):
        if age_match and age_match.start() <= match.start() < age_match.end():
            continue
        value = numeric_value(match)
        if not 0 < value <= 10 ** 12:
            continue
        # Do not attach a later amount to an earlier cue across another amount.
        previous_number = [m for m in AMOUNT.finditer(text[:match.start()])]
        left = previous_number[-1].end() if previous_number else 0
        next_number = AMOUNT.search(text, match.end())
        right = next_number.start() if next_number else len(text)
        nearby = [c for c in cue_matches if left <= c.start() < right]
        closest = min(nearby, key=lambda c: min(abs(c.end() - match.start()), abs(c.start() - match.end()))) if nearby else None
        kind = closest.lastgroup if closest and min(abs(closest.end() - match.start()), abs(closest.start() - match.end())) <= 35 else "unknown"
        if kind == "unknown" and not match.group("unit") and not re.search(r"\b(?:inr|rs)\.?\s*$|\u20b9\s*$", text[:match.start()], re.I):
            continue
        local = text[left:right] if kind == "premium_budget" else ""
        period = "monthly" if re.search(r"\b(?:monthly|per month|a month)\b", local, re.I) else \
                 "annual" if re.search(r"\b(?:annual|annually|yearly|per year|a year)\b", local, re.I) else "unknown"
        currency = "INR" if re.search(r"\b(?:inr|rs)\.?\s*$|\u20b9\s*$", text[max(0, match.start()-8):match.start()], re.I) else "unspecified"
        amounts.append({"value": value, "kind": kind, "period": period, "currency": currency,
                        "text": match.group(), "start": match.start(), "end": match.end()})
    return {"needs": needs, "age": age, "amounts": amounts,
            "entities": spans, "extractor_error": error,
            "extractor": "spaCy EntityRuler + regex (no trained NER)" if pipeline else "Regex rules (spaCy unavailable)"}
