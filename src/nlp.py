"""Phase 3: intent / emotion / objection / stage classifiers + regex need/entity extractor.
Pure stdlib+numpy multinomial Naive Bayes (sklearn/scipy DLLs blocked on this host).
Same API: train_all() -> metrics dict, predict(text) -> dict."""
import os, json, re, math
from collections import Counter, defaultdict
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {}

NEED_KEYS = {"health": "FAMILY_HEALTH", "life": "LIFE_PROTECTION", "child": "CHILD_PROTECTION",
    "retire": "RETIREMENT", "income": "INCOME_PROTECTION", "critical": "CRITICAL_ILLNESS",
    "accident": "ACCIDENT_PROTECTION"}
OBJ_KEYS = {"expensive": "PRICE_TOO_HIGH", "too high": "PRICE_TOO_HIGH", "already have": "ALREADY_INSURED",
    "don't need": "NO_NEED", "don't trust": "DO_NOT_TRUST_INSURER", "compare": "COMPARING_OPTIONS",
    "past": "BAD_PAST_EXPERIENCE", "exclu": "COVERAGE_NOT_CLEAR", "family approval": "NEED_FAMILY_APPROVAL",
    "discuss with my family": "NEED_FAMILY_APPROVAL",
    "not interested": "NOT_READY", "stop": "NOT_READY"}

def tok(t):
    return re.findall(r"[a-z']+", t.lower())

class NB:
    """Multinomial NB with Laplace smoothing (unigrams+bigrams)."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        self.classes = sorted(set(y))
        self.ci = {c: i for i, c in enumerate(self.classes)}
        feats, counts, docs = [], Counter(), Counter()
        for t, c in zip(X, y):
            w = tok(t) + [a + " " + b for a, b in zip(tok(t), tok(t)[1:])]
            feats.append((w, c)); docs[c] += 1
            counts.update(set(w))
        self.vocab = {w: i for i, w in enumerate(counts)}
        V, C = len(self.vocab), len(self.classes)
        self.logprob = [[0.0] * V for _ in range(C)]
        self.logprior = [math.log(docs[c] / len(y)) for c in self.classes]
        tokcount = [[self.alpha] * V for _ in range(C)]
        tot = [self.alpha * V] * C
        for w, c in feats:
            i = self.ci[c]
            for f in w:
                if f in self.vocab:
                    tokcount[i][self.vocab[f]] += 1
                    tot[i] += 1
        for i in range(C):
            for j in range(V):
                self.logprob[i][j] = math.log(tokcount[i][j] / tot[i])
        return self
    def _vec(self, t):
        w = tok(t) + [a + " " + b for a, b in zip(tok(t), tok(t)[1:])]
        return [f for f in w if f in self.vocab]
    def predict(self, X):
        return [self.classes[self._scores(t).index(max(self._scores(t)))] for t in X]
    def _scores(self, t):
        f = self._vec(t)
        return [self.logprior[i] + sum(self.logprob[i][self.vocab[w]] for w in f)
                for i in range(len(self.classes))]
    def predict_proba(self, X):
        out = []
        for t in X:
            s = self._scores(t)
            m = max(s); e = [math.exp(x - m) for x in s]; z = sum(e)
            out.append([x / z for x in e])
        return out

def load_rows(split="train"):
    p = os.path.join(ROOT, "data", "processed", "insurance_sales", f"{split}.jsonl")
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if r["speaker"] == "customer"]

def _scores(y_true, y_pred, classes):
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
    f1s = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if b == c != a)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c != b)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    macro = sum(f1s) / max(len(f1s), 1)
    support = Counter(y_true)
    weighted = sum(f * support[c] for c, f in zip(classes, f1s)) / max(len(y_true), 1)
    return {"accuracy": round(acc, 3), "macro_f1": round(macro, 3),
            "weighted_f1": round(weighted, 3), "test_examples": len(y_true)}

def train_all():
    global MODELS
    tr, te = load_rows("train"), load_rows("test")
    tasks = {"intent": "customer_intent", "emotion": "customer_emotion",
             "objection": "objection_type", "stage": "sales_stage"}
    metrics = {}
    for name, col in tasks.items():
        m = NB().fit([r["text"] for r in tr], [r[col] for r in tr])
        MODELS[name] = m
        truth, predicted = [r[col] for r in te], m.predict([r["text"] for r in te])
        metrics[name] = _scores(truth, predicted, m.classes)
        metrics[name]["task"] = col
        metrics[name]["dataset"] = "insurance_sales_dialogue_dataset v2 (synthetic)"
        os.makedirs(os.path.join(ROOT, "results", "metrics"), exist_ok=True)
        matrix = {a: {b: sum(x == a and y == b for x, y in zip(truth, predicted))
                      for b in m.classes} for a in m.classes}
        with open(os.path.join(ROOT, "results", "metrics", f"{name}_confusion.json"), "w") as f:
            json.dump(matrix, f, indent=2)
    os.makedirs(os.path.join(ROOT, "results", "metrics"), exist_ok=True)
    json.dump(metrics, open(os.path.join(ROOT, "results", "metrics", "nlp_metrics.json"), "w"), indent=2)
    import pickle
    pickle.dump({k: {"classes": m.classes, "vocab": m.vocab, "logprob": m.logprob, "logprior": m.logprior}
                 for k, m in MODELS.items()},
                open(os.path.join(ROOT, "results", "nlp_models.pkl"), "wb"))
    return metrics

def _load():
    global MODELS
    if MODELS: return
    import pickle
    p = os.path.join(ROOT, "results", "nlp_models.pkl")
    if os.path.exists(p):
        for k, d in pickle.load(open(p, "rb")).items():
            m = NB(); m.classes = d["classes"]; m.vocab = d["vocab"]
            m.logprob = d["logprob"]; m.logprior = d["logprior"]
            m.ci = {c: i for i, c in enumerate(m.classes)}
            MODELS[k] = m

def predict(text, ctx="", models=None):
    if models is None:
        _load()
        models = MODELS
    def _p(name, fallback):
        try:
            m = models[name]
            pr = m.predict_proba([text])[0]
            i = int(max(range(len(pr)), key=lambda j: pr[j]))
            return m.classes[i], float(pr[i]), dict(zip(m.classes, pr))
        except Exception:
            return fallback, 0.5, {}
    intent, ic, _ = _p("intent", "OBJECTION")
    emotion, ec, eprobs = _p("emotion", "NEUTRAL")
    obj, oc, _ = _p("objection", "NONE")
    stage, sc, _ = _p("stage", "DISCOVERY")
    if stop_requested(text):
        intent, ic, obj, oc, stage = "REJECTION", 1.0, "NOT_READY", 1.0, "REJECTION"
    return {"intent": intent, "intent_conf": ic, "emotion": emotion, "emotion_conf": ec,
            "emotion_probs": eprobs, "objection": obj, "objection_conf": oc, "stage": stage}

def stop_requested(text):
    return bool(re.search(r"\b(?:please stop|stop contacting|stop calling|not interested|leave me alone|don't call|do not call|end (?:this|the) conversation)\b", text, re.I)
                or text.strip().lower() in ("stop", "no thanks", "no thank you"))


def extract_needs(text):
    t = text.lower()
    needs = [v for k, v in NEED_KEYS.items() if k in t]
    objs = [v for k, v in OBJ_KEYS.items() if k in t]
    budget = next((b for b in ("low", "mid", "high") if f"{b} budget" in t), "unknown")
    if budget == "unknown" and any(w in t for w in ["cheap", "expensive", "can't afford", "too high"]):
        budget = "low"
    if "individual" in t and "health" in t:
        needs = ["INDIVIDUAL_HEALTH"]
    if "family" in t and ("health" in t or "hospital" in t):
        needs = ["FAMILY_HEALTH"]
    if "retirement" in t:
        needs = ["RETIREMENT"]
    amount = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|l)\b", t)
    existing = "unknown"
    if "already have" in t or "current policy" in t or "existing policy" in t:
        existing = f"partial-{amount.group(1)}L" if amount else "insured-unspecified"
    age_match = re.search(r"(?:age(?:d)?\s*|i am\s+|i'm\s+)(\d{1,2})\b", t)
    return {"needs": list(dict.fromkeys(needs)), "objection_hint": objs[0] if objs else "NONE",
            "budget": budget, "existing_coverage": existing,
            "age": int(age_match.group(1)) if age_match else None,
            "stop_requested": stop_requested(text)}


def live_cues(text):
    """Explicit live-dialogue cues; these do not rewrite the trained NLP classifier."""
    t = text.lower().strip()
    payment = bool(re.search(r"\b(?:payment|premium|cost|price|pay)\b", t)
                   and re.search(r"\b(?:what|how much|tell me|calculate)\b", t))
    product_request = bool(re.search(r"\b(?:i need|we need|looking for|i want)\b", t)
                           and re.search(r"\b(?:insurance|policy|cover|health|life|critical illness|retirement|income protection|child protection|accident protection)\b", t))
    ready = bool(re.search(r"\b(?:(?:i am|i'm|we are|we're) ready to (?:buy|proceed|apply)"
                           r"|(?:i|we) (?:would )?(?:want|like) to (?:buy|proceed|apply)|let'?s proceed)\b", t))
    if re.search(r"\b(?:not|never|don't|do not|cannot|can't|unsure|maybe|if)\b", t):
        ready = False
    acknowledged = bool(re.search(r"\b(?:that makes sense|i understand the terms|that answers my question)\b", t))
    one_at_a_time = bool(re.search(r"\b(?:one by one|1 by 1|one (?:question )?at a time|step by step|ask me questions)\b", t))
    cooperative = one_at_a_time or bool(re.search(
        r"\b(?:happy|ready|willing) to (?:share|provide|answer)\b|\b(?:yes|sure|okay|ok)[,.]? (?:please |go ahead|ask|tell me|continue)", t))
    if re.search(r"\b(?:not (?:happy|ready|willing) to|(?:don't|do not) (?:ask|want to share))\b", t):
        cooperative = one_at_a_time = False
    family_approval = bool(re.search(r"\b(?:family approval|(?:ask|discuss|check) (?:with )?my (?:family|wife|husband|partner))\b", t))
    bare_amount = re.fullmatch(
        r"(?:around|about|roughly|approximately|approx\.?)?\s*(?:rs\.?|inr|\u20b9)?\s*"
        r"(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|l|crores?|k|thousand)?", t)
    amount = None
    if bare_amount:
        value = Decimal(bare_amount.group(1).replace(",", ""))
        unit = bare_amount.group(2) or ""
        multiplier = 100000 if unit in ("l", "lakh", "lakhs") else 10000000 if unit.startswith("crore") else 1000 if unit in ("k", "thousand") else 1
        value *= multiplier
        if 0 < value <= 10 ** 12:
            amount = int(value) if value == value.to_integral_value() else float(value)
    coverage = bool(re.search(r"\b(?:coverage|cover amount|sum insured)\b", t))
    premium_budget = bool(re.search(r"\b(?:premium|budget|payment)\b", t))
    kind = "coverage" if coverage and not premium_budget else "premium_budget" if premium_budget and not coverage else "unknown"
    period = "monthly" if re.search(r"\b(?:monthly|per month|a month)\b", t) else "annual" if re.search(r"\b(?:annual|annually|yearly|per year|a year)\b", t) else "unknown"
    return {"payment_question": payment, "product_request": product_request,
            "cooperative": cooperative, "one_at_a_time": one_at_a_time,
            "ready_to_proceed": ready, "acknowledged": acknowledged,
            "family_approval": family_approval, "amount": amount, "amount_kind": kind, "period": period}
