"""Phase 7-8: customer simulator + multi-objective reward + Gymnasium env (spec §24-28)."""
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .common import ACTIONS, ACTION_TO_STRATEGY, PROFILES, NEEDS, load_config
from .state_products import build_state, encode, STATE_DIM, recommend, ablate_observation
from .generation import ResponseGenerator, PRODUCT_ACTIONS

# profile -> preferred actions (spec §25); everything else is neutral/weak
PREF = {"PRICE_SENSITIVE": ["AFFORDABILITY_FRAMING", "OFFER_ALTERNATIVE", "VALUE_FRAMING"],
    "SKEPTICAL": ["BUILD_TRUST", "COMPARE_OPTIONS", "EXPLAIN_PRODUCT"],
    "READY_TO_BUY": ["ASK_FOR_COMMITMENT", "EXPLAIN_PRODUCT", "PERSONALIZE"],
    "HESITANT": ["ASK_CLARIFYING_QUESTION", "DISCOVER_NEEDS", "BUILD_TRUST", "FOLLOW_UP"],
    "FAMILY_ORIENTED": ["PERSONALIZE", "RISK_EXPLANATION", "VALUE_FRAMING"],
    "RISK_AVERSE": ["RISK_EXPLANATION", "BUILD_TRUST", "EXPLAIN_PRODUCT"],
    "INFORMATION_SEEKING": ["EXPLAIN_PRODUCT", "COMPARE_OPTIONS", "ASK_CLARIFYING_QUESTION"],
    "LOYAL_TO_EXISTING_INSURER": ["DISCOVER_COVERAGE_GAP", "COMPARE_OPTIONS", "YES_YES_FRAMING"]}
BAD_EARLY_CLOSE = ["ASK_FOR_COMMITMENT", "COMPARE_OPTIONS", "OFFER_ALTERNATIVE"]

class CustomerSim:
    def __init__(self, seed=0, observable_dialogue=False):
        self.rng = random.Random(seed)
        self.observable_dialogue = observable_dialogue
    def reset(self, profile=None):
        profile = profile or {}
        self.profile = self.rng.choice(PROFILES)
        self.need = self.rng.choice(NEEDS)
        self.profile = profile.get("personality", self.profile)
        self.need = profile.get("primary_need", self.need)
        self.age = profile.get("age", self.rng.randint(18, 75))
        self.budget = profile.get("budget", self.rng.choice(["low", "mid", "high"]))
        self.existing = profile.get("existing_coverage", self.rng.choice(["none", "partial-5L", "full-10L"]))
        self.s = {"trust": round(self.rng.uniform(0.3, 0.6), 2),
                  "satisfaction": round(self.rng.uniform(0.3, 0.6), 2),
                  "engagement": round(self.rng.uniform(0.3, 0.6), 2),
                  "purchase_intent": round(self.rng.uniform(0.2, 0.4), 2),
                  "patience": self.rng.randint(4, 8),
                  "objection": self.rng.choice(["NONE", "PRICE_TOO_HIGH", "NEED_MORE_INFORMATION", "ALREADY_INSURED"]),
                  "stage": "GREETING", "turn": 0, "prev_action": None, "emotion": "NEUTRAL",
                  "need_known": False, "stop_requested": False}
        for field in ("trust", "satisfaction", "engagement", "purchase_intent", "patience", "objection"):
            if field in profile:
                self.s[field] = profile[field]
        return self.s
    def step(self, action, suitable=True):
        s = self.s
        s["turn"] += 1
        good = action in PREF.get(self.profile, [])
        # reaction
        if action == "RESPECT_REJECTION":
            reaction = "positive"
        elif action in PRODUCT_ACTIONS and not suitable:
            reaction = "negative"
        elif s["objection"] != "NONE" and action in ("HANDLE_OBJECTION", "YES_YES_FRAMING", "BUILD_TRUST", "AFFORDABILITY_FRAMING", "DISCOVER_COVERAGE_GAP"):
            reaction = "positive" if self.rng.random() < 0.7 else "neutral"
        elif good:
            reaction = "positive" if self.rng.random() < 0.65 else "neutral"
        elif action in BAD_EARLY_CLOSE and s["turn"] < 4:
            reaction = "negative"
        else:
            reaction = self.rng.choices(["positive", "neutral", "negative"], [0.25, 0.5, 0.25])[0]
        d = 0.07 if reaction == "positive" else (-0.07 if reaction == "negative" else 0.0)
        s["trust"] = float(np.clip(s["trust"] + d + (0.02 if good else 0), 0, 1))
        s["satisfaction"] = float(np.clip(s["satisfaction"] + d, 0, 1))
        s["engagement"] = float(np.clip(s["engagement"] + d + 0.01, 0, 1))
        s["purchase_intent"] = float(np.clip(s["purchase_intent"] + d * 1.2 + (0.03 if good else 0), 0, 1))
        resolving = action in ("HANDLE_OBJECTION", "YES_YES_FRAMING", "BUILD_TRUST", "AFFORDABILITY_FRAMING",
                               "DISCOVER_COVERAGE_GAP", "EXPLAIN_PRODUCT", "COMPARE_OPTIONS")
        if resolving and reaction == "positive" and s["objection"] != "NONE" and self.rng.random() < 0.5:
            s["objection"] = "NONE"
        if action in ("DISCOVER_NEEDS", "ASK_CLARIFYING_QUESTION", "DISCOVER_COVERAGE_GAP"):
            s["need_known"] = True
        s["emotion"] = "CONCERNED" if reaction == "negative" else "SATISFIED" if reaction == "positive" else "NEUTRAL"
        if action == "RESPECT_REJECTION":
            s["stage"] = "REJECTION"
        elif s["satisfaction"] < 0.15 or s["turn"] > s["patience"] + 4:
            s["stage"], s["stop_requested"] = "REJECTION", True
        elif s["objection"] != "NONE":
            s["stage"] = "OBJECTION_HANDLING"
        elif not s["need_known"]:
            s["stage"] = "DISCOVERY"
        elif s["purchase_intent"] >= 0.7:
            s["stage"] = "COMMITMENT"
        elif s["turn"] >= 4:
            s["stage"] = "EVALUATION"
        else:
            s["stage"] = "PRODUCT_EDUCATION"
        s["prev_action"] = action
        done = action == "RESPECT_REJECTION"
        reply = {"positive": "That makes sense, tell me more.", "neutral": "Hmm, I need to think about it.",
                 "negative": "No, that's not what I want."}[reaction]
        if s["stop_requested"]:
            reply = "Please stop. I am not interested."
        elif action in ("DISCOVER_NEEDS", "ASK_CLARIFYING_QUESTION"):
            reply = f"I need {self.need.lower().replace('_', ' ')} with a {self.budget} budget."
        elif self.observable_dialogue:
            if s["objection"] != "NONE":
                reply = {"PRICE_TOO_HIGH": "That is too expensive for my budget.",
                         "NEED_MORE_INFORMATION": "I need more information about the exclusions.",
                         "ALREADY_INSURED": "I already have an insurance policy."}.get(s["objection"], reply)
            elif s["purchase_intent"] > 0.8 and suitable:
                reply = "I understand the terms and I am ready to proceed."
            elif s["purchase_intent"] >= 0.7 and suitable:
                reply = "That makes sense. How would the next step work?"
        return reply, reaction, done

def compute_reward(prev, new, action, info, w):
    r = 0.0
    if info.get("need_discovered"): r += w["need_discovery"]
    if info.get("useful_clarification"): r += w["useful_clarification"]
    if info.get("objection_resolved"): r += w["objection_resolution"]
    if new["engagement"] > prev["engagement"]: r += w["engagement_improved"]
    if new["trust"] > prev["trust"]: r += w["trust_improved"]
    if info.get("suitable"): r += w["suitable_product"]
    if new["purchase_intent"] > 0.6 and prev.get("purchase_intent", 0) <= 0.6: r += w["qualified_lead"]
    if info.get("purchased") and info.get("suitable"): r += w["suitable_purchase"]
    if info.get("repeated"): r += w["repetition"]
    if info.get("premature"): r += w["pressure"] * 0.5 + w["premature_closing"]
    if info.get("pressure"): r += w["pressure"]
    if info.get("unsupported"): r += w["unsupported_claim"]
    if info.get("unsuitable"): r += w["unsuitable_recommendation"]
    if info.get("reaction") == "negative": r += w["frustration"]
    return round(float(r), 3)

def action_mask(stage, turn, need_known, stop_requested=False, available_products=True):
    m = [True] * len(ACTIONS)
    if stage == "REJECTION" or stop_requested:
        return [a == "RESPECT_REJECTION" for a in ACTIONS]
    m[ACTIONS.index("RESPECT_REJECTION")] = False
    if stage == "GREETING" or turn < 3:
        m[ACTIONS.index("ASK_FOR_COMMITMENT")] = False
    if not need_known:
        for a in PRODUCT_ACTIONS:
            m[ACTIONS.index(a)] = False
    else:
        m[ACTIONS.index("DISCOVER_NEEDS")] = False
    if not available_products:
        for a in PRODUCT_ACTIONS:
            m[ACTIONS.index(a)] = False
    return m

class InsuranceEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, reward_weights=None, seed=42, max_turns=20, disabled_features=(),
                 banned_actions=(), generator=None, observable_dialogue=False):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)
        self.w = {**load_config()["reward"], **(reward_weights or {})}
        self.max_turns = max_turns
        self.disabled_features = disabled_features
        self.banned_actions = set(banned_actions)
        self.generator = generator or ResponseGenerator()
        self._done = True
        self.observable_dialogue = observable_dialogue
        self.sim = CustomerSim(seed, observable_dialogue)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.sim = CustomerSim(seed, self.observable_dialogue)
        cs = self.sim.reset((options or {}).get("profile"))
        self.t, self.prev_act, self._need_known = 0, None, False
        self._cs = cs
        self._history, self._done = [], False
        self.customer_text = "I'd like to discuss insurance options."
        self._state = self._build_state("neutral")
        return self._observe(), self._info()

    def _build_state(self, reaction):
        cs = self._cs
        s = build_state({"intent": "REJECTION" if cs["stop_requested"] else "OBJECTION" if cs["objection"] != "NONE" else "PRODUCT_INQUIRY",
                         "emotion": cs["emotion"], "objection": cs["objection"],
                         "objection_conf": 0.5 if cs["objection"] != "NONE" else 0},
                        {"needs": [self.sim.need] if self._need_known else []}, self.sim.profile, cs["stage"],
                        trust=cs["trust"], eng=cs["engagement"], sat=cs["satisfaction"],
                        purchase=cs["purchase_intent"], prev_action=self.prev_act or "DISCOVER_NEEDS",
                        prev_reaction=reaction, turn=self.t, budget=self.sim.budget,
                        existing=self.sim.existing, gap="yes" if self.sim.existing != "full-10L" else "no")
        s["age"] = self.sim.age
        return s

    def _mask(self):
        available = bool(recommend(self.sim.need, self.sim.budget, self.sim.age))
        mask = action_mask(self._cs["stage"], self.t, self._need_known, self._cs["stop_requested"], available)
        for i, a in enumerate(ACTIONS):
            if a in self.banned_actions and a != "RESPECT_REJECTION":
                mask[i] = False
        return mask

    def _observe(self):
        return ablate_observation(encode(self._state), self.disabled_features)

    def _info(self):
        return {"action": self.prev_act or "DISCOVER_NEEDS",
                "strategy": ACTION_TO_STRATEGY[self.prev_act or "DISCOVER_NEEDS"],
                "sales_stage": self._cs["stage"], "objection": self._cs["objection"],
                "need_known": self._need_known, "action_mask": self._mask(), "state": dict(self._state),
                "customer_text": self.customer_text, "stop_requested": self._cs["stop_requested"]}
    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before starting another episode")
        if not self.action_space.contains(action):
            raise ValueError("Action outside action_space")
        a = ACTIONS[int(action)]
        prev = dict(self._cs)
        mask = self._mask()
        valid = mask[int(action)]
        recs = recommend(self.sim.need, self.sim.budget, self.sim.age)
        suitable = bool(recs) and self._need_known
        generated = self.generator.generate(a, self._state, self._history, self.customer_text)
        need_discovered = not self._need_known and a in ("DISCOVER_NEEDS", "ASK_CLARIFYING_QUESTION", "DISCOVER_COVERAGE_GAP")
        if a in ("DISCOVER_NEEDS", "ASK_CLARIFYING_QUESTION", "DISCOVER_COVERAGE_GAP"):
            self._need_known = True
        reply, reaction, sim_done = self.sim.step(a, suitable)
        self.t += 1
        purchased = self._cs["purchase_intent"] > 0.8 and a == "ASK_FOR_COMMITMENT" and suitable and valid
        if purchased:
            self._cs["stage"] = "PURCHASE"
        info = {"action": a, "strategy": ACTION_TO_STRATEGY[a], "sales_stage": self._cs["stage"], "customer_reaction": reaction,
                "purchase_intent": self._cs["purchase_intent"], "trust": self._cs["trust"],
                "satisfaction": self._cs["satisfaction"], "reaction": reaction,
                "objection": self._cs["objection"], "need_known": self._need_known,
                "need_discovered": need_discovered, "useful_clarification": need_discovered and a == "ASK_CLARIFYING_QUESTION",
                "objection_resolved": prev["objection"] != "NONE" and self._cs["objection"] == "NONE",
                "suitable": suitable and a in ("COMPARE_OPTIONS", "OFFER_ALTERNATIVE", "ASK_FOR_COMMITMENT", "PERSONALIZE", "EXPLAIN_PRODUCT"),
                "purchased": purchased, "repeated": generated["validation"]["repeated"],
                "premature": (not valid) or (a == "ASK_FOR_COMMITMENT" and self.t < 3),
                "pressure": prev["stop_requested"] and a != "RESPECT_REJECTION",
                "unsupported": bool(generated["validation"]["unsupported"]),
                "unsuitable": a in PRODUCT_ACTIONS and not suitable,
                "reply": reply, "seller_response": generated["text"], "product_facts": generated["facts"],
                "generation_source": generated["source"], "llm_rejected": generated["llm_rejected"],
                "llm_error": generated["llm_error"]}
        info["safety_violation"] = any(info[k] for k in ("pressure", "premature", "unsupported", "unsuitable"))
        r = compute_reward(prev, self._cs, a, info, self.w)
        self.prev_act = a
        self.customer_text = reply
        self._history.append(generated["text"])
        self._state = self._build_state(reaction)
        info.update(self._info())
        terminated = bool(sim_done or purchased)
        truncated = bool(self.t >= self.max_turns and not terminated)
        self._done = terminated or truncated
        return self._observe(), r, terminated, truncated, info
