"""Text conversation entry point using the trained policy and verified generator."""
import os
import numpy as np

from . import nlp, ppo_numpy
from .agents import rule_policy
from .common import ACTIONS, PROFILES
from .generation import ResponseGenerator
from .dialogue_flow import DiscoveryFlow
from .sim_env import action_mask
from .state_products import Memory, build_state, encode, STATE_DIM, recommend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConversationSession:
    def __init__(self, policy="ppo", generator="template", profile="HESITANT", age=None, budget="unknown",
                 checkpoint=None, predictor=None, nlp_backend=None):
        if policy not in ("ppo", "rule") or profile not in PROFILES:
            raise ValueError("Invalid policy or customer profile")
        self.policy_name = policy
        self.model = None
        if policy == "ppo":
            path = os.fspath(checkpoint) if checkpoint else os.path.join(ROOT, "results", "checkpoints", "ppo.npz")
            if not os.path.isfile(path):
                raise RuntimeError("Train the policy with python run_all.py before starting a PPO conversation")
            self.model = ppo_numpy.load(path, STATE_DIM, len(ACTIONS))
        self.rule = rule_policy()
        from .understanding import get_predictor, selected_backend
        self.nlp_backend = nlp_backend or selected_backend()
        if self.nlp_backend not in ("nb", "transformer"):
            raise ValueError("Invalid NLP backend")
        self.predictor = predictor or (get_predictor() if self.nlp_backend == "transformer" else lambda text: nlp.predict(text, backend="nb"))
        self.generator = ResponseGenerator(generator)
        self.memory = Memory()
        self.responses = []
        self.closed = False
        self.need, self.budget, self.existing, self.age = None, budget, "unknown", age
        self.profile, self.previous_action, self.turn = profile, "DISCOVER_NEEDS", 0
        self.amount_context = {"value": None, "kind": "unknown", "period": "unknown"}
        self.premium_budget = None
        self.discovery = DiscoveryFlow()

    def discovery_context(self):
        return {"need": self.need, "age": self.age, "existing": self.existing, "budget": self.budget}

    def prepare_turn(self, text, acoustic=None):
        """Build the same observable state for live inference and voice-policy training."""
        if self.closed:
            raise RuntimeError("This conversation is closed; start a new session")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Customer text must not be empty")
        raw_understanding = self.predictor(text)
        understanding = dict(raw_understanding)
        entities = nlp.extract_needs(text)
        cues = nlp.live_cues(text)
        previous_need = self.need
        sources = {key: "nlp_model" for key in ("intent", "emotion", "objection")}
        if not entities["stop_requested"]:
            explicit_intent = "PRICE_INQUIRY" if cues["payment_question"] else "PURCHASE_INTEREST" if cues["ready_to_proceed"] else "PRODUCT_INQUIRY" if cues["product_request"] or cues["cooperative"] else None
            if explicit_intent:
                understanding.update(intent=explicit_intent, intent_conf=None)
                sources["intent"] = "explicit_cue"
            if cues["family_approval"]:
                understanding.update(objection="NEED_FAMILY_APPROVAL", objection_conf=None)
                sources["objection"] = "explicit_cue"
            elif entities["objection_hint"] != "NONE":
                understanding.update(objection=entities["objection_hint"], objection_conf=None)
                sources["objection"] = "explicit_cue"
            elif cues["payment_question"] or cues["product_request"] or cues["ready_to_proceed"] or cues["acknowledged"] or cues["cooperative"]:
                understanding.update(objection="NONE", objection_conf=None)
                sources["objection"] = "explicit_cue"
        understanding["sources"] = sources
        understanding["objections"] = list(raw_understanding.get("objections", []))
        if understanding["objection"] != "NONE" and understanding["objection"] not in understanding["objections"]:
            understanding["objections"].insert(0, understanding["objection"])
        if entities["needs"]:
            self.need = entities["needs"][0]
        for name in ("budget", "existing_coverage"):
            if entities[name] != "unknown":
                setattr(self, "existing" if name == "existing_coverage" else name, entities[name])
        if entities["age"] is not None:
            self.age = entities["age"]
        stop = entities["stop_requested"]
        if not stop:
            for kind in ("coverage", "premium_budget"):
                exact = [a for a in entities["amounts"] if a["kind"] == kind]
                if len(exact) == 1 and entities["existing_coverage"] == "unknown":
                    self.amount_context = {key: exact[0][key] for key in ("value", "kind", "period")}
                    if kind == "coverage":
                        self.discovery.coverage = exact[0]["value"]
                    else:
                        self.premium_budget = dict(exact[0])
        answered = False
        if not stop:
            context, answered = self.discovery.observe(text, cues, self.discovery_context())
            self.need, self.age, self.existing, self.budget = (context[key] for key in ("need", "age", "existing", "budget"))
            if answered:
                understanding.update(intent="PRODUCT_INQUIRY", intent_conf=None, objection="NONE", objection_conf=None)
                sources.update(intent="guided_answer", objection="guided_answer")
        task, decision_reason = None, None
        if not stop:
            if self.discovery.active and self.discovery.coverage is not None:
                self.amount_context = {"value": self.discovery.coverage, "kind": "coverage", "period": "unknown"}
            if cues["amount"] is not None and not answered:
                self.amount_context = {"value": cues["amount"], "kind": "unknown", "period": "unknown"}
                task, decision_reason = "clarify_amount", "ambiguous_amount"
            elif self.amount_context["value"] is not None and cues["amount_kind"] != "unknown" and not cues["payment_question"]:
                self.amount_context.update(kind=cues["amount_kind"], period=cues["period"])
            if cues["payment_question"]:
                task, decision_reason = "explain_payment", "no_verified_premium_quote"
            elif self.amount_context["value"] is not None and self.amount_context["kind"] == "unknown":
                task, decision_reason = "clarify_amount", "ambiguous_amount"
            elif self.need and previous_need is None and cues["product_request"] and understanding["objection"] == "NONE":
                task, decision_reason = "initial_discovery", "initial_discovery_mask"
            if not self.need and not cues["family_approval"]:
                task, decision_reason = "clarify_need", "unconfirmed_insurance_need"
            if self.discovery.active and not cues["payment_question"] and not cues["family_approval"] and entities["objection_hint"] in ("NONE", "ALREADY_INSURED"):
                guided_task = self.discovery.task(self.discovery_context())
                if guided_task:
                    task, decision_reason = guided_task, "guided_discovery"
        stage = "REJECTION" if stop else "DISCOVERY" if not self.need and self.turn else "GREETING" if self.turn == 0 else "OBJECTION_HANDLING" if understanding["objection"] != "NONE" else "COMMITMENT" if cues["ready_to_proceed"] else "EVALUATION"
        if not stop and (task == "clarify_need" or task and task.startswith("discovery_")):
            stage = "DISCOVERY"
        state = build_state(understanding, {"needs": [self.need] if self.need else []}, self.profile, stage,
                            prev_action=self.previous_action, turn=self.turn, budget=self.budget, existing=self.existing)
        state.update(age=self.age, conversation_summary=self.memory.summary(),
                     amount_context=dict(self.amount_context), dialogue_task=task,
                     coverage_target=self.discovery.coverage,
                     workflow=self.discovery.snapshot(self.discovery_context(), task),
                     state_source="NLP estimates; trust and purchase intent are unobserved defaults")
        state["extracted_entities"] = entities
        state["premium_budget"] = self.premium_budget
        state["objections"] = understanding["objections"]
        # Audio can affect wording style only. It cannot supply consent, policy state or purchase readiness.
        state["acoustic_emotion"] = acoustic
        state["communication_style"] = "patient, brief and non-pressuring" if acoustic and acoustic.get("status") == "estimated" and acoustic.get("label") in ("ANGRY", "SAD") else "neutral and non-pressuring"
        available = self.age is not None and bool(recommend(state["need"], self.budget, self.age, self.discovery.coverage))
        mask = action_mask(stage, self.turn, bool(self.need), stop_requested=stop, available_products=available)
        # Commitment requires an observable, affirmative buyer signal, not a hidden intent score.
        if stage != "COMMITMENT" or task is not None:
            mask[ACTIONS.index("ASK_FOR_COMMITMENT")] = False
        if task == "initial_discovery":
            mask = [allowed and action in ("ASK_CLARIFYING_QUESTION", "DISCOVER_COVERAGE_GAP")
                    for allowed, action in zip(mask, ACTIONS)]
        if task == "clarify_need":
            mask = [action in ("DISCOVER_NEEDS", "ASK_CLARIFYING_QUESTION") for action in ACTIONS]
        elif task and task.startswith("discovery_"):
            mask = [action == "ASK_CLARIFYING_QUESTION" for action in ACTIONS]
        elif task in ("guided_review", "handoff_summary"):
            mask = [action == ("EXPLAIN_PRODUCT" if available else "ASK_CLARIFYING_QUESTION") for action in ACTIONS]
        info = {"state": state, "sales_stage": stage, "objection": understanding["objection"],
                "need_known": bool(self.need), "action_mask": mask, "customer_text": text}
        observation = np.asarray(encode(state), dtype=np.float32)
        return {"observation": observation, "info": info, "text": text, "state": state,
                "mask": mask, "stop": stop, "task": task, "available": available,
                "decision_reason": decision_reason, "understanding": understanding,
                "raw_understanding": raw_understanding, "turn": self.turn, "owner": self}

    def complete_turn(self, prepared, action_id=None):
        if self.closed or prepared["turn"] != self.turn or prepared["owner"] is not self:
            raise RuntimeError("Prepared turn is closed, stale, or belongs to another session")
        state, text, mask = prepared["state"], prepared["text"], prepared["mask"]
        task, stop, available = prepared["task"], prepared["stop"], prepared["available"]
        understanding, raw_understanding = prepared["understanding"], prepared["raw_understanding"]
        decision_reason = prepared["decision_reason"]
        observation, info = prepared["observation"], prepared["info"]
        if action_id is None:
            action_id = self.model.act(observation, mask, deterministic=True)[0] if self.model else self.rule(observation, mask, info)
            if not mask[action_id]:
                action_id = next(i for i, allowed in enumerate(mask) if allowed)
        elif not isinstance(action_id, (int, np.integer)) or not 0 <= action_id < len(ACTIONS) or not mask[action_id]:
            raise ValueError("External action is not allowed by this turn's mask")
        action = ACTIONS[action_id]
        policy_action, decision_source = action, self.policy_name
        if decision_reason == "guided_discovery":
            decision_source = "guided_workflow"
        if not stop and task in ("clarify_amount", "explain_payment"):
            action = "AFFORDABILITY_FRAMING" if task == "explain_payment" and available else "ASK_CLARIFYING_QUESTION"
            decision_source = "conversation_guard"
        dialogue = [{"customer": turn["customer"], "agent": turn["agent"]} for turn in self.memory.h]
        result = self.generator.generate(action, state, self.responses, text, dialogue=dialogue)
        self.memory.add({"customer": text, "agent": result["text"], "reaction": understanding["emotion"]})
        self.responses.append(result["text"])
        self.previous_action, self.turn = action, self.turn + 1
        self.closed = stop and action == "RESPECT_REJECTION"
        self.discovery.replied(task, self.discovery_context())
        state["workflow"] = self.discovery.snapshot(self.discovery_context(), task)
        return {**result, "understanding": understanding, "raw_understanding": raw_understanding,
                "state": state, "closed": self.closed, "policy_action": policy_action,
                "closed_reason": "buyer_stop" if self.closed else None,
                "decision_source": decision_source, "decision_reason": decision_reason}

    def reply(self, text, acoustic=None):
        return self.complete_turn(self.prepare_turn(text, acoustic))
