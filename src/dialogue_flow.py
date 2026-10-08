"""Explicit, optional one-question intake around the learned conversational policy."""
import re

from .nlp import live_cues

FIELDS = ("need", "age", "coverage", "existing", "budget")
QUESTIONS = {
    "need": "What type of insurance are you looking for?",
    "age": "What is your age?",
    "coverage": "What total insurance coverage amount would you like?",
    "existing": "Do you already have an insurance policy?",
    "budget": "Which synthetic premium band should I use: low, medium or high?",
}


class DiscoveryFlow:
    def __init__(self):
        self.active = False
        self.pending = None
        self.coverage = None
        self.coverage_unspecified = False
        self.reviewed = None
        self.summary_requested = False

    def observe(self, text, cues, context):
        context = dict(context)
        t = text.lower().strip().rstrip(".!?")
        if cues["cooperative"]:
            self.active = True
        if not self.active:
            return context, False
        answered = False
        if self.pending == "age" and re.fullmatch(r"\d{1,3}(?: years? old)?", t):
            age = int(t.split()[0])
            if 18 <= age <= 100:
                context["age"], answered = age, True
        if self.pending == "coverage":
            amount = cues["amount"]
            if amount is None:
                match = re.fullmatch(r"(?:i (?:want|need)(?: about)?\s+)?(.+?)\s+(?:of )?(?:coverage|cover)", t)
                if match:
                    amount = live_cues(match.group(1))["amount"]
            if amount is not None and cues["amount_kind"] != "premium_budget" and not cues["payment_question"]:
                self.coverage, self.coverage_unspecified, answered = amount, False, True
            elif t in ("not sure", "i am not sure", "unsure", "i don't know", "i do not know", "don't know", "do not know", "skip"):
                self.coverage_unspecified, answered = True, True
        if self.pending == "existing":
            if re.fullmatch(r"(?:no(?: thanks)?|none|not yet|i (?:don't|do not) have (?:any |an? )?(?:insurance|policy|cover)(?: policy)?)", t):
                context["existing"], answered = "none", True
            elif re.fullmatch(r"(?:yes|yes i do|i have (?:an? )?(?:insurance|policy)(?: policy)?)", t):
                context["existing"], answered = "insured-unspecified", True
        if self.pending == "budget":
            match = re.fullmatch(r"(?:a )?(low|medium|mid|high)(?: budget| band| premium band)?", t)
            if match:
                context["budget"], answered = {"medium": "mid"}.get(match.group(1), match.group(1)), True
        if cues["ready_to_proceed"] or (self.pending == "review" and t in ("yes", "yes please", "sure", "please", "send the summary")):
            self.summary_requested = True
        return context, answered

    def missing(self, context):
        return [field for field in FIELDS if
                (field == "coverage" and self.coverage is None and not self.coverage_unspecified) or
                (field != "coverage" and context.get(field) in (None, "unknown"))]

    def signature(self, context):
        return tuple(context.get(field) if field != "coverage" else (self.coverage, self.coverage_unspecified)
                     for field in FIELDS)

    def task(self, context):
        missing = self.missing(context)
        if missing:
            return "discovery_" + missing[0]
        if self.summary_requested:
            return "handoff_summary"
        if self.reviewed != self.signature(context):
            return "guided_review"
        return None

    def replied(self, task, context):
        if task and task.startswith("discovery_"):
            self.pending = task.removeprefix("discovery_")
        elif task in ("guided_review", "handoff_summary"):
            self.reviewed = self.signature(context)
            self.pending = "review" if task == "guided_review" else None
            if task == "handoff_summary":
                self.summary_requested = False

    def snapshot(self, context, task=None):
        missing = self.missing(context)
        return {"active": self.active, "pending_field": task.removeprefix("discovery_") if task and task.startswith("discovery_") else self.pending,
                "completed_fields": [field for field in FIELDS if field not in missing],
                "missing_fields": missing, "coverage_target": self.coverage,
                "phase": "collecting_details" if missing else "summary_ready" if task == "handoff_summary" else "ready_for_review",
                "policy_issued": False}
