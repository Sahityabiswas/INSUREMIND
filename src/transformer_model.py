"""Shared MiniLM encoder with independent intent and multi-label objection heads."""
import torch
from torch import nn
from transformers import AutoModel

from .common import INTENTS, OBJECTIONS

OBJECTION_LABELS = OBJECTIONS[1:]


class InsuranceEncoder(nn.Module):
    def __init__(self, encoder_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True,
                                                trust_remote_code=False, attn_implementation="eager")
        size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.intent = nn.Linear(size, len(INTENTS))
        self.objections = nn.Linear(size, len(OBJECTION_LABELS))

    def forward(self, **tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = self.dropout(pooled)
        return self.intent(pooled), self.objections(pooled)
