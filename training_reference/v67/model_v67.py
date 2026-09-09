# Authorized reference implementation; not an end-to-end runnable training release.
# See training_reference/README.md and NOTICE.md.
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "v14"), str(ROOT / "v7"), str(ROOT / "v40")]

from features_v14 import ACTION_NUM_DIM, ENTITY_NUM_DIM, MAX_COUNT_CLASS, STATE_NUM_DIM
from features_v40 import HISTORY_DIM, RESOURCE_DIM, TACTIC_DIM
from model_torch import PolicyValueNet
from model_v14 import EncoderBlock, PreNormCrossAttention, gelu


PLAN_DIM = 8
PLAN_TYPE_CLASSES = 9
PLAN_CARD_BUCKETS = 257


class V67PlanTransformer(nn.Module):
    """V40 policy with training-only heads for the rest of the current turn."""

    def __init__(self, deck_count: int = 1, dim: int = 128):
        super().__init__()
        self.base = PolicyValueNet(decks=deck_count)
        self.dim = dim
        self.entity_delta = nn.Sequential(nn.Linear(ENTITY_NUM_DIM, 96), gelu(), nn.LayerNorm(96))
        self.entity_project = nn.Sequential(nn.Linear(96, dim), gelu(), nn.LayerNorm(dim))
        self.state_project = nn.Sequential(nn.Linear(192 + STATE_NUM_DIM + HISTORY_DIM + RESOURCE_DIM, dim), gelu(), nn.LayerNorm(dim))
        self.action_num = nn.Sequential(nn.Linear(ACTION_NUM_DIM, 64), gelu(), nn.LayerNorm(64))
        self.tactic_num = nn.Sequential(nn.Linear(TACTIC_DIM, 64), gelu(), nn.LayerNorm(64))
        self.action_project = nn.Sequential(nn.Linear(128 + 96 + 64 + 64 + 1, dim), gelu(), nn.LayerNorm(dim))
        self.state_blocks = nn.ModuleList([EncoderBlock(dim), EncoderBlock(dim)])
        self.cross = PreNormCrossAttention(dim)
        self.action_block = EncoderBlock(dim)
        self.delta = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), gelu(), nn.Linear(dim, 1))
        self.gate = nn.Sequential(nn.LayerNorm(dim + 1), nn.Linear(dim + 1, 64), gelu(), nn.Linear(64, 1), nn.Sigmoid())
        aux_dim = dim * 2 + STATE_NUM_DIM + HISTORY_DIM + RESOURCE_DIM
        self.count_head = nn.Sequential(nn.LayerNorm(aux_dim), nn.Linear(aux_dim, 128), gelu(), nn.Linear(128, MAX_COUNT_CLASS + 1))
        self.plan_multi_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), gelu(), nn.Linear(128, PLAN_DIM))
        self.plan_type_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 96), gelu(), nn.Linear(96, PLAN_TYPE_CLASSES))
        self.plan_card_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), gelu(), nn.Linear(128, PLAN_CARD_BUCKETS))
        self.residual_scale = 0.42
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def base_grouped(self, state, cards, zones, action_features, action_cards, deck):
        pooled, state_embedding = self.base.encode_state(state, cards, zones)
        batch, actions, _ = action_features.shape
        action_embedding = self.base.action(action_features.reshape(batch * actions, 64)).reshape(batch, actions, 128)
        card_embedding = self.base.card(action_cards)
        deck_embedding = self.base.deck(deck).unsqueeze(1).expand(-1, actions, -1)
        joined = torch.cat((pooled.unsqueeze(1).expand(-1, actions, -1), state_embedding.unsqueeze(1).expand(-1, actions, -1), action_embedding, card_embedding, deck_embedding), dim=-1)
        logits = self.base.policy(joined).squeeze(-1)
        return logits, state_embedding, action_embedding, card_embedding

    def forward(self, state, cards, zones, entity_cards, entity_zones, entity_nums, state_nums, history_nums, resource_nums, action_features, action_cards, action_nums, tactic_nums, deck, action_mask):
        base_logits, state_embedding, action_embedding, card_embedding = self.base_grouped(state, cards, zones, action_features, action_cards, deck)
        entity_mask = entity_cards != 0
        entity = self.entity_project(self.base.card(entity_cards) + self.base.zone(entity_zones) + self.entity_delta(entity_nums))
        global_token = self.state_project(torch.cat((state_embedding, state_nums, history_nums, resource_nums), dim=-1)).unsqueeze(1)
        state_tokens = torch.cat((global_token, entity), dim=1)
        state_mask = torch.cat((torch.ones((state.shape[0], 1), dtype=torch.bool, device=state.device), entity_mask), dim=1)
        for block in self.state_blocks:
            state_tokens = block(state_tokens, state_mask)
        plan_token = state_tokens[:, 0]

        anchor = base_logits.unsqueeze(-1)
        action = self.action_project(torch.cat((action_embedding, card_embedding, self.action_num(action_nums), self.tactic_num(tactic_nums), anchor), dim=-1))
        action = self.cross(action, state_tokens, state_mask)
        action = self.action_block(action, action_mask)
        residual = self.delta(action).squeeze(-1)
        gate = self.gate(torch.cat((action, anchor), dim=-1)).squeeze(-1)
        logits = (base_logits + self.residual_scale * gate * residual).masked_fill(~action_mask, -1e4)
        action_mask_f = action_mask.unsqueeze(-1)
        action_mean = (action * action_mask_f).sum(1) / action_mask_f.sum(1).clamp_min(1)
        action_max = action.masked_fill(~action_mask_f, -1e4).max(1).values
        count_logits = self.count_head(torch.cat((action_mean, action_max, state_nums, history_nums, resource_nums), dim=-1))
        plans = (self.plan_multi_head(plan_token), self.plan_type_head(plan_token), self.plan_card_head(plan_token))
        return logits, base_logits.masked_fill(~action_mask, -1e4), residual, gate, count_logits, plans


def policy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip training-only plan heads so the result loads in the V40 runtime."""
    return {key: value for key, value in state.items() if not key.startswith("plan_")}
