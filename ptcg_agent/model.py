# Extracted from the preserved GALEX submission by tools/build_release.py.
# Inference calculations are retained; see docs/RELEASE.md and NOTICE.md.
import numpy as np
from .weights import load_parameters as _restore_parameter_map


def _smooth_activation(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))

def _dense_layer(x, weight, bias):
    return x @ weight.T + bias

def _normalize_layer(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-05) * weight + bias

def _probability_vector(x, axis=-1):
    y = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(y)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-09)

class BasePolicy:

    def __init__(self, path=None):
        self.w = _restore_parameter_map()
        self.heads = 4
        self.residual_scale = 0.42

    def seq_block(self, x, prefix, linear_indices, norm_index=None):
        for index in linear_indices:
            x = _smooth_activation(_dense_layer(x, self.w[f'{prefix}.{index}.weight'], self.w[f'{prefix}.{index}.bias']))
        if norm_index is not None:
            x = _normalize_layer(x, self.w[f'{prefix}.{norm_index}.weight'], self.w[f'{prefix}.{norm_index}.bias'])
        return x

    def base_scores(self, x, cards, zones, action_cards, deck_index):
        w = self.w
        card_ids = np.clip(cards, 0, w['base.card.weight'].shape[0] - 1)
        zone_ids = np.clip(zones, 0, w['base.zone.weight'].shape[0] - 1)
        mask = (cards != 0)[..., None]
        token = self.seq_block(w['base.card.weight'][card_ids] + w['base.zone.weight'][zone_ids], 'base.token', (0, 2), 4) * mask
        pooled = token.sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        state = self.seq_block(x[:, :128], 'base.state', (0, 2), 4)
        action = self.seq_block(x[:, 128:192], 'base.action', (0, 2), 4)
        deck = np.full(len(x), deck_index, dtype=np.int64)
        action_ids = np.clip(action_cards, 0, w['base.card.weight'].shape[0] - 1)
        deck_ids = np.clip(deck, 0, w['base.deck.weight'].shape[0] - 1)
        joined = np.concatenate((pooled, state, action, w['base.card.weight'][action_ids], w['base.deck.weight'][deck_ids]), axis=1)
        hidden = _smooth_activation(_dense_layer(joined, w['base.policy.0.weight'], w['base.policy.0.bias']))
        hidden = _smooth_activation(_dense_layer(hidden, w['base.policy.3.weight'], w['base.policy.3.bias']))
        logits = _dense_layer(hidden, w['base.policy.5.weight'], w['base.policy.5.bias']).reshape(-1)
        return (logits, pooled[0], state[0], action, w['base.card.weight'][action_ids])

    def self_attn(self, x, mask, prefix):
        y = _normalize_layer(x, self.w[f'{prefix}.norm.weight'], self.w[f'{prefix}.norm.bias'])
        tokens, dim = y.shape
        head_dim = dim // self.heads
        q = _dense_layer(y, self.w[f'{prefix}.q.weight'], self.w[f'{prefix}.q.bias']).reshape(tokens, self.heads, head_dim).transpose(1, 0, 2)
        k = _dense_layer(y, self.w[f'{prefix}.k.weight'], self.w[f'{prefix}.k.bias']).reshape(tokens, self.heads, head_dim).transpose(1, 0, 2)
        v = _dense_layer(y, self.w[f'{prefix}.v.weight'], self.w[f'{prefix}.v.bias']).reshape(tokens, self.heads, head_dim).transpose(1, 0, 2)
        logits = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(float(head_dim))
        logits[:, :, ~mask] = -10000.0
        attn = _probability_vector(logits, -1)
        out = np.matmul(attn, v).transpose(1, 0, 2).reshape(tokens, dim)
        return x + _dense_layer(out, self.w[f'{prefix}.out.weight'], self.w[f'{prefix}.out.bias'])

    def cross_attn(self, x, source, source_mask, prefix):
        q_in = _normalize_layer(x, self.w[f'{prefix}.q_norm.weight'], self.w[f'{prefix}.q_norm.bias'])
        kv_in = _normalize_layer(source, self.w[f'{prefix}.kv_norm.weight'], self.w[f'{prefix}.kv_norm.bias'])
        tokens, dim = q_in.shape
        source_tokens = kv_in.shape[0]
        head_dim = dim // self.heads
        q = _dense_layer(q_in, self.w[f'{prefix}.q.weight'], self.w[f'{prefix}.q.bias']).reshape(tokens, self.heads, head_dim).transpose(1, 0, 2)
        k = _dense_layer(kv_in, self.w[f'{prefix}.k.weight'], self.w[f'{prefix}.k.bias']).reshape(source_tokens, self.heads, head_dim).transpose(1, 0, 2)
        v = _dense_layer(kv_in, self.w[f'{prefix}.v.weight'], self.w[f'{prefix}.v.bias']).reshape(source_tokens, self.heads, head_dim).transpose(1, 0, 2)
        logits = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(float(head_dim))
        logits[:, :, ~source_mask] = -10000.0
        attn = _probability_vector(logits, -1)
        out = np.matmul(attn, v).transpose(1, 0, 2).reshape(tokens, dim)
        return x + _dense_layer(out, self.w[f'{prefix}.out.weight'], self.w[f'{prefix}.out.bias'])

    def ff(self, x, prefix):
        y = _normalize_layer(x, self.w[f'{prefix}.norm.weight'], self.w[f'{prefix}.norm.bias'])
        y = _smooth_activation(_dense_layer(y, self.w[f'{prefix}.net.0.weight'], self.w[f'{prefix}.net.0.bias']))
        y = _dense_layer(y, self.w[f'{prefix}.net.2.weight'], self.w[f'{prefix}.net.2.bias'])
        return x + y

    def encoder_block(self, x, mask, prefix):
        x = self.self_attn(x, mask, f'{prefix}.attn')
        return self.ff(x, f'{prefix}.ff')

class NumpyPolicy(BasePolicy):

    def __init__(self, path):
        super().__init__(path)

    def scores_and_count(self, x, cards, zones, extra, action_features, action_cards, deck_index):
        base_logits, _, state_embedding, action_embedding, action_card_embedding = self.base_scores(x, cards, zones, action_cards, deck_index)
        w = self.w
        entity_cards = np.clip(extra['entity_cards'], 0, w['base.card.weight'].shape[0] - 1)
        entity_zones = np.clip(extra['entity_zones'], 0, w['base.zone.weight'].shape[0] - 1)
        entity_delta = self.seq_block(extra['entity_nums'], 'entity_delta', (0,), 2)
        entity = self.seq_block(w['base.card.weight'][entity_cards] + w['base.zone.weight'][entity_zones] + entity_delta, 'entity_project', (0,), 2)
        state_input = np.concatenate((state_embedding, extra['state_nums'], extra['history_nums'], extra['resource_nums']))[None, :]
        global_token = self.seq_block(state_input, 'state_project', (0,), 2)
        state_tokens = np.concatenate((global_token, entity), axis=0)
        state_mask = np.concatenate((np.ones(1, bool), extra['entity_cards'] != 0))
        state_tokens = self.encoder_block(state_tokens, state_mask, 'state_blocks.0')
        state_tokens = self.encoder_block(state_tokens, state_mask, 'state_blocks.1')
        action_num = self.seq_block(extra['action_nums'], 'action_num', (0,), 2)
        tactic_num = self.seq_block(extra['tactic_nums'], 'tactic_num', (0,), 2)
        action = self.seq_block(np.concatenate((action_embedding, action_card_embedding, action_num, tactic_num, base_logits[:, None]), axis=1), 'action_project', (0,), 2)
        action_mask = np.ones(len(action), bool)
        action = self.cross_attn(action, state_tokens, state_mask, 'cross')
        action = self.encoder_block(action, action_mask, 'action_block')
        delta_hidden = _normalize_layer(action, w['delta.0.weight'], w['delta.0.bias'])
        delta_hidden = _smooth_activation(_dense_layer(delta_hidden, w['delta.1.weight'], w['delta.1.bias']))
        residual = _dense_layer(delta_hidden, w['delta.3.weight'], w['delta.3.bias']).reshape(-1)
        gate_hidden = _normalize_layer(np.concatenate((action, base_logits[:, None]), axis=1), w['gate.0.weight'], w['gate.0.bias'])
        gate_hidden = _smooth_activation(_dense_layer(gate_hidden, w['gate.1.weight'], w['gate.1.bias']))
        gate = 1.0 / (1.0 + np.exp(-_dense_layer(gate_hidden, w['gate.3.weight'], w['gate.3.bias']).reshape(-1)))
        scores = base_logits + self.residual_scale * gate * residual
        action_mean = action.mean(axis=0)
        action_max = action.max(axis=0)
        count_input = np.concatenate((action_mean, action_max, extra['state_nums'], extra['history_nums'], extra['resource_nums']))[None, :]
        count_hidden = _normalize_layer(count_input, w['count_head.0.weight'], w['count_head.0.bias'])
        count_hidden = _smooth_activation(_dense_layer(count_hidden, w['count_head.1.weight'], w['count_head.1.bias']))
        count_logits = _dense_layer(count_hidden, w['count_head.3.weight'], w['count_head.3.bias']).reshape(-1)
        return (scores, count_logits)
