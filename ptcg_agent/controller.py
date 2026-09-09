# Extracted from the preserved GALEX submission by tools/build_release.py.
# Inference calculations are retained; see docs/RELEASE.md and NOTICE.md.
from pathlib import Path
import numpy as np
from .model import NumpyPolicy, _probability_vector
from .schema import (
    _decode_game_observation,
)
from .features import (
    _new_history_tracker,
    encode_options,
    encode_v40,
    legal_count_mask,
    update_memory,
)


_DEPLOYED_FEATURE_MASK = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)

class DecisionController:

    def __init__(self, base, deck_index, deck):
        self.base = Path(base)
        self.deck_index = int(deck_index)
        self.deck = deck
        self.model = NumpyPolicy(None)
        self.feature_mask = _DEPLOYED_FEATURE_MASK.copy()
        self.memory = _new_history_tracker()
        self.last_step = -1

    def __call__(self, observation, configuration=None):
        if observation.get('select') is None:
            self.memory = _new_history_tracker()
            self.last_step = -1
            return self.deck
        step = int(observation.get('step', self.last_step + 1) or 0)
        if step < self.last_step:
            self.memory = _new_history_tracker()
        self.last_step = step
        self.memory = update_memory(self.memory, observation)
        obs = _decode_game_observation(observation)
        count = len(obs.select.option)
        if count == 0:
            return []
        x, cards, zones, _, action_cards = encode_options(obs)
        extra = encode_v40(observation, x, action_cards, self.memory, self.deck)
        extra['tactic_nums'] = extra['tactic_nums'] * self.feature_mask[None, :]
        scores, count_logits = self.model.scores_and_count(x.astype(np.float32), cards, zones, extra, x[:, 128:192].astype(np.float32), action_cards, self.deck_index)
        minimum = int(obs.select.minCount)
        maximum = int(min(obs.select.maxCount, count))
        take = max(minimum, maximum)
        if minimum < maximum:
            mask = legal_count_mask(minimum, maximum)
            masked = count_logits.copy()
            masked[~mask] = -10000.0
            probs = _probability_vector(masked)
            predicted = int(masked.argmax())
            if minimum <= predicted <= maximum and probs[predicted] >= 0.4:
                take = predicted
        ranking = np.argsort(-scores)
        return [int(index) for index in ranking[:take]]
