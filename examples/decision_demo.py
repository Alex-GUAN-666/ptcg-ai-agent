"""Show option scores and the chosen index for a synthetic draw-count request."""

import importlib.util
import json

import numpy as np

from ptcg_agent import Agent
from ptcg_agent.features import _new_history_tracker, encode_options, encode_v40, update_memory
from ptcg_agent.schema import _decode_game_observation
from tools.smoke_submission import fixture, validate_selection


def demo_observation() -> dict:
    # A simulator may request a number by offering NUMBER options. This made-up
    # observation is for the interface demo, not a validated reachable position.
    return fixture([{"type": 0, "number": number} for number in (0, 1, 2)], 1, 1)


def run_demo() -> dict:
    observation = demo_observation()
    player = Agent()
    decoded = _decode_game_observation(observation)
    x, cards, zones, _, action_cards = encode_options(decoded)
    memory = update_memory(_new_history_tracker(), observation)
    extra = encode_v40(observation, x, action_cards, memory, player.deck)
    extra["tactic_nums"] *= player.feature_mask[None, :]
    scores, count_logits = player.model.scores_and_count(
        x.astype(np.float32), cards, zones, extra, x[:, 128:192], action_cards, player.deck_index
    )
    selected = player(observation)
    validate_selection(selected, observation)
    return {
        "scope": "synthetic_interface_demo_not_a_match",
        "request": "Choose one offered draw quantity",
        "options": [{"index": i, "number": option["number"],
                     "score": float(scores[i]), "selected": i in selected}
                    for i, option in enumerate(observation["select"]["option"])],
        "selected_indices": selected,
        "count_logit_classes": int(count_logits.size),
        "cg_package_discoverable": importlib.util.find_spec("cg") is not None,
        "note": "Scores are preferences, not win probabilities. Official card metadata and match performance are not validated.",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
