"""Run a trusted, locally held GALEX submission on synthetic interface fixtures.

This imports and executes main.py; it is NOT a security sandbox. Only run code
you trust. The recorded artifact hashes must match. No simulator or match
outcomes are tested. The original code and weights are not distributed here.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile

from tools.inspect_submission import ROOT, check_expected, inspect_archive, read_archive, require


def fixture(options: list[dict], minimum: int, maximum: int, step: int = 1) -> dict:
    """Artificial schema-shaped input, not a recorded or simulated legal position."""
    side = {
        "active": [], "bench": [], "benchMax": 5, "deckCount": 47,
        "discard": [], "prize": [None] * 6, "handCount": 7, "hand": [],
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    opponent = copy.deepcopy(side)
    opponent["hand"] = None
    return {
        "step": step,
        "logs": [],
        "current": {
            "turn": 1, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": -1,
            "stadium": [], "looking": None,
            "players": [copy.deepcopy(side), opponent],
        },
        "select": {
            "type": 8, "context": 38, "minCount": minimum, "maxCount": maximum,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": options, "deck": None, "contextCard": None, "effect": None,
        },
    }


def validate_selection(action, observation: dict) -> None:
    selection = observation["select"]
    require(isinstance(action, list), "Agent must return a list")
    require(all(type(x) is int for x in action), "Indices must be Python integers")
    require(len(set(action)) == len(action), "Duplicate option index")
    require(selection["minCount"] <= len(action) <= selection["maxCount"],
            "Selected count out of bounds")
    require(all(0 <= x < len(selection["option"]) for x in action),
            "Option index out of bounds")


def run_smoke(archive: Path) -> dict:
    import numpy as np

    recorded = json.loads((ROOT / "evidence/submission_manifest.json").read_text(encoding="utf-8"))
    check_expected(inspect_archive(archive), recorded)
    files = read_archive(archive)
    checks = []
    with tempfile.TemporaryDirectory(prefix="ptcg_trusted_smoke_") as temporary:
        folder = Path(temporary)
        # Names have already been restricted to main.py and deck.csv; no extractall.
        for name, data in files.items():
            (folder / name).write_bytes(data)
        module_name = "_ptcg_trusted_submission"
        spec = importlib.util.spec_from_file_location(module_name, folder / "main.py")
        require(spec is not None and spec.loader is not None, "Cannot load module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            deck = [int(line) for line in files["deck.csv"].decode().splitlines()]
            require(module.agent({"select": None}, None) == deck, "Deck callback mismatch")
            checks.append("deck_callback")
            cases = [
                ("empty_options", 0, 0, 0), ("single_option", 1, 1, 1),
                ("choose_one_of_two", 2, 1, 1), ("choose_two_of_four", 4, 2, 2),
                ("variable_zero_to_three", 5, 0, 3),
                ("variable_one_to_four", 5, 1, 4), ("count_head_limit", 20, 0, 16),
                ("fixed_count_above_head_limit", 20, 17, 17),
            ]
            outputs = {}
            for step, (name, count, minimum, maximum) in enumerate(cases, start=1):
                obs = fixture([{"type": 0, "number": i} for i in range(count)], minimum, maximum, step)
                action = module.agent(obs, None)
                validate_selection(action, obs)
                outputs[name] = action
                checks.append(name)
            obs = fixture([{"type": 1}, {"type": 2}], 1, 1, 1)
            validate_selection(module.agent(obs, None), obs)
            checks.append("step_reset")
            first = module.agent(obs, None)
            require(module.agent(copy.deepcopy(obs), None) == first, "Repeated input is not stable")
            checks.append("repeated_input")
            encoded = module._decode_game_observation(obs)
            x, cards, zones, _, action_cards = module.encode_options(encoded)
            extra = module.encode_v40(obs, x, action_cards, module._new_history_tracker(), deck)
            extra["tactic_nums"] *= module._DEPLOYED_FEATURE_MASK[None, :]
            scores, counts = module._ACTIVE_CONTROLLER.model.scores_and_count(
                x.astype(np.float32), cards, zones, extra, x[:, 128:192], action_cards, 0
            )
            require(scores.shape == (2,) and counts.shape == (17,), "Output shape mismatch")
            require(np.isfinite(scores).all() and np.isfinite(counts).all(), "Non-finite outputs")
            checks.append("finite_scores_and_count_logits")
            return {
                "scope": "synthetic_interface_only_not_a_game_evaluation",
                "python": platform.python_version(), "numpy": np.__version__,
                "cg_package_detected": importlib.util.find_spec("cg") is not None,
                "checks_passed": len(checks), "checks": checks,
                "example_selections": outputs,
                "limitations": [
                    "No official simulator matches or win rates were evaluated.",
                    "Synthetic observations need not represent reachable game states.",
                    "Card/attack metadata fidelity is not checked; cg.api may be absent.",
                    "Passing does not prove all possible selections or game rules are handled.",
                ],
            }
        finally:
            sys.modules.pop(module_name, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--trusted", action="store_true", help="Acknowledge that Python code will execute")
    args = parser.parse_args()
    if not args.trusted:
        parser.error("Read main.py first, then supply --trusted; this command executes that code.")
    print(json.dumps(run_smoke(args.archive), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
