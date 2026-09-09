"""Compare the public inference package with a trusted original submission.

The original source is hash-checked before execution. This is a numerical and
interface regression check on synthetic inputs, NOT an official game evaluation.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import platform
import random
import sys
import tempfile

import numpy as np

from ptcg_agent import Agent
from tools.inspect_submission import ROOT, check_expected, inspect_archive, read_archive
from tools.smoke_submission import fixture, validate_selection


def observations() -> list[tuple[str, dict]]:
    result = [("deck", {"select": None})]
    for name, count, minimum, maximum in [
        ("empty", 0, 0, 0), ("single", 1, 1, 1), ("draw_quantity", 3, 1, 1),
        ("fixed_two", 4, 2, 2), ("optional", 5, 0, 3),
        ("variable", 5, 1, 4), ("head_limit", 20, 0, 16),
        ("above_head_limit", 20, 17, 17),
    ]:
        raw = fixture([{"type": 0, "number": i} for i in range(count)], minimum, maximum, len(result))
        result.append((name, raw))
    rng = random.Random(7601)
    for index in range(24):
        count = rng.choice([2, 3, 7, 20, 32])
        maximum = rng.randint(1, min(count, 16))
        minimum = rng.randint(0, maximum)
        raw = fixture([{"type": rng.choice([0, 1, 2, 7, 8, 9, 10, 13, 14]),
                        "number": rng.randint(0, 6)} for _ in range(count)], minimum, maximum, index + 20)
        raw["current"]["turn"] = index // 3 + 1
        raw["current"]["turnActionCount"] = index % 3
        raw["current"]["energyAttached"] = bool(index % 2)
        raw["current"]["supporterPlayed"] = bool(index % 3)
        raw["current"]["players"][0]["deckCount"] = rng.randint(5, 45)
        raw["current"]["players"][0]["handCount"] = rng.randint(1, 12)
        raw["select"]["remainDamageCounter"] = rng.randint(0, 6)
        result.append((f"synthetic_{index:02d}", raw))
    result.append(("step_reset", fixture([{"type": 1}, {"type": 2}], 1, 1, 1)))
    result.append(("repeat", copy.deepcopy(result[-1][1])))
    result.append(("deck_reset", {"select": None}))
    result.append(("after_deck_reset", fixture([{"type": 0, "number": i} for i in range(3)], 1, 1, 1)))
    return result


def capture_scores(controller):
    captured = {}
    original = controller.model.scores_and_count

    def wrapped(*args, **kwargs):
        scores, counts = original(*args, **kwargs)
        captured["scores"] = scores.copy()
        captured["counts"] = counts.copy()
        return scores, counts

    controller.model.scores_and_count = wrapped
    return captured


def compare(archive: Path) -> dict:
    expected = json.loads((ROOT / "evidence/submission_manifest.json").read_text())
    check_expected(inspect_archive(archive), expected)
    files = read_archive(archive)
    with tempfile.TemporaryDirectory(prefix="ptcg_release_compare_") as temporary:
        folder = Path(temporary)
        for name, data in files.items():
            (folder / name).write_bytes(data)
        name = "_ptcg_original_for_comparison"
        spec = importlib.util.spec_from_file_location(name, folder / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            original = module._ACTIVE_CONTROLLER
            released = Agent()
            if original.model.w.keys() != released.model.w.keys():
                raise AssertionError("Parameter keys changed")
            for key in original.model.w:
                np.testing.assert_array_equal(original.model.w[key], released.model.w[key], err_msg=key)
            before, after = capture_scores(original), capture_scores(released)
            score_error = count_error = 0.0
            compared_outputs = 0
            cases = []
            for label, raw in observations():
                before.clear(); after.clear()
                expected_selection = original(copy.deepcopy(raw))
                selection = released(copy.deepcopy(raw))
                if selection != expected_selection:
                    raise AssertionError(f"Selection changed for {label}")
                if raw.get("select") is not None:
                    validate_selection(selection, raw)
                if before.keys() != after.keys():
                    raise AssertionError(f"Different scoring paths for {label}")
                if before:
                    np.testing.assert_array_equal(before["scores"], after["scores"], err_msg=label)
                    np.testing.assert_array_equal(before["counts"], after["counts"], err_msg=label)
                    score_error = max(score_error, float(np.max(np.abs(before["scores"] - after["scores"]))))
                    count_error = max(count_error, float(np.max(np.abs(before["counts"] - after["counts"]))))
                    compared_outputs += 1
                if vars(original.memory) != vars(released.memory) or original.last_step != released.last_step:
                    raise AssertionError(f"History changed for {label}")
                cases.append({"case": label, "selected_indices": selection,
                              "scores_compared": bool(before)})
            return {
                "scope": "original_vs_modular_synthetic_regression_not_match_evaluation",
                "python": platform.python_version(), "numpy": np.__version__,
                "original_main_sha256": expected["files"]["main.py"]["sha256"],
                "parameter_tensors_equal": len(original.model.w),
                "requests_compared": len(cases), "score_vectors_compared": compared_outputs,
                "max_abs_score_difference": score_error, "max_abs_count_logit_difference": count_error,
                "selections_and_history_equal": True,
                "cg_package_discoverable": importlib.util.find_spec("cg") is not None,
                "limitations": ["Artificial observations, not validated reachable game positions",
                                "No official simulator matches or training were rerun",
                                "Equality in this environment does not validate SDK metadata or all possible inputs"],
                "cases": cases,
            }
        finally:
            sys.modules.pop(name, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--trusted", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.trusted:
        parser.error("This executes the original code; inspect it and explicitly pass --trusted")
    report = compare(args.archive)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
