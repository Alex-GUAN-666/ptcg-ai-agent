# Authorized reference implementation; not an end-to-end runnable training release.
# See training_reference/README.md and NOTICE.md.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "v66/artifacts/structured_v40_history.npz"
OUTPUT = ROOT / "v67/artifacts/plan_targets.npz"


def plan_class(action_type: int) -> int:
    if action_type in (7,):
        return 0  # play/search/draw
    if action_type == 8:
        return 1  # attach
    if action_type == 9:
        return 2  # evolve
    if action_type == 10:
        return 3  # ability
    if action_type in (11,):
        return 4  # discard/resource choice
    if action_type == 12:
        return 5  # retreat/switch
    if action_type == 13:
        return 6  # attack
    return 7      # end/other


def first_selected(data: dict[str, np.ndarray], decision: int) -> tuple[int, int]:
    start, end = int(data["starts"][decision]), int(data["ends"][decision])
    labels = data["labels"][start:end]
    selected = np.flatnonzero(labels > 0)
    if not len(selected):
        return 0, 0
    row = start + int(selected[0])
    action_type = int(round(float(data["action_features"][row, 0]) * 24.0))
    card = int(data["action_cards"][row])
    return plan_class(action_type) + 1, (card % 256) + 1 if card > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    archive = np.load(args.dataset, mmap_mode="r")
    data = {key: archive[key] for key in archive.files}
    n = len(data["episode_id"])
    plan_multi = np.zeros((n, 8), np.uint8)
    next_type = np.zeros(n, np.uint8)
    next_card = np.zeros(n, np.uint16)
    atomic_weight = np.ones(n, np.float16)
    group_id = np.zeros(n, np.int64)
    turns = np.rint(data["history_nums"][:, 1].astype(np.float32) * 80.0).astype(np.int32)
    group = 0
    start = 0
    while start < n:
        episode = int(data["episode_id"][start]); turn = int(turns[start]); split = int(data["split"][start])
        end = start + 1
        while end < n and int(data["episode_id"][end]) == episode and int(turns[end]) == turn and int(data["split"][end]) == split:
            end += 1
        size = end - start
        atomic_weight[start:end] = np.float16(1.0 / max(size, 1))
        group_id[start:end] = group
        future = np.zeros(8, np.uint8)
        for decision in range(end - 1, start - 1, -1):
            if decision + 1 < end:
                typ, card = first_selected(data, decision + 1)
                next_type[decision] = typ
                next_card[decision] = card
                if typ > 0:
                    future[typ - 1] = 1
            plan_multi[decision] = future
        group += 1
        start = end
        if group % 100000 == 0:
            print(f"groups={group} decisions={start}/{n}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, plan_multi=plan_multi, next_type=next_type, next_card=next_card, atomic_weight=atomic_weight, group_id=group_id, source_split=data["split"])
    report = {"decisions": n, "groups": group, "dataset": str(args.dataset), "output": str(args.output), "split_unchanged": True, "future_plan_rate": float(plan_multi.any(1).mean()), "next_card_rate": float((next_card > 0).mean())}
    (args.output.parent / "plan_targets.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("V67_PLAN_TARGETS=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
