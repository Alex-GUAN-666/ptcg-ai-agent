# Training references — partial source, not a runnable training release

These supplied files explain how the mature behavior-cloning architecture and
training objectives were implemented. They are **not required for inference**.
Installing PyTorch alone is not enough to run the trainers: upstream feature,
base-model, dataset, and loss modules are missing.

## Read in this order

| File | What to inspect |
| --- | --- |
| [v67/model_v67.py](v67/model_v67.py) | Attention-based policy, three auxiliary heads, and removal of `plan_*` tensors at export |
| [v67/build_plan_targets.py](v67/build_plan_targets.py) | Labels from subsequent decisions in the same episode/turn/split group |
| [v67/train_plan_bc.py](v67/train_plan_bc.py) | Head warm-up, then joint BC fine-tuning from an earlier checkpoint |
| [v76/build_dataset_latest.py](v76/build_dataset_latest.py) | Target-deck filter and per-date replay limits |
| [v76/train_plan_scratch_all.py](v76/train_plan_scratch_all.py) | Same mature architecture, random initialization, all parameters trainable |

V67 and V76 are internal iteration labels. V76 is not a different algorithm;
it changes initialization and training data while reusing the mature model.
Historical PPO scripts are not included because PPO was not the final
submission's training route.

## What is still needed

- Earlier project modules, including `features_v14`, `features_v40`,
  `model_torch`, `model_v14`, and the dataset/loss utilities dynamically imported
  by these scripts.
- Prepared episode/option arrays and auxiliary targets, or the authorized
  replay data and complete pipeline needed to generate them.
- Checkpoint/export metadata linking a particular training run to the preserved
  submission, plus the compatible simulator/SDK for game evaluation.

The [dataset summary](dataset_summary.json) reports 31,594 scanned episodes,
2,165,927 target-deck decisions, and 12,424,531 option rows before trainer
filtering. It was not regenerated here; team identifiers and local paths are
omitted. Dataset rows are not independent games.

## Changes made for publication

The selected scripts retain their original logic. Publication adds reference-
status comments and replaces a machine-specific Windows project root with a
path relative to this directory. That path change does not supply missing
dependencies or establish portability of the complete training pipeline.
Original and published hashes are in [source_manifest.json](source_manifest.json).

See [method details](../docs/METHOD.md), [reproduction limitations](../docs/REPRODUCIBILITY.md),
and [NOTICE](../NOTICE.md). No training run, reported accuracy, or match win rate
was reproduced by publishing these references.
