# Method and implementation notes

## Task: ranking a changing set of options

The Python callback receives an observation and a selection request. The
environment, not the model, supplies candidate options. The agent returns indices
into that list, subject to minimum and maximum selection counts.

Behavior cloning learns to reproduce choices in recorded trajectories. It is
supervised imitation learning: the target is a demonstrated choice, not a reward
obtained from a new self-play rollout. The supplied final-model clarification
states that PPO was not used. Older files mentioning PPO are not evidence that
the final submission used it.

## Inspected deployed architecture

The following facts are from the locally inspected, unchanged `submission2.zip`.
The source is withheld from this public repository until its release is cleared.

| Component | Role |
| --- | --- |
| Base scorer | Combines card/zone embeddings, state/action features, and a deck embedding into one score per option |
| State encoder | Two 128-dimensional, four-head self-attention blocks over a global token and board/card entities |
| Action encoder | Cross-attention from candidate actions to state, then self-attention among candidate actions |
| Gated residual | Refines each base score using an action-dependent gate |
| Count head | Predicts selection counts in classes 0 through 16 |
| Callback controller | Maintains public-history features and returns top-ranked option indices |

The final score is `base_score + 0.42 × sigmoid(gate) × residual`.

For a variable-count request, the code masks disallowed count classes and uses
the predicted count if its probability is at least 0.40 and it falls within the
request's bounds. Otherwise it defaults to the maximum permitted count. It then
selects the highest-scoring options. This threshold is part of the supplied
artifact; no threshold-tuning experiment was performed for this portfolio.

The feature pipeline includes 128 base state features, 64 base action features,
48 history features, 128 resource features, and a 96-dimensional tactical vector.
The deployed mask enables the first 72 tactical dimensions. Resource features
estimate remaining availability from the known own deck and visible information;
they are not access to hidden opponent cards or future randomness.

## What “1.40M parameters” means here

The embedded buffer holds exactly 1,402,487 little-endian float32 values across
133 tensors, or 5,609,948 bytes. This counts the **stored artifact**, including
74,113 `base.value.*` values not referenced by the inspected action-scoring path.
It is not a claim that every stored parameter is used in each inference call.

The buffer's SHA-256 checksum, tensor offsets, shapes, and finite float values are
checked by the public inspector. Hashes identify the local artifact, not its
author or the file that Kaggle actually evaluated.

## Training context: V76, not a reproduced experiment

The provided `train_plan_scratch_all.py` initializes the model from scratch and
trains all parameters. It delegates the loss/data-loader implementation to older
modules. The associated method uses a listwise imitation objective, selection-
count supervision, and training-only auxiliary turn-plan targets. Those plan
heads are not present in the final embedded parameter layout.

The supplied V76 dataset report records 31,594 scanned episodes, 2,165,927
decisions, and 12,424,531 option rows. These are **reported data-preparation
counts**, not a dataset regenerated here or a proven description of the exact
checkpoint embedded in GALEX's final file.

The mentor identified V76 as the final solution's source. Source-level behavior
is consistent with that lineage, but the provided materials do not prove a
byte-for-byte training-checkpoint mapping. For example, the deployed deck
embedding has 204 rows, while the supplied V76 metadata lists 181 deck IDs.
An exact checkpoint and export manifest are needed to resolve that gap.

## Engineering limitations to investigate

- **Card metadata fallback:** feature construction attempts to load card/attack
  data from `cg.api`, then catches exceptions and keeps default tables. A NumPy
  smoke test can pass without demonstrating faithful competition features.
- **Schema alignment:** some helper functions expect a `public` field while
  others read `current`; numeric area mappings in a target-slot helper also
  differ from the embedded `AreaType` enumeration. These are review findings,
  not measured explanations for the leaderboard result. Official observations
  are needed to establish their effect.
- **Count range:** the count head supports 0–16, while the callback has a fallback
  for other cases. A few passing fixtures do not validate every request shape.
- **History behavior:** reset and repeated-input paths receive synthetic smoke
  checks, not a full audit of log semantics across complete games.

The original award artifact has not been “fixed” or rewritten. Any later change
should be a separately versioned improvement with its own simulator evaluation.
