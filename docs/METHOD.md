# Method: deck-specific behavior cloning

## Task: ranking a changing set of options

The Python callback receives an observation and a selection request. The
environment, not the model, supplies candidate options. The agent returns indices
into that list, subject to minimum and maximum selection counts.

Behavior cloning fits a policy to demonstrated observation/action pairs using
supervised learning. The supplied final-model clarification and course retrospective
identify BC as the submitted approach; PPO was not used in the final model.
See the [behavior-cloning definition](https://imitation.readthedocs.io/en/latest/algorithms/bc.html).

One game turn can require many selection requests. The player makes a fresh
prediction for each request, rather than predicting a complete action sequence
once per turn. The initial deck callback returns a fixed 60-card list.

## Demonstrations and data selection

The course retrospective (11:13–12:39 and 20:41–22:06) describes collecting
high-scoring public replay episodes using a rolling window of about 15 days and
focusing on the attack-oriented deck nicknamed **312**. This is a shortened deck
identifier, not the number of cards in the deck.

The supplied V76 builder targets `3121746f2b28` using `--target-only` and specifies
different per-date limits across July 20–August 3, 2026. Its dataset report gives:

| Prepared-data measure | Reported count |
| --- | --- |
| Replay episodes scanned | 31,594 |
| Target-deck decisions | 2,165,927 |
| Candidate-option rows across those decisions | 12,424,531 |

These counts cover the supplied prepared dataset, before the trainer's
positive-weight filtering; option rows are not independent games. The dataset
was not regenerated for this repository. Exact filtering and weighting details
also depend on upstream V40 modules absent from the supplied bundle.

### Why the replay window mattered

In the experiments described in the course materials, some middle-period models performed
better than earlier and later ones. The proposed explanation was that early
demonstrators were still improving, while later opponents increasingly countered
the target deck, changing the situations and behaviors available to imitate.

This is a **course-reported observation and hypothesis**, not a controlled
replay-window ablation. The practical lesson is to consider demonstrator quality
and relevance to the intended strategy, rather than assuming newer or larger
datasets are always better. A useful follow-up would compare replay windows
with the architecture, evaluation opponents, and player seats held fixed.

## Inspected deployed architecture

The following describes the locally inspected, unchanged `submission2.zip`.
The same inference calculations and weight bytes are now published in
[model.py](../ptcg_agent/model.py), [features.py](../ptcg_agent/features.py), and
[controller.py](../ptcg_agent/controller.py). See [release transformations](RELEASE.md).

| Component | Role |
| --- | --- |
| Base scorer | Combines card/zone embeddings, state/action features, and a deck embedding into one score per option |
| State encoder | Two 128-dimensional, four-head self-attention blocks over a global token and board/card entities |
| Action encoder | Cross-attention from candidate actions to state, then self-attention among candidate actions |
| Gated residual | Refines each base score using an action-dependent gate |
| Count head | Predicts selection counts in classes 0 through 16 |
| Callback controller | Maintains public-history features and returns top-ranked option indices |

The final score is `base_score + 0.42 × sigmoid(gate_logit) × residual`.
The factor 0.42 scales the gated correction, not the base policy score.

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

### What "1.40M parameters" means

The preserved inference buffer holds 1,402,487 little-endian float32 values
across 133 tensors, or 5,609,948 bytes. This is the stored artifact count,
including 74,113 `base.value.*` values not used by the action-scoring path.

The public inspector checks the buffer's SHA-256 checksum, tensor offsets,
shapes, and finite float values. The meeting's approximate model-size estimate
is not used in place of this measured export count.

## Training and model evolution

The retrospective (23:01–29:29) explains the development sequence. V67/V76 source
files support the later stages; V20/V40 history comes from the course materials.
These are internal iteration numbers, not standardized algorithm names. The
[training-reference guide](../training_reference/README.md) links the corresponding
source files and identifies missing dependencies.

| Version | Main change | Purpose |
| --- | --- | --- |
| V20 | State/action attention and interactions among candidate actions | Condition action scores on the board and other available choices |
| V40 | Tactical feature branch | Represent damage, energy cost, HP, and evolution beyond card identity |
| V67 | Three training-only turn-plan heads | Add supervision about the remainder of the current turn |
| V76 | Same V67 architecture, updated demonstrations, random initialization of all parameters | Jointly learn the complete policy from scratch rather than warm-starting it |

The V67 training script already includes a head-only warm-up followed by joint
fine-tuning from a V40 checkpoint. V76's distinguishing change is **no warm start**,
with all parameters trainable from the beginning; it is not the first version
ever to update the earlier layers jointly.

### Auxiliary targets and losses

The V67 target builder uses later recorded decisions in the same episode/turn/
split group to supervise three heads:

- Which of eight broad action categories occur later in the turn.
- The next action category, including a no-next-action class.
- A bucketed next-card identifier.

These are training labels, not future information supplied to the player at
inference. The model consumes current-state features; `policy_state_dict`
removes all `plan_*` tensors when exporting the policy.

The inherited training objective combines listwise behavior cloning, a base-
policy imitation term, pairwise ranking, selection-count loss, residual/gate
regularization, and weighted auxiliary losses. This supervises both action
preference and the number of selected options.

### V76 training configuration

The supplied entry point uses AdamW, batch size 128 by default, and two
configurable phases:

| Phase | Default epochs | Learning rate |
| --- | --- | --- |
| Scratch training | 20 | 0.0003 |
| Refinement | 8 | 0.00008 |

The course retrospective reports useful checkpoints after roughly ten or more
epochs and little benefit from refinement. That observation does not change the
script defaults or establish the exact epoch used by GALEX. The script selects
a checkpoint primarily by validation top-1 agreement, with loss and auxiliary
accuracy as tie-breakers.

The meeting mentions roughly 70% validation action accuracy for an earlier
version and around 79% for later models. These are **course-reported measurements
from different runs**, not a reproduced, fixed-data ablation. In the inspected
trainer, top-1 agreement means the highest-scoring option is among the
demonstrator's selected options. It is neither exact multi-option-set accuracy
nor a game win rate. No V76 training run was reproduced for this portfolio.

## Engineering limitations to investigate

- **Card metadata fallback:** feature construction attempts to load card/attack
  data from `cg.api`, then catches exceptions and keeps default tables. A NumPy
  smoke test can pass without demonstrating faithful competition features.
- **Schema alignment:** some helper functions expect a `public` field while
  others read `current`; numeric area mappings in a target-slot helper also
  differ from the embedded `AreaType` enumeration. These are review findings,
  not measured explanations for the leaderboard result.
- **Count range:** the count head supports 0–16, while the callback has a fallback
  for other cases. A few passing fixtures do not validate every request shape.
- **History behavior:** reset and repeated-input paths receive synthetic smoke
  checks, not a full audit of log semantics across complete games.

Exact checkpoint-to-export lineage and full training dependencies remain open;
see [reproduction status](REPRODUCIBILITY.md). The preserved competition artifact
is unchanged. Any later model modification should be evaluated as a separate version.
