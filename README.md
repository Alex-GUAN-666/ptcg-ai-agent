# The Pokémon Company - PTCG AI Battle Challenge Simulation

**Team GALEX · 264 / 6,807 teams · Top 3.9% · Kaggle Silver Medal**

A case study of a Pokémon Trading Card Game agent that learns decisions from
recorded games through **behavior cloning (imitation learning)**. The approach
combines deck-specific demonstrations, attention-based action scoring, and
lightweight NumPy inference.

[Competition](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
· [Leaderboard](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard)
· [Result certificate](evidence/galex_certificate.png)
· [中文说明](START_HERE_ZH.md)

## What does the agent do?

At each decision, the simulator supplies the visible game state and a set of
legal options. The agent scores those options and returns their indices, subject
to the permitted selection count. It also supplies a 60-card deck when requested.

A turn can contain several decisions: the policy runs again at each request,
rather than generating an entire turn in one pass. The final approach was
behavior cloning, not PPO self-play.

## Approach

1. **Select relevant demonstrations.** The supplied V76 pipeline filters replay
   decisions for the target deck, identified as `3121746f2b28` ("312"). The course
   reference materials describe a rolling window of high-scoring games for
   learning that deck's action patterns.
2. **Represent the decision.** Encode cards, board state, candidate actions,
   public history, resource estimates, and tactical features such as damage,
   energy cost, and evolution.
3. **Rank options and choose a count.** A base policy is refined by state/action
   attention and a gated residual. A separate head predicts how many options to
   select. Training also uses auxiliary targets for the rest of the turn.
4. **Export a compact player.** The preserved submission contains `main.py` with
   embedded float32 weights and `deck.csv`. Inference uses NumPy; training-only
   plan heads are omitted from the export.

The development progressed from attention-based scoring (V20), to tactical
features (V40), to auxiliary turn targets (V67). V76 reused the V67 architecture
and trained all parameters from scratch on an updated dataset.
See [method, data selection, and version history](docs/METHOD.md).

The course retrospective reports that the newest replay window did not always
produce the strongest player, with demonstration quality and changing opponent
strategies offered as possible explanations.

## Results

| Measure | Result | Evidence |
| --- | --- | --- |
| Team placement | 264 / 6,807 · top 3.9% · silver | [GALEX certificate](evidence/galex_certificate.png) |
| Final leaderboard score | 944.2 rating points | Archived leaderboard/submission screenshots |
| Stored model size | 1,402,487 float32 values · 133 tensors | [Artifact manifest](evidence/submission_manifest.json) |
| Verification toolkit | 24 unit tests; 12 local synthetic agent checks | [Validation scope](docs/VALIDATION.md) |

The parameter count includes stored value-head tensors unused by action scoring.
The checks validate artifact integrity and interface behavior, not tournament
strength; leaderboard rating and imitation accuracy are not win percentages.

## What is available and how to run it

This release contains **method documentation, result evidence, and executable
verification tools**. Original agent code and weights are not distributed while
release permission is being confirmed. A fresh clone runs the toolkit, not the
trained player; full training also requires missing upstream dependencies.

With Python 3.12, run the public tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

No third-party packages or competition files are needed for these tests.
For inspecting an authorized local copy of `submission2.zip`, or running the
optional trusted-code smoke checks, see [running the tools](docs/REPRODUCIBILITY.md).

## About this repository

A personal portfolio maintained by **Yuzhen Guan (Alex)**, documenting the GALEX
competition entry. The post-competition documentation and verification toolkit
were prepared with AI assistance. Method notes draw on the preserved submission,
externally supplied reference implementations, and course materials.
See [sources and release status](docs/PROVENANCE.md).

This is an independent participant project, not an official Pokémon or Kaggle product.
