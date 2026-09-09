# Evidence and evaluation scope

Recorded on 2026-09-09.

## Competition evidence

The [participant-provided certificate](../evidence/galex_certificate.png) records
team **GALEX**, **264th of 6,807 teams**, a **Silver Medal**, and an award date of
2026-08-31. The archived [submission page](../evidence/submission_scores.png)
shows **944.2** for the final entry. The same page contains older scores and
failed submissions; they are not a controlled improvement study or win rates.

## Checks performed locally

| Check | Outcome | What it does not prove |
| --- | --- | --- |
| Original ZIP members | `main.py` + `deck.csv` only | Competition acceptance or authorship |
| Parameter buffer | SHA-256, 1,402,487 finite float32 values, 133 contiguous tensor layouts verified | Training quality or exact checkpoint lineage |
| Deck text | 60 positive integer IDs, 19 unique IDs | Full deck legality under competition rules |
| Inspection toolkit suite | 24 tests passed | Playing strength |
| Released-model suite | 13 tests passed using the actual weights | Official simulator compatibility |
| Original-artifact smoke | 12 historical synthetic checks passed | Reachable game states or all game rules |
| Modular-release comparison | 37 requests: selections/history identical; 133 tensor arrays identical | Equivalence on every possible observation |
| Numerical comparison | 34 score/count-vector pairs, maximum absolute differences 0.0 | Official card-data parity or match win rate |

Runtime: Python 3.12.14 / NumPy 2.3.5 on Linux. The `cg` package was absent,
so these checks use fallback metadata, not official card-data parity.
The original `main.py`, deck bytes, and weights were not modified.
The release changes source organization; see [transformations](RELEASE.md).

Evidence files:

- [Original artifact manifest](../evidence/submission_manifest.json).
- [Historical original-artifact smoke report](../evidence/local_smoke_report.json).
- [Generated-module manifest](../evidence/release_manifest.json).
- [Original-to-release parity report](../evidence/release_parity.json).

[GitHub Actions](https://github.com/Alex-GUAN-666/ptcg-ai-agent/actions/workflows/toolkit-tests.yml)
runs the toolkit suite and the actual-model suite/demo. The model job targets
both Linux and Windows; its live result is the authority for CI status.

## Reading the training metrics

The reference materials report validation action agreement around 79% for
later models. Those figures are not reproduced measurements in this repository.
The supplied trainer's top-1 metric checks whether its highest-scoring option
belongs to the demonstrator's selected options; count prediction is evaluated
separately. Neither metric is a tournament win rate.

## Not reproduced

- Full games in the official simulator, seat-balanced matchups, or win rates.
- End-to-end dataset construction, training, checkpoint selection, or export.
- The reference report's validation/test accuracy or head-to-head measurements.
- Ablations attributing a performance gain to any individual model component.

A green test run means these artifact/interface checks passed, not that a
tournament was rerun.

## A useful next evaluation

With a compatible simulator and authorized resources, preserve the original
policy as a baseline, record simulator versions/seeds, test both player seats,
and log invalid selections, failures, time limits, and wins/losses/draws.
Use a fixed dataset split and opponent pool for controlled comparisons.
Publish actual logs and uncertainty estimates before claiming an improvement.
