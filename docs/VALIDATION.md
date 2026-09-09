# Evidence and evaluation scope

Recorded on 2026-09-09.

## Competition evidence

The [participant-provided certificate](../evidence/galex_certificate.png) records
team **GALEX**, **264th of 6,807 teams**, a **Silver Medal**, and an award date of
2026-08-31. The supplied leaderboard/submission screenshots show a final score
of **944.2**. The score is not a win percentage.

These are the participant's archived evidence. They have not been independently
re-fetched from an authenticated Kaggle result API. The related mentor method
report discusses another placement; that ranking is not substituted for GALEX's.

## Checks performed locally

| Check | Outcome | What it does not prove |
| --- | --- | --- |
| Exact ZIP members | `main.py` + `deck.csv` only | Competition acceptance or authorship |
| Parameter buffer | SHA-256, 1,402,487 finite float32 values, 133 contiguous tensor layouts verified | Training quality or exact V76 checkpoint lineage |
| Deck text | 60 positive integer IDs, 19 unique IDs | Full deck legality under competition rules |
| Public unit suite | 24 tests passed | Original model strength |
| Trusted original-artifact smoke | 12 checks passed | Reachable game states, all game rules, or match win rate |

Runtime used: Python 3.12.14 / NumPy 2.3.5 on Linux. The `cg` package was absent,
so these smoke results include fallback metadata, not official card-data parity.
The original `main.py` and `deck.csv` were not modified. Source and weight hashes are recorded in
[submission_manifest.json](../evidence/submission_manifest.json); smoke outputs
are recorded separately in [local_smoke_report.json](../evidence/local_smoke_report.json).

## Not reproduced

- Full games in the official simulator, seat-balanced matchups, or win rates.
- End-to-end dataset construction, training, checkpoint selection, or export.
- The mentor report's validation/test accuracy or head-to-head measurements.
- Ablations attributing a performance gain to any individual model component.

The public tests use fabricated data. A green toolkit test run must not be
described as passing an original-agent tournament test.

## A useful next evaluation

After simulator access and code-release rights are resolved, preserve the original
artifact as a baseline, record the simulator version and seeds, test both player
seats, and log invalid selections, failures, time limits, and wins/losses/draws.
Use an unchanged dataset split and opponent pool for controlled comparisons.
Publish actual logs and uncertainty estimates before claiming an improvement.
