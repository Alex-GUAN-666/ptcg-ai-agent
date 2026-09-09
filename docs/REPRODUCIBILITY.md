# Reproduction: what is runnable today?

## 1. Run the released player on a synthetic decision

Use **Python 3.12** and open a terminal in the repository root:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` in macOS/Linux. Alternatively, use the environment's
Python directly, as shown in the [Windows guide](../START_HERE_ZH.md).

```bash
python -m pip install -r requirements.txt
python -m examples.decision_demo
```

The actual packaged model ranks three NUMBER options (0, 1, and 2) in a
fabricated DRAW_COUNT request. It returns an option **index**, not the option's
value. In the recorded NumPy-only environment the selected index is 2.
Printed scores are preferences, not probabilities.

No GPU, API key, separately downloaded checkpoint, or LLM service is needed.
The package contains the preserved float32 weights. Without the competition
`cg` SDK, the original feature code uses fallback card metadata. This example
does not represent a validated reachable game state or measure playing strength.

For integration into a compatible simulator:

```python
from ptcg_agent import Agent

player = Agent()  # one independent history per game
deck = player({"select": None})
# selected_indices = player(observation_from_your_compatible_simulator)
```

A module-level `agent(observation, configuration=None)` callback is also exported.
It shares one instance and must not be used for concurrent independent games.
The modular package is not automatically a validated Kaggle submission ZIP.

## 2. Run the public tests

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests_model -v
```

The first suite has 24 standard-library tests for the inspection toolkit.
The second has 13 tests that load the released weights, verify hashes, exercise
synthetic requests, check history isolation, and compare recorded selections.
Neither suite runs official games or trains a model. CI runs the model suite
on Linux and Windows. See [validation scope](VALIDATION.md).

## 3. Optional checks against the separately held original ZIP

These commands are for holders of the original `submission2.zip`; that ZIP is
not needed to run the public model or tests.

Static integrity inspection, without executing the source:

```bash
python -m tools.inspect_submission /path/to/submission2.zip --expected evidence/submission_manifest.json
```

For a trusted original only, execute its synthetic checks and compare it with
the modular release:

```bash
python -m tools.smoke_submission /path/to/submission2.zip --trusted
python -m tools.compare_release /path/to/submission2.zip --trusted
```

On Windows, substitute your quoted path, for example
`"D:\Projects\PTCG\submission2.zip"`. These tools require matching original
member/weight hashes before execution. The trusted-code tools are **not security
sandboxes**. Keep the original outside the repository or under ignored
`local_artifacts/`.

The [recorded comparison](../evidence/release_parity.json) used Python 3.12.14 /
NumPy 2.3.5, without `cg`: 133 identical parameter tensors, 37 requests with
matching selections/history, and 34 score/count-vector pairs with zero
difference. For mechanical source extraction, see [build_release.py](../tools/build_release.py)
and [release notes](RELEASE.md).

## 4. What is missing for full reproduction?

[Training references](../training_reference/README.md) expose the mature model
and training logic but depend on earlier `v7`, `v14`, `v20`, and `v40`
modules and prepared data absent from the supplied bundles. The references are
not a working end-to-end trainer; PyTorch alone does not resolve those imports.
Original simulator installation, training arrays, and exact checkpoint/export
metadata are also missing.

The supplied scratch-training entry point initializes all parameters randomly.
Its defaults describe code configuration, not an independently reproduced run.
See [training details](METHOD.md#training-and-model-evolution).

There is an unresolved checkpoint/export alignment detail: the preserved
inference artifact has 204 deck-embedding rows, while the supplied V76 metadata
lists 181 deck IDs. The exact checkpoint and export metadata are needed to
connect the training bundle to the final binary weights.

A full reproduction needs the missing dependencies/data, applicable access and
usage rights, checkpoint alignment, and a clean-environment simulator run.
No reported training accuracy, seat-balanced matchup, or leaderboard placement
has been reproduced by the local demo or automated tests.
