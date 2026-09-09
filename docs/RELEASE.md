# Modular inference release

This release makes the preserved player readable and locally executable. It
does not retrain, improve, or replace the competition policy.

## What changed

The original archive contains a large `main.py` with embedded base64 weights
and a `deck.csv`. [The release builder](../tools/build_release.py) parses that
file's syntax tree without executing it and:

- Splits the observation schema, feature functions, NumPy model, and final
  controller into separate modules.
- Moves the exact original float32 bytes into `parameters.f32`, with a JSON
  layout and checksum validation. No pickle or executable checkpoint is loaded.
- Renames the runtime classes to `BasePolicy`, `NumpyPolicy`, and
  `DecisionController`.
- Omits an earlier controller shadowed by a later definition and an unused,
  overridden base forward method. Original feature overrides are retained in
  order; calculations in the selected implementation are preserved.

The public `Agent` wrapper creates one controller per game; the optional module
callback lazily creates one shared instance. Do not share a single instance
between concurrent games.

| Module | Responsibility |
| --- | --- |
| [schema.py](../ptcg_agent/schema.py) | Decode simulator-shaped observations |
| [features.py](../ptcg_agent/features.py) | Encode options and visible-state/history features |
| [model.py](../ptcg_agent/model.py) | Compute option scores and selection-count logits |
| [controller.py](../ptcg_agent/controller.py) | Update history and apply the original selection rules |
| [weights.py](../ptcg_agent/weights.py) | Validate and load the packaged read-only float32 arrays |

The original ZIP is not duplicated in this repository. Member hashes are in
[submission_manifest.json](../evidence/submission_manifest.json); generated
module and asset hashes are in [release_manifest.json](../evidence/release_manifest.json).

## Preservation checks

[compare_release.py](../tools/compare_release.py) loads the hash-checked original
and the modular player in the same runtime. The recorded comparison covers:

- All 133 stored parameter tensors, compared bit-for-bit.
- 37 synthetic requests: deck/empty cases, count bounds, varied candidates,
  repeated input, and history-reset paths.
- Identical selected indices and history state for those requests.
- 34 evaluated score/count-vector pairs with maximum absolute difference 0.0.

See the [full report](../evidence/release_parity.json). These are finite-fixture
checks, not a mathematical proof of equivalence for all observations. The local
runtime did not have `cg` installed; official card metadata and full games remain
untested. Known schema/fallback behavior is intentionally not silently fixed in
this preservation release.

## Training references are separate

The [training reference directory](../training_reference/README.md) contains the
supplied mature architecture, auxiliary-label builder, warm-start BC trainer,
updated dataset launcher, and scratch trainer. It is not imported by the
inference package. Original internal version labels are retained there so that
source references remain understandable.

No missing upstream module has been replaced with invented training code. The
exact checkpoint/export connection remains unverified: the deployed embedding
has 204 deck rows, while the supplied training metadata lists 181 deck IDs.
This does not prevent loading the preserved inference weights, but it prevents
claiming that the included reference scripts reproduce those exact weights.

This package is also not a newly validated Kaggle submission ZIP. See
[reproduction instructions](REPRODUCIBILITY.md) and [publication notice](../NOTICE.md).
