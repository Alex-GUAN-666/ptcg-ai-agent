# Reproduction: what is runnable today?

## 1. Public toolkit, no competition files

Use Python 3.12, open a terminal in the repository root, and run:

```bash
python -m unittest discover -s tests -v
```

The public tests need only the Python standard library. They manufacture small
fake archives to test checksums, tensor layouts, ZIP-member restrictions, size
limits, and the selection-contract validator. They do not load model weights.

## 2. Inspect an authorized local final submission

Keep your existing archive outside the repository, or under `local_artifacts/`.
That directory is ignored by Git. Do not upload the file through the browser:
ignore rules are not a substitute for checking what you publish.

```bash
python -m tools.inspect_submission /path/to/submission2.zip --expected evidence/submission_manifest.json
```

On Windows, replace `/path/to/submission2.zip` with your own quoted path, e.g.
`"C:\Users\YourName\Downloads\submission2.zip"`. The `--expected` comparison
requires the recorded `main.py`, `deck.csv`, and embedded-weight hashes. Changing
ZIP compression alone need not change those member hashes.

The command parses literal metadata with `ast`, validates the embedded float32
buffer, and prints JSON. It does not import the submitted Python module, extract
arbitrary archive paths, or perform network requests. It supports this specific
two-file format, not every possible Kaggle submission.

## 3. Optional trusted-code smoke check

Only if you trust the original submission source:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell,
or `source .venv/bin/activate` in macOS/Linux. Then:

```bash
python -m pip install -r requirements-smoke.txt
python -m tools.smoke_submission /path/to/submission2.zip --trusted
```

Unlike the inspector, this **executes the original code** in your Python process.
It is not a security sandbox. The smoke tool requires a match to the recorded
artifact before importing it. Its temporary files are removed afterwards.

The local check used Python 3.12.14 and NumPy 2.3.5. It tests the deck callback,
selection counts and index bounds on artificial observations, repeat/reset
paths, and finite score/count outputs. See the
[recorded local report](../evidence/local_smoke_report.json).

No GPU is required for these checks. No simulator match, original training run,
or leaderboard reproduction is performed.

## 4. What is missing for full reproduction?

The supplied V76 scripts reference earlier `v7`, `v14`, `v20`, `v40`, `v60`, and
`v67` modules. The available V67/V76 bundles do not supply the complete dependency
chain, original simulator installation, structured feature dataset, or exact
training checkpoint/export metadata. Several launch scripts also contain
machine-specific Windows paths.

These dependencies must be collected, their release rights confirmed, paths
parameterized, and a clean-environment training/export run verified before a
“train from scratch” command is advertised. There is deliberately no invented
one-command training recipe in this repository.
