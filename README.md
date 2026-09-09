# The Pokémon Company - PTCG AI Battle Challenge Simulation

**Team GALEX · 264 / 6,807 teams · Top 3.9% · Kaggle Silver Medal**

A Pokémon Trading Card Game agent that learns from recorded decisions through
**behavior cloning (imitation learning)**. It ranks the options offered by the
simulator using a compact attention-based policy, then selects the required
number of actions. This repository publishes readable NumPy inference code,
the preserved submission's weights, and selected training references.

[Competition](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
· [Leaderboard](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard)
· [Certificate](evidence/galex_certificate.png)
· [中文说明](START_HERE_ZH.md)

## Run a decision

With **Python 3.12**, open a terminal in the repository root:

```bash
python -m pip install -r requirements.txt
python -m examples.decision_demo
```

The demo loads the actual weights, scores three synthetic options, and prints
the selected index. No GPU, API key, or LLM service is required.

This is a **synthetic interface demo, not a simulator match**. Without the
competition's `cg` SDK, card metadata uses the original code's fallback tables.
Full game evaluation and end-to-end training are not reproduced here.
See [setup and reproduction scope](docs/REPRODUCIBILITY.md).

## How the player works

At each request, the environment supplies the player's observation, candidate
options, and selection-count limits. One turn can contain many requests; the
policy reads the updated state and decides again each time.

1. **Represent the situation.** Combine cards, board state, public history,
   resource estimates, and tactical features such as damage, energy, and evolution.
2. **Compare available options.** A base policy scores each option. State
   self-attention, action-to-state cross-attention, and attention among options
   produce a gated correction to those scores.
3. **Choose options and a count.** A separate count head handles variable-size
   selections; the controller returns the highest-ranked permitted indices.

Training fits demonstrated choices over each request's candidate set. Three
training-only auxiliary heads supervise later action categories within a turn;
they are removed from inference. The final approach is **BC, not PPO self-play**.
See [model, data selection, and training history](docs/METHOD.md).

## Read the code

| Start here | What it contains |
| --- | --- |
| [Public agent](ptcg_agent/agent.py) | One-game `Agent` and the `agent(observation, configuration)` callback |
| [Feature pipeline](ptcg_agent/features.py) | State/action encoding, public history, resources, and tactics |
| [NumPy policy](ptcg_agent/model.py) | Embeddings, attention, gated residual scoring, and count head |
| [Decision controller](ptcg_agent/controller.py) | State updates, count constraints, and option selection |
| [Weights and layout](ptcg_agent/assets) | Unchanged float32 parameter buffer and tensor metadata |
| [Training references](training_reference/README.md) | Auxiliary-target construction and BC/scratch-training scripts; incomplete dependencies |

Internal labels such as V67 and V76 identify development iterations, not
different learning algorithms. They are retained in training-reference paths
for traceability; they are not needed to run the player.

## Results and checks

| Measure | Result | Evidence |
| --- | --- | --- |
| Team placement | 264 / 6,807 · top 3.9% · silver | [GALEX certificate](evidence/galex_certificate.png) |
| Final leaderboard score | 944.2 rating points | [Archived submission page](evidence/submission_scores.png) |
| Stored model size | 1,402,487 float32 values · 133 tensors | [Artifact manifest](evidence/submission_manifest.json) |
| Original-to-module comparison | 37 synthetic requests; identical selections and history | [Parity report](evidence/release_parity.json) |
| Test suites | 24 toolkit tests + 13 released-model tests | [Validation scope](docs/VALIDATION.md) |

The stored count includes unused value-head tensors. In the recorded local
comparison, all 133 weight tensors and 34 evaluated score/count vector pairs
matched exactly. These are preservation and interface checks, not win-rate
measurements or proof of full simulator compatibility.

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests_model -v
```

## Development and contribution

Maintained by **Yuzhen Guan (Alex)** as a personal portfolio of the GALEX team entry.
My development workflow combined public baseline ideas and supplied reference
implementations with AI-assisted iteration. Contributions included replay/data
preparation, experiment comparison, and submission integration/debugging.
I used **GPT and Grok** to discuss and refine state/action representations,
tactical features, and model-component integration. These tools assisted
development; the deployed agent does not call an LLM.

The post-competition modular release, documentation, and verification tests
were prepared with Codex assistance. They make the existing system inspectable;
they are not claimed as the cause of its competition result.

Source is published with permission reported by the participant; this is not a
blanket open-source relicensing. See [NOTICE](NOTICE.md),
[provenance](docs/PROVENANCE.md), and [release transformations](docs/RELEASE.md).
This is an independent participant project, not an official Pokémon or Kaggle product.
