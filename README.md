# The Pokémon Company - PTCG AI Battle Challenge Simulation

**Team GALEX · 264 / 6,807 teams · Top 3.9% · Kaggle Silver Medal**

A portfolio case study and verification toolkit for a Pokémon Trading Card Game
agent trained through **behavior cloning (imitation learning)**.

[Competition](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
· [Leaderboard](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard)
· [Result certificate](evidence/galex_certificate.png)
· [中文说明](START_HERE_ZH.md)

## What does the agent do?

At each decision, the game supplies an observation and a set of legal options.
The agent scores those options and returns their indices. Some decisions require
one option; others permit several, so the agent must decide both **what to select**
and **how many to select**. It returns a 60-card deck at the initial deck callback.

This is a decision-making model, not a chatbot. The final submission was based on
the mentor-provided V76 behavior-cloning approach; **PPO was not used in the final
model**, according to the supplied training clarification.

## Technical highlights

- Encodes observable board state, the player's visible cards, public action
  history, resource estimates, and candidate actions.
- Uses two state self-attention blocks, action-to-state cross-attention, and
  candidate-action self-attention to refine a base action score.
- Combines base scores with a gated residual and a separate selection-count head.
- Packages the submitted inference callback in `main.py` with embedded float32
  weights, alongside `deck.csv`; it uses NumPy rather than PyTorch for inference.
  Official card metadata remains important for faithful game behavior.

See [method and implementation notes](docs/METHOD.md) for the details and known
limitations. These describe the supplied team solution, not a claim that all
model components were independently authored by this repository's maintainer.

## Results and verification

| Item | Recorded result | Evidence / scope |
| --- | --- | --- |
| Competition placement | 264 / 6,807; top 3.9%; silver | Participant-provided Kaggle certificate |
| Final leaderboard score | 944.2 | Participant-provided leaderboard/submission screenshot; not a win percentage |
| Stored model parameters | 1,402,487 float32 values in 133 tensors | [Static manifest](evidence/submission_manifest.json); includes unused value-head tensors |
| Public toolkit tests | 24 passed locally | Standard-library tests using fabricated artifacts |
| Original submission smoke checks | 12 passed locally | [Local report](evidence/local_smoke_report.json); synthetic interface checks, not matches |

Top percentage is `264 / 6807 × 100 = 3.88%`, rounded to 3.9%.
No claim of an independently reproduced tournament score, training run, or win
rate is made. See [evaluation scope](docs/VALIDATION.md).

## Run the public tools

With Python 3.12, from this repository's root:

```bash
python -m unittest discover -s tests -v
```

No third-party packages or competition files are needed for these tests.

If you already have an authorized local copy of `submission2.zip`, verify its
contents **without executing the submitted code**:

```bash
python -m tools.inspect_submission /path/to/submission2.zip --expected evidence/submission_manifest.json
```

For the optional trusted-code smoke check, see
[reproduction instructions](docs/REPRODUCIBILITY.md). The inspector is an integrity
check, not a malware scanner or proof of authorship.

## Release status

This repository currently publishes **the case study, result evidence, inspection
tools, and synthetic tests**. The original agent source, weights, replay data, and
mentor reports are **not included**, pending redistribution permission and
licensing review. A fresh clone therefore does not run the trained agent by itself.

The supplied V76 training bundle also has missing dependencies; full training
reproduction is not currently available. See
[provenance and release checklist](docs/PROVENANCE.md).

The verification utilities and documentation were prepared after the competition
with AI assistance. They did not produce the competition result. This is an
independent participant portfolio, not an official Pokémon or Kaggle project.
