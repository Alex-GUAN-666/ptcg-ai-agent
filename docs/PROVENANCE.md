# Provenance and release status

## Attribution

This is a portfolio for **team GALEX's** entry in
**The Pokémon Company - PTCG AI Battle Challenge Simulation**.

The core solution and V76 training materials were supplied by a mentor. Credit
for the competition result is at the GALEX team level; the method and training
materials are attributed to the mentor. This repository does not assign sole
authorship of the model to its maintainer. Individual implementation and training
roles will be documented separately after confirmation.

The post-competition inspection utilities, tests, and explanatory documentation
were prepared with Codex assistance. They are distinct from the submitted
competition system and did not cause its leaderboard performance.

The supplied method report references
[Imitation learning for deck 312](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/writeups/imitation-learning-for-deck-312).
That is a method reference, not verification of GALEX's rank or a grant of
redistribution rights. The linked page's contents were not independently
retrieved during this audit.

## Public contents

- Original portfolio documentation and verification utilities.
- Synthetic test inputs and generated inspection summaries; no weight values.
- The participant-provided result certificate.

The repository does **not** distribute the mentor's Python implementation,
model weights, full reports, conversations, replay dataset, game SDK, or card art.
It has no blanket open-source license pending a deliberate licensing decision.
No license or trademark rights over third-party material are asserted.

## Before releasing the original model

- [ ] Obtain confirmation from the relevant mentor/team rights holders for
  publishing source code and model weights; record the requested attribution.
- [ ] Check competition rules and licenses for incorporated SDK/data/code.
- [ ] Identify the actual final checkpoint and its export relationship to the
  preserved submission; keep later changes separately versioned.
- [ ] Preserve original author/license notices and scan for credentials,
  private paths, and unrelated personal files.
- [ ] Verify installation and official-simulator behavior from a clean environment.
- [ ] Document each participant's actual role before adding personal authorship
  or leadership claims to the portfolio or CV.

Being publicly viewable does not make a third-party work freely reusable.
Changing comments or variable names does not establish independent authorship.
