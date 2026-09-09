# Sources, credits, and release status

## Attribution

This is a portfolio for **team GALEX's** entry in
**The Pokémon Company - PTCG AI Battle Challenge Simulation**.

The repository is maintained by **Yuzhen Guan (Alex)**. The core agent and V76
training approach were supplied by competition mentor **kaggle竞赛圈**, who also
provided the technical retrospective. Credit for the competition placement is
at the GALEX team level.

The post-competition inspection utilities, tests, and explanatory documentation
were prepared with Codex assistance. They are distinct from the submitted
competition system and did not cause its leaderboard performance.

## Basis for the case study

| Source | Used for |
| --- | --- |
| Participant-provided certificate and leaderboard/submission screenshots | GALEX's 264 / 6,807 placement, silver medal, and 944.2 final rating |
| Preserved `submission2.zip` | Inference behavior, parameter layout, and file hashes recorded in the [manifest](../evidence/submission_manifest.json) |
| Supplied V67/V76 model, target-builder, and trainer files | Auxiliary targets, warm-start versus scratch training, export behavior, and script defaults |
| Supplied V76 dataset report | Counts of scanned episodes, prepared decisions, and option rows |
| Participant-supplied mentor meeting transcript | Data-selection rationale, version history, and reported experimental observations |

The transcript's recording date was not supplied. Relevant segments are
11:13–12:39 (demonstrations), 20:41–22:06 (replay selection and temporal effects),
23:01–25:40 (model evolution), and 26:39–29:29 (training and per-decision inference).
The public [method notes](METHOD.md) summarize these sections rather than
reproducing the meeting or identifying other attendees.

Source code is used for exact configuration details where the spoken account is
approximate. Meeting-reported accuracy is labeled separately from measured
artifact properties and the team's competition result.

## Public contents

- Original portfolio documentation and verification utilities.
- Synthetic test inputs and generated inspection summaries; no weight values.
- The participant-provided result certificate.

Original agent source, weights, raw replay data, mentor reports, and the full
meeting transcript are not distributed. Source/weight release permission is
still being confirmed. No repository-wide open-source license has been applied;
third-party rights and attribution must be resolved before extending the release.

## Before releasing the original model

- [ ] Confirm permission to publish the original source and weights, with the
  relevant mentor/team rights holders and their requested attribution.
- [ ] Check competition rules and licenses for incorporated SDK/data/code.
- [ ] Identify the final checkpoint and its export relationship to the preserved
  submission; version subsequent changes separately.
- [ ] Preserve original author/license notices and check for credentials,
  private paths, and unrelated personal files.
- [ ] Verify installation and official-simulator behavior from a clean environment.
