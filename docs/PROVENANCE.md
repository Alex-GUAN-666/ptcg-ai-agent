# Sources, credits, and release status

## Attribution

This personal portfolio is maintained by **Yuzhen Guan (Alex)** and documents
**GALEX's** entry in **The Pokémon Company - PTCG AI Battle Challenge Simulation**.
The placement is a team result, not a claim of sole authorship of every component.

The participant describes a workflow combining public baseline ideas, supplied
reference implementations, and GPT/Grok-assisted discussion of state/action
representations, tactical features, and component integration. Reported
contributions include data preparation, experiment comparison, and submission
integration/debugging. This is a participant account, not an independent audit
of individual contributions.

The post-competition modular release, inspection utilities, tests, and
documentation were prepared with Codex assistance. They are distinct from
competition-time work and did not cause the leaderboard result.

## Evidence and method sources

| Source | Used for |
| --- | --- |
| [GALEX certificate](../evidence/galex_certificate.png) and [submission screenshot](../evidence/submission_scores.png) | Team placement of 264 / 6,807, silver medal, and 944.2 final rating |
| Preserved `submission2.zip` | Deployed inference calculations, float32 weights, deck, and [file hashes](../evidence/submission_manifest.json) |
| Supplied V67/V76 model, target-builder, and trainer files | Auxiliary supervision, warm-start versus scratch training, and script defaults |
| Supplied V76 dataset report | Aggregate prepared-data counts, not a regenerated dataset |
| Participant-supplied technical reports and session transcript | Development history, data-selection rationale, and reported experimental observations |

The reference report also discusses another entry placed 191st. That placement
is not GALEX's result. Its reported accuracy and matchup measurements are not
claimed as independently reproduced GALEX experiments.

Source code takes precedence over approximate spoken explanations for exact
configuration details. Reference-reported measurements are labeled separately
from measured artifact properties and the team's archived result.
The transcript's recording date was not supplied. Public notes summarize its
technical content without reproducing the transcript or identifying attendees.

## Publication permission and included material

On **2026-09-09**, the participant confirmed that the source/artifact provider
permitted public publication. This is the participant's reported confirmation;
no claim of exclusive ownership or broad relicensing follows from it.
See [NOTICE](../NOTICE.md).

Included:

- Readable inference source derived from the preserved final submission.
- The original float32 weight buffer and deck bytes, with integrity manifests.
- Five selected training-reference Python files and an aggregate dataset summary.
- Original portfolio utilities, synthetic fixtures, tests, and result images.

Not included:

- Raw replay collections, participant/team-level training records, course PDFs,
  the full transcript, local credentials, or unrelated personal files.
- Vendored simulator/SDK packages, card artwork, or external library sources.
- Historical PPO scripts that were not the final training route.
- A claim of complete dataset-to-checkpoint or official-game reproduction.

## Traceability and remaining gaps

The inspected source did not contain explicit author, copyright, or license
headers. The release does not assign a new repository-wide open-source license;
any applicable upstream notices or terms must still be respected.

[release_manifest.json](../evidence/release_manifest.json) records modular
source hashes and the unchanged parameter buffer.
[training_reference/source_manifest.json](../training_reference/source_manifest.json)
records the selected scripts' original/published hashes and limited edits.

Missing upstream modules, the exact checkpoint/export lineage, and compatible
simulator setup remain unresolved. The publication confirmation does not resolve
those technical gaps or license separately obtained competition resources.
See [reproduction status](REPRODUCIBILITY.md).
