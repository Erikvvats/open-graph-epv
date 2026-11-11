# Publication status — 11 November 2026

This is a bounded research portfolio project. It is not a validated operational
football model. The README leads with the small baseline gains and their limits.

| Requirement | Status |
|---|---|
| Question, data, method and baseline result | English README |
| Honest limitations | README, methods and error-concentration figure |
| 2–3 figures | Three figures from the frozen two-team experiment |
| Runnable example | Notebook with executed text outputs; plain-Python runner |
| Dependency specification | Pinned requirements, tested in Python 3.12.7 |
| Reviewable code | Source, tests, explanation guide and AI-assistance statement |
| Public-data boundary | Standalone allowlist; public demo provenance; no model weights |
| Attribution | SkillCorner MIT notice; StatsBomb logo and separate agreement |
| Automated tests | 68 tests pass locally and on GitHub Actions |
| Browser demo | 51 case/model/frame combinations, playback, targets and mobile checked |
| External publication | [Public GitHub repository](https://github.com/Erikvvats/open-graph-epv), published 11 November 2026 |

`reports/public_provenance.json` records hashes of public SkillCorner source files
used for the demo, the frozen experiment and generated figures. The teacher's
lineage is recorded in `reports/portfolio_lineage.json`: the contextual teacher
links to a pinned public StatsBomb shot corpus. Its seasons cover the English
Premier League, available Bundesliga matches, men's World Cups and men's Euros.
The standalone implementation does not import the private parent project.

The source allowlist omits `data/`, `outputs/`, model checkpoints, caches,
credentials and parent-workspace files. It deliberately includes small public
SkillCorner demonstration extracts and reports derived from the open-data
experiment. There are no Barça/private-data artifacts in this publication set.
The tests check packaging boundaries and reject symbolic links that could bring
in outside files. File hashes establish traceability, not statistical validity.

## Checks and their limits

The notebook is executed sequentially as ordinary Python, including its
single-training-example optimization exercise. It has not been run through a
Jupyter kernel here. The demo is tested in headless Chrome and visually inspected
at desktop and mobile sizes. Full training and downloads were not repeated for
this presentation update; frozen outputs and their provenance were verified.

The local checks on 10 November reused installed dependencies and tested an
isolated copy of the publication files. On 11 November, a fresh GitHub Actions
Ubuntu runner installed the pinned requirements and passed all 68 tests plus the
five-cell Python walkthrough. [Successful CI run](https://github.com/Erikvvats/open-graph-epv/actions/runs/43032347080)
checks source commit `d34000ab27bee1a0d6632c285698314900bce620`. See
`reports/portfolio_verification.json` for the recorded checks. This does not
retrain the benchmark or execute a Jupyter kernel.

## What remains for the author

Read and explain the notebook and core code in your own words, using
`docs/explain_the_code.md`. Automated checks cannot certify this requirement.
Verify that the project description accurately reflects your contribution.

The standalone folder is now published at
[Erikvvats/open-graph-epv](https://github.com/Erikvvats/open-graph-epv). The README,
notebook and CI result are available there. Future updates should retain this
standalone boundary; do not upload the parent workspace. The included browser
demo works after cloning/downloading; GitHub does not execute an HTML file when
browsing repository source.

For an application, describe the experiment and what you learned from its
limitations. Do not claim validated tactical intelligence or superiority over
SkillCorner. Personal understanding of the implementation remains essential
before discussing the project in an interview.
