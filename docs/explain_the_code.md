# Explain the project before presenting it

Start with the notebook. Then trace a single frame through these files. Read the
functions themselves, including their checks; this guide is not a substitute for
understanding each line you present in an interview.

| Step | Code | What you should explain |
|---|---|---|
| Fetch public inputs | `src/open_graph_epv/source.py`, `xg.py` | Pinned revisions, original licenses, file hashes |
| Estimate shot quality | `contextual_xg.py`, `xg_context_features.py` | Goal labels, context features, calibration, fallback |
| Build one graph | `data.py:frame_graph` | Coordinate orientation, velocity validity, masking |
| Build a future label | `data.py:future_target` | Strict future interval, period end, first shot |
| Assign two teams | `team_epv.py:relative_targets`, `to_home_away` | Label decomposition versus prediction; identity mapping |
| Predict | `team_epv_model.py:TeamGraphEPV.forward` | Tensor shapes, all valid edges, mean aggregation, pooling |
| Train/select/evaluate | `team_epv.py` | Two-component MSE, match split, seed ensemble, validation |
| Inspect results | `reports/portfolio_results.json`, `demo/viewer.js` | Recorded predictions versus realized labels |

For every tensor operation in `forward`, write down its input/output dimensions.
For example, `x` is B×23×10; a relative-position tensor is B×23×23×2; two output
values are B×2. Explain how the mask prevents absent players contributing to sums,
and why dividing by valid counts matters. The residual update, node encoder and
head are defined in `model.py`; inheritance does not remove the need to read them.

Questions to answer in your own words:

1. Why is MSE appropriate for the conditional mean, while Huber changes the target?
2. Why can both team predictions be positive when only one observed label is?
3. Why is the unused softmax mass not a no-shot probability?
4. Which information is available at inference, and which is used only for labels?
5. Why split by match rather than randomly by frame? What dependence remains?
6. What does the ball-only baseline establish? What does disabling messages test?
7. Why does a good StatsBomb xG score not establish SkillCorner calibration?
8. Why was the set model selected when the GNN has the lowest test RMSE?
9. What does the 48.6% error concentration reveal, and what does it not prove?
10. What evidence would make you decide not to use this model operationally?

A defensible project description:

> I investigated whether tracking context improves a short-horizon xG value proxy.
> The gains over simple controls were small. Label transfer and limited independent
> matches constrain the conclusions. I used AI assistance in development and can
> demonstrate the data flow, assumptions, tests and failure analysis.

Only use the last sentence when it is true for you. The author still needs to
review the implementation personally. Do not claim independent implementation,
production readiness, causal recommendations or validated goal probabilities.
The historical code remains for reproducibility; the portfolio story is the
bounded two-team experiment, not every exploratory model in the repository.
