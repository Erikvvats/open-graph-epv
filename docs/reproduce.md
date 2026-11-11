# Reproduce the experiment

Run commands from the standalone repository root. Python 3.12 is the tested
portfolio environment; `requirements.txt` pins the observed package versions.
The short offline walkthrough needs only the included example. Full training
requires several GB for public tracking and event data, and more compute.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
graph-epv xg-download --out data/statsbomb/shots.csv
graph-epv xg-train --shots data/statsbomb/shots.csv --out outputs/xg_v1
graph-epv download --matches all --tracking
python -m open_graph_epv.contextual_xg prepare --out outputs/contextual_xg_data_v1
python -m open_graph_epv.contextual_xg select --out outputs/contextual_xg_v1
python -m open_graph_epv.contextual_xg evaluate --out outputs/contextual_xg_audit_v1
python -m open_graph_epv.contextual_xg transfer --out outputs/skillcorner_contextual_xg_v1
contextual-epv prepare --out outputs/contextual_epv_graphs_v1
team-epv prepare --out outputs/team_epv_data_v1
python examples/run_team_epv.py --legacy ""
python examples/report_team_epv.py
```

`--legacy ""` skips the optional comparison with historical signed-value weights.
Output directories must be new; do not overwrite a frozen experiment. Existing
verified downloads are reused. The code pins both upstream revisions, records
input checksums and freezes model selection before evaluating test predictions.
Floating-point results may vary across hardware/library versions. A new run is
not expected to reproduce artifact hashes byte for byte.

To score raw tracking after training:

```bash
python examples/score_team_epv.py --match 2010085 --frame 100 --family gnn
```

Or use the API:

```python
from open_graph_epv.team_epv import load_bundle
model = load_bundle('outputs/team_epv_v1/bundle')  # validation winner
result = model.score_frame(current_frame, previous_frame, match_metadata)
print(result['home_team_epv'], result['away_team_epv'])
```

A frame needs known possession, an observed ball and sufficient observed players.
Missing/ineligible inputs are not zero-valued tactical states. Read the loader
and its tests before integrating it in another application.

The shipped results are frozen reports from the existing experiment. The
portfolio notebook does **not** download data, fit the xG teacher or retrain the
nine benchmark networks. It illustrates graph construction, target assignment,
optimization and result interpretation on a single declared training example.

`examples/build_portfolio_assets.py` is a maintainer export for the original
local experiment. It verifies source hashes and requires its frozen audits and
historical diagnostic artifacts. It is not an additional prerequisite for using
this repository or reproducing the benchmark.

To create a separate publication archive:

```bash
python examples/package_portfolio.py --out outputs/open-graph-epv-portfolio.zip
```

Upload the extracted `open-graph-epv` folder as a separate repository root. The
parent research workspace is outside this package. No automated command here
creates a repository or publishes anything.
