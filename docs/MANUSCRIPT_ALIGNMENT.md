# Manuscript Alignment

This file maps the paper's main empirical claims to artifact files.

## Experimental Protocol

- Ten seeds: `config/seeds.txt`
- Lookback, horizon, bootstrap paths, graph propagation, neural training, and
  objective weights: `config/experiment_config.yaml`
- Raw-data requirements: `data/README.md`

## Table 1

Manuscript table: explicit graph-risk penalty versus feature-only graph-score
input.

Artifact files:

- `results/placement_table.csv`
- `paper_values/table1_values.csv`

These files contain the six reported deltas for CVaR95 loss, crowding, and
co-crash risk in the Industry and ETF universes.

## Table 2

Manuscript table: diagnostic decomposition of placement and graph-score quality.

Artifact files:

- `results/diagnostic_table.csv`
- `paper_values/table2_values.csv`

The quality deltas are computed as Graph-penalty contract minus
Randomized-score contract using the same seed-clustered episode-level evaluation
as Table 1.

## Figure 1

Manuscript figure: crowding by CVaR95 risk frontier on the Industry and ETF
test sets.

Artifact files:

- `results/figure1_points.csv`
- `paper_values/figure1_points.csv`
- `figures/fig1_risk_frontier_core.pdf`

The caption numbers are stored in `paper_values/caption_numbers.json`.

## Node-Exposure Mechanism Figure

Mechanism figure: change in induced exposure by graph-risk score quintile.

Artifact files:

- `results/mechanism_node_exposure_data.csv.gz`
- `results/mechanism_exposure_bins.csv`
- `paper_values/mechanism_exposure_bins.csv`
- `figures/fig2_node_exposure_mechanism.pdf`

The deltas are Graph-penalty contract minus Feature-only contract. The
supplementary penalty-channel version is stored in
`figures/figS_penalty_channel_mechanism.pdf`.

## Effort Invariance

The effort checks used in the text are stored in `results/effort_checks.csv`.
