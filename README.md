# Graph-Risk Signals in Portfolio-Delegation Contracts

This anonymous artifact supports the CIKM short-paper submission
"Placement or Quality? Diagnosing Graph-Risk Signals in Portfolio-Delegation Contracts."

The package contains:

- the experiment engine for graph-risk placement diagnostics;
- scripts that reproduce the paper-facing tables and risk-frontier figure from cached summary outputs;
- documentation for obtaining the public input data needed for a full rerun;
- documentation on data provenance, anonymization, and artifact scope.

## Quick Start

Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

Rebuild paper tables and the figure from cached result summaries:

```bash
python run_make_tables.py
python run_make_figure.py
python scripts/smoke_test.py
```

The generated paper-facing outputs are:

- `results/placement_table.csv`
- `results/diagnostic_table.csv`
- `results/effort_checks.csv`
- `figures/fig1_risk_frontier_core.pdf`
- `paper_values/table1_values.csv`
- `paper_values/table2_values.csv`
- `paper_values/caption_numbers.json`

## Full Experiment Rerun

The raw market data are not redistributed in this review artifact. They are public
or publicly obtainable, but redistribution terms differ by provider. See
`data/README.md` for file names and sources.

After placing the raw files in `data/raw/`, a compact run can be started with:

```bash
python run_reproduce.py --data_dir data/raw --out_dir outputs/full --seeds 7,11,13,17,19,23,29,31,37,41 --run_neural
```

For a smoke-scale execution:

```bash
python run_reproduce.py --data_dir data/raw --out_dir outputs/smoke --quick --seeds 7 --run_neural --epochs 5 --skip_etf
```

The full multi-seed neural run is computationally heavier than the table and
figure rebuild. It regenerates the cached summaries from raw public inputs.

## Repository Layout

```text
config/       Experiment parameters and seed list.
src/          Experiment engine and paper-facing helper modules.
scripts/      Ordered entry points for data preparation, experiments, tables, figures, and checks.
data/         Raw-data instructions; raw files are intentionally not committed.
results/      Cached summary CSVs and generated paper-facing result tables.
figures/      Paper figure output.
paper_values/ Compact values used directly in the manuscript.
docs/         Reproducibility, anonymization, and artifact-limit notes.
```

## Data Redistribution Note

The Fama-French files and ETF price panels used here are public or publicly
obtainable inputs. This repository does not redistribute raw market data. It
includes cached aggregate result summaries and scripts for verifying table
construction and rerunning the pipeline after obtaining the raw inputs.

## License

Code in this artifact is released under the MIT License. The raw input data are
not included and remain subject to their original providers' terms.
