# Reproducibility Guide

## Modes

This artifact supports two reproduction modes.

1. Paper-output reproduction from cached summaries:

   ```bash
   python run_make_tables.py
   python run_make_figure.py
   python scripts/smoke_test.py
   ```

   This checks the paper-facing tables, figure points, caption values, and
   anonymization scan without requiring raw market data.

2. Full experiment rerun from public raw data:

   ```bash
   python run_reproduce.py --data_dir data/raw --out_dir outputs/full --seeds 7,11,13,17,19,23,29,31,37,41 --run_neural
   ```

   This rebuilds experiment outputs from the public data files described in
   `data/README.md`. Runtime depends on hardware because neural variants are
   trained for ten seeds.

## Main Paper Values

The manuscript tables use:

- `paper_values/table1_values.csv`
- `paper_values/table2_values.csv`
- `paper_values/caption_numbers.json`

The cached source summaries used to create them are kept in `results/` with
their original stable method identifiers.

`docs/MANUSCRIPT_ALIGNMENT.md` maps the paper's main tables, figure, and effort
checks to the corresponding artifact files.

## Random Seeds

The ten seeds are listed in `config/seeds.txt`.
