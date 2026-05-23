# Data Inputs

Raw market data are not committed to this anonymous review artifact.

The full experiment expects the following files in `data/raw/`:

- `49_Industry_Portfolios_Daily.csv` or a zip containing that file from the
  Kenneth R. French Data Library.
- `F-F_Research_Data_Factors_daily.csv` from the Kenneth R. French Data Library.
- Optional `F-F_Research_Data_5_Factors_2x3_daily.csv`.
- Optional `F-F_Momentum_Factor_daily.csv`.
- Optional VIX/VXV files named with `VIXCLS` and `VXVCLS`.
- Optional ETF wide price table matching `master_daily_features_macro_dailyonly`
  or `master_daily_features`. The expected ETF columns are:
  `EEM`, `GLD`, `HYG`, `IEF`, `IWM`, `LQD`, `QQQ`, `SPY`, `TLT`, `UUP`,
  `XLB`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLRE`, `XLU`, `XLV`, and `XLY`.

The paper-facing cached aggregate summaries in `results/` allow table and figure
construction to be checked without redistributing raw data. Raw data should be
obtained from the original public providers and remain subject to those
providers' terms.
