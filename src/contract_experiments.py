#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Graph-risk placement experiments for portfolio-delegation contracts
==================================================================

This module contains the experimental engine used for:

    Placement or Quality? Diagnosing Graph-Risk Signals in
    Portfolio-Delegation Contracts

Main modules
------------
1. Data loading:
   - Fama-French 49 Industry daily returns
   - Fama-French 3/5 factors and Momentum
   - VIX/VXV parquet files
   - ETF wide price table for robustness

2. State construction:
   - rolling tail co-exceedance graph
   - rolling correlation graph
   - GLS/covariance-optimal RPE weights
   - marginal graph systemic-risk score
   - node/global features

3. Contract environments:
   - LQ-Gaussian hidden effort/systematic-exposure model
   - closed-form IC best response
   - IR adjustment by base payment

4. Baselines and learned contracts:
   - Fixed salary
   - Linear PnL contract
   - Equal-weight RPE
   - GLS-RPE
   - Graph-RPE
   - Equal-RPE + systemic penalty
   - Graph-RPE + centrality penalty
   - Full fixed graph contract
   - Node-only MLP contract, optional
   - graph-penalty contract: graph-score input with an explicit graph-risk penalty
   - feature-only contract: graph-score input without the explicit graph-risk penalty
   - randomized-score contract: architecture-matched randomized graph-risk scores
   - Full GNN graph contract, optional ablation
   - z_min sensitivity robustness for the hard-effort lower bound
   - constrained-training vs post-hoc projection ablation

5. Outputs:
   - CSV result tables
   - figures
   - run summary

Example:
    python run_reproduce.py --data_dir data/raw --out_dir outputs/full
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
# Keep CPU linear algebra stable and avoid PyTorch/OpenMP thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import random
import re
import sys
import time
import traceback
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib
    matplotlib.use("Agg")  # stable headless/batch plotting on Windows and servers
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
except Exception:  # pragma: no cover
    torch = None
    class _TorchUnavailableModule:
        pass

    class _TorchUnavailableLayer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available; neural training cannot be used.")

    class _TorchUnavailableNN:
        Module = _TorchUnavailableModule
        Linear = _TorchUnavailableLayer
        ReLU = _TorchUnavailableLayer

    nn = _TorchUnavailableNN()
    TORCH_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Rolling state construction
    window: int = 252
    episode_horizon: int = 20
    step: int = 20
    tail_q: float = 0.05
    graph_eps: float = 1e-8
    ridge: float = 1e-5

    # Main sample splits for industry experiments
    industry_start: str = "1990-01-01"
    train_end: str = "2005-12-31"
    val_start: str = "2006-01-01"
    val_end: str = "2007-12-31"
    test_start: str = "2008-01-01"
    test_end: str = "2026-03-31"

    # ETF sample split; automatically truncated by available data
    etf_start: str = "2015-10-08"
    etf_train_end: str = "2018-12-31"
    etf_val_start: str = "2019-01-01"
    etf_val_end: str = "2019-12-31"
    etf_test_start: str = "2020-01-01"
    etf_test_end: str = "2026-03-16"

    # Agent/environment parameters. Units are episode-return units.
    alpha0: float = 0.050       # productive effort loading
    pi0: float = 0.004          # systematic-exposure risk premium per episode
    pi_beta_mult: float = 0.001 # beta-heterogeneity in systematic premium
    k_cost: float = 1.0         # effort cost curvature
    h_cost: float = 1.0         # systematic exposure deviation cost curvature
    gamma_agent: float = 3.0    # agent risk aversion
    reservation_utility: float = 0.0
    s_max: float = 3.0
    e_max: float = 2.0

    # Contract caps for neural policies
    theta_cap: float = 0.70
    eta_cap: float = 0.70
    lambda_cap: float = 0.10
    # Strong graph-aware version: force neural contracts to maintain the same
    # effort incentive intensity as RPE baselines. Since e*=alpha*(theta+eta)
    # and alpha0=0.05, z_min=0.80 implies mean effort about 0.04.
    hard_effort_z_min: float = 0.80
    hard_effort_z_cap: float = 1.20
    # The peer benchmark used in the contract. In the strong version we do not
    # use the tail graph as peer benchmark; graph information enters through
    # message passing and systemic-risk scores only.
    neural_peer_benchmark: str = "gls"

    # Principal objective
    cvar_alpha: float = 0.95
    cvar_kappa: float = 0.65
    payment_var_weight: float = 0.05
    turnover_weight: float = 0.01
    # Extra objective terms that align training/grid-search with the graph-aware
    # systemic-risk-control claim. `principal_CE` remains the pure financial CE;
    # `policy_objective` includes these two penalties.
    crowding_obj_weight: float = 0.10
    effort_target: float = 0.040
    effort_target_weight: float = 50.0

    # Simulation details
    co_crash_frac: float = 0.20
    random_seed: int = 7
    # Number of bootstrap scenario paths per decision state. This is crucial for
    # tail-risk evidence: a single realized future path makes co-crash/CVaR too noisy.
    mc_paths: int = 50
    include_realized_path: bool = True
    # Make future stress scenarios depend on the true market tail graph.
    # rho=0 disables propagation; rho in (0,1) applies (I-rho W_tail)^(-1)
    # to residual scenario shocks.
    graph_propagation_rho: float = 0.25
    effort_targets: Tuple[float, ...] = (0.030, 0.035, 0.040, 0.045)

    # Grid search coefficient candidates. A moderately fine grid is used because
    # target-effort frontiers need enough feasible contract intensities.
    theta_grid: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 0.65)
    eta_grid: Tuple[float, ...] = (0.00, 0.15, 0.30, 0.45, 0.60)
    lambda_grid: Tuple[float, ...] = (0.00, 0.02, 0.05, 0.08, 0.12)

    # Neural training
    epochs: int = 250
    lr: float = 1e-3
    hidden_dim: int = 32
    weight_decay: float = 1e-4
    patience: int = 50
    neural_exact_ic: bool = False  # If True, neural layer uses the exact batched linear-system IC solve.
    torch_threads: int = 1         # Avoid excessive CPU-thread contention on Windows/Anaconda.

    # Quick mode overrides
    quick: bool = False

    def apply_quick(self) -> None:
        if self.quick:
            self.step = 60
            self.epochs = min(self.epochs, 50)
            self.mc_paths = min(self.mc_paths, 10)
            self.theta_grid = (0.10, 0.35, 0.60)
            self.eta_grid = (0.00, 0.30, 0.60)
            self.lambda_grid = (0.00, 0.035, 0.080)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


PAPER_METHOD_NAMES: Dict[str, str] = {
    "FixedSalary": "fixed-salary contract",
    "LinearPnL": "linear output-sharing contract",
    "Linear_TailCVaRPenalty": "linear contract with tail-risk penalty",
    "EqualRPE": "equal-weight RPE contract",
    "GLS_RPE": "GLS peer-benchmark contract",
    "GraphRPE_tail_Diagnostic": "tail-graph peer-benchmark contract",
    "EqualRPE_TailCVaRPenalty": "equal-weight RPE contract with tail-risk penalty",
    "GLS_RPE_TailCVaRPenalty": "GLS benchmark with tail-risk penalty",
    "GLS_RPE_CentralityPenalty": "GLS benchmark with centrality penalty",
    "GLS_RPE_LegacyMarginalPenalty": "GLS benchmark with marginal-risk penalty",
    "FullFixed_GLSBenchmark_GraphTailCVaR": "fixed graph-risk penalty contract",
    "RawNodeMLP_GLS_NoGraph_HardEffort": "no-graph neural baseline",
    "GraphSignal-ICL-NoPenalty": "feature-only contract",
    "GraphSignal-ICL": "graph-penalty contract",
    "GraphScoreMLP_GLS_TailCVaR_HardEffort": "graph-penalty contract",
    "RandomSignal-ICL": "randomized-score contract",
    "FullGNN_TailMessage_GLSBenchmark_HardEffort": "message-passing graph learner",
    "RandomGraphGNN_GLSBenchmark_HardEffort": "random-graph message-passing learner",
    "ETF_RawNodeMLP_GLS_NoGraph_HardEffort": "ETF no-graph neural baseline",
    "ETF_GraphSignal-ICL-NoPenalty": "ETF feature-only contract",
    "ETF_GraphSignal-ICL": "ETF graph-penalty contract",
    "ETF_RandomSignal-ICL": "ETF randomized-score contract",
    "ETF_FullGNN_TailMessage_GLSBenchmark_HardEffort": "ETF message-passing graph learner",
    "ETF_RandomGraphGNN_GLSBenchmark_HardEffort": "ETF random-graph message-passing learner",
}


def paper_method_name(method: object) -> object:
    """Paper-facing method label; raw `method` remains the stable experiment id."""
    if pd.isna(method):
        return np.nan
    key = str(method)
    if key.startswith("GraphSignal-ICL_zmin_"):
        return "graph-penalty contract (effort-bound sensitivity)"
    if key == "GraphSignal-ICL_constrained":
        return "graph-penalty contract with constrained outputs"
    if key == "GraphSignal-ICL_posthoc_projection":
        return "graph-penalty contract with post-hoc projection"
    return PAPER_METHOD_NAMES.get(key, key)


def comparison_display_name(comparison: object) -> str:
    text = str(comparison)
    if " minus " not in text:
        return text
    treatment, baseline = text.split(" minus ", 1)
    return f"{paper_method_name(treatment)} minus {paper_method_name(baseline)}"


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add manuscript-facing labels without changing internal method identifiers."""
    out = df.copy()
    if "method" in out.columns:
        values = out["method"].map(paper_method_name)
        if "display_method" not in out.columns:
            insert_at = list(out.columns).index("method") + 1
            out.insert(insert_at, "display_method", values)
        else:
            out["display_method"] = out["display_method"].where(out["display_method"].notna(), values)
    if "comparison" in out.columns:
        values = out["comparison"].map(comparison_display_name)
        if "display_comparison" not in out.columns:
            insert_at = list(out.columns).index("comparison") + 1
            out.insert(insert_at, "display_comparison", values)
        else:
            out["display_comparison"] = out["display_comparison"].where(out["display_comparison"].notna(), values)
    if "random_control" in out.columns:
        values = out["random_control"].map(paper_method_name)
        if "display_random_control" not in out.columns:
            insert_at = list(out.columns).index("random_control") + 1
            out.insert(insert_at, "display_random_control", values)
        else:
            out["display_random_control"] = out["display_random_control"].where(out["display_random_control"].notna(), values)
    return out


def write_display_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    add_display_columns(df).to_csv(path, index=index)


def _version_score_for_filename(path: Path) -> Tuple[int, int, str]:
    """Prefer direct CSV/parquet files over ZIPs and latest numbered copies."""
    name = path.name.lower()
    ext_rank = 2 if path.suffix.lower() in {".csv", ".parquet"} else (1 if path.suffix.lower() == ".zip" else 0)
    nums = re.findall(r"\((\d+)\)", name)
    copy_rank = int(nums[-1]) if nums else 0
    return (ext_rank, copy_rank, name)


def find_file(data_dir: Path, patterns: Sequence[str], required: bool = True) -> Optional[Path]:
    """Find the best matching file inside --data_dir.

    This is robust when code and data live in different folders, and when the
    data folder contains duplicate uploads such as VIXCLS(13).parquet and
    VIXCLS(15).parquet. It prefers exact filename matches, direct CSV/parquet
    files over ZIPs, and the highest numbered duplicate.
    """
    files = sorted(list(data_dir.iterdir()), key=lambda f: f.name.lower()) if data_dir.exists() else []
    lower_files = [(f, f.name.lower()) for f in files]
    for pat in patterns:
        pat_lower = pat.lower()
        exact = [f for f, name in lower_files if name == pat_lower]
        if exact:
            return sorted(exact, key=_version_score_for_filename)[-1]
        substring = [f for f, name in lower_files if pat_lower in name]
        if substring:
            return sorted(substring, key=_version_score_for_filename)[-1]
        globbed = sorted(data_dir.glob(pat), key=_version_score_for_filename)
        if globbed:
            return globbed[-1]
    if required:
        raise FileNotFoundError(f"Could not find any of {patterns} in {data_dir}")
    return None

def read_text_maybe_zip(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
            csv_names = [n for n in names if n.lower().endswith((".csv", ".txt"))]
            if not csv_names:
                raise ValueError(f"No CSV/TXT file found inside {path}")
            with zf.open(csv_names[0]) as fh:
                data = fh.read()
        return data.decode("latin1")
    return path.read_text(encoding="latin1", errors="ignore")


def to_date_index(series: pd.Series) -> pd.DatetimeIndex:
    s = series.astype(str).str.strip()
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def winsorize(x: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    if x.size == 0:
        return x
    a = np.nanquantile(x, lo)
    b = np.nanquantile(x, hi)
    return np.clip(x, a, b)


def row_normalize(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = np.array(A, dtype=float, copy=True)
    np.fill_diagonal(A, 0.0)
    A[~np.isfinite(A)] = 0.0
    A[A < 0] = 0.0
    n = A.shape[0]
    rs = A.sum(axis=1)
    W = np.zeros_like(A)
    for i in range(n):
        if rs[i] > eps:
            W[i] = A[i] / rs[i]
        else:
            W[i] = 1.0 / (n - 1)
            W[i, i] = 0.0
    return W


def make_equal_weights(n: int) -> np.ndarray:
    W = np.ones((n, n), dtype=float) / (n - 1)
    np.fill_diagonal(W, 0.0)
    return W


def safe_solve(A: np.ndarray, b: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    n = A.shape[0]
    A2 = np.asarray(A, dtype=float) + ridge * np.eye(n)
    b2 = np.asarray(b, dtype=float)
    try:
        x = np.linalg.solve(A2, b2)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A2, b2, rcond=None)[0]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def cvar_np(loss: np.ndarray, alpha: float = 0.95) -> float:
    loss = np.asarray(loss, dtype=float)
    loss = loss[np.isfinite(loss)]
    if loss.size == 0:
        return float("nan")
    k = max(1, int(math.ceil((1.0 - alpha) * loss.size)))
    return float(np.mean(np.sort(loss)[-k:]))


def objective_from_pi(pi: np.ndarray, payments: np.ndarray, cfg: ExperimentConfig) -> float:
    loss = -np.asarray(pi)
    ce = float(np.mean(pi) - cfg.cvar_kappa * cvar_np(loss, cfg.cvar_alpha))
    if payments.size:
        ce -= cfg.payment_var_weight * float(np.nanstd(payments))
    return ce


def policy_objective_score(principal_ce: float, crowding: float, mean_effort: float, cfg: ExperimentConfig) -> float:
    """Policy-selection objective used in grid search and neural validation.

    principal_ce is reported as a clean financial metric. policy_objective is
    the paper-aligned training/selection metric: principal CE minus systemic
    crowding penalty and an effort shortfall penalty.
    """
    effort_gap = max(0.0, float(cfg.effort_target) - float(mean_effort))
    return float(principal_ce - cfg.crowding_obj_weight * float(crowding) - cfg.effort_target_weight * effort_gap * effort_gap)


def rolling_sum_matrix(x: np.ndarray, h: int) -> np.ndarray:
    """Return rolling H-day sums for a T x N matrix."""
    if x.shape[0] < h:
        return np.empty((0, x.shape[1]))
    cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    return cs[h:] - cs[:-h]


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------

def load_industry_returns(path: Path, section: str = "value") -> pd.DataFrame:
    """Load Fama-French 49 Industry daily returns; returns are decimal."""
    text = read_text_maybe_zip(path)
    lines = text.splitlines()
    if section.lower().startswith("value"):
        start_marker = "Average Value Weighted Returns -- Daily"
        end_marker = "Average Equal Weighted Returns -- Daily"
    else:
        start_marker = "Average Equal Weighted Returns -- Daily"
        end_marker = None

    start = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"Could not find section {start_marker} in {path}")

    end = len(lines)
    if end_marker is not None:
        for i in range(start + 1, len(lines)):
            if end_marker in lines[i]:
                end = i - 2
                break

    section_lines = [ln for ln in lines[start:end] if ln.strip()]
    csv_text = "\n".join(section_lines)
    df = pd.read_csv(io.StringIO(csv_text))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = to_date_index(df["date"])
    df = df.dropna(subset=["date"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([-99.99, -999.0], np.nan)
    df = df / 100.0
    # Remove all-empty rows/cols and sort
    df = df.dropna(how="all").sort_index()
    return df


def parse_french_factor_file(path: Path) -> pd.DataFrame:
    """Parse a Fama-French CSV/TXT with text header; returns are decimal."""
    text = read_text_maybe_zip(path)
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(",") and any(tok in stripped for tok in ["Mkt-RF", "Mom", "MOM"]):
            header_idx = i
            break
    if header_idx is None:
        # fallback: first line that begins with comma
        for i, line in enumerate(lines):
            if line.strip().startswith(","):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"Could not find factor header in {path}")

    data_lines = []
    for line in lines[header_idx:]:
        s = line.strip()
        if not s:
            if data_lines:
                break
            continue
        # After header, stop if first token is not empty header or an 8-digit date
        first = s.split(",")[0].strip()
        if data_lines and not re.match(r"^\d{8}$", first):
            break
        data_lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = to_date_index(df["date"])
    df = df.dropna(subset=["date"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df / 100.0
    df = df.sort_index()
    # Standardize momentum name
    for c in list(df.columns):
        if c.strip().lower() == "mom":
            df = df.rename(columns={c: "Mom"})
    return df


def load_vix_file(path: Optional[Path], value_col: str) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as e:
            # VIX/VXV are optional stress-state variables. If pyarrow/fastparquet
            # is not installed, continue without them instead of failing the
            # whole experiment. Install pyarrow to enable VIX HighVIX analysis.
            print(f"Warning: could not read {path.name} because parquet support is missing: {e}. Continuing without {value_col}.")
            return None
    else:
        df = pd.read_csv(path)
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "observation" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    val_col = value_col if value_col in df.columns else [c for c in df.columns if c != date_col][0]
    out = df[[date_col, val_col]].copy()
    out.columns = ["date", value_col]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["date"]).set_index("date").sort_index()
    return out


def load_etf_wide_prices(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ETF wide table. Returns price table and macro table."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    date_col = [c for c in df.columns if c.lower() == "date"][0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    etf_cols = [
        "EEM", "GLD", "HYG", "IEF", "IWM", "LQD", "QQQ", "SPY", "TLT", "UUP",
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    ]
    etf_cols = [c for c in etf_cols if c in df.columns]
    prices = df[etf_cols].apply(pd.to_numeric, errors="coerce")
    macro_cols = [c for c in df.columns if c not in etf_cols]
    macro = df[macro_cols].apply(pd.to_numeric, errors="coerce")
    return prices, macro


# -----------------------------------------------------------------------------
# Episode construction
# -----------------------------------------------------------------------------

@dataclass
class EpisodeSet:
    name: str
    dates: pd.DatetimeIndex
    symbols: List[str]
    features: np.ndarray        # B x N x D
    W_equal: np.ndarray         # B x N x N
    W_tail: np.ndarray          # B x N x N
    W_corr: np.ndarray          # B x N x N
    W_gls: np.ndarray           # B x N x N
    W_random: np.ndarray        # B x N x N
    m_marginal: np.ndarray      # B x N, legacy Q@s0 score
    m_tail_cvar: np.ndarray     # B x N, tail-event marginal loss score
    m_centrality: np.ndarray    # B x N
    m_random: np.ndarray        # B x N, permuted graph-risk score for clean random controls
    s0: np.ndarray              # B x N
    pi: np.ndarray              # B x N
    mu: np.ndarray              # B x N
    alpha: np.ndarray           # B x N
    k_cost: np.ndarray          # B x N
    h_cost: np.ndarray          # B x N
    gamma: np.ndarray           # B x N
    sigma_episode: np.ndarray   # B x N
    varF_episode: np.ndarray    # B
    Fsum: np.ndarray            # B, realized future factor sum
    resid_sum: np.ndarray       # B x N, realized future residual sum
    beta_noise: np.ndarray      # B x N, realized exposure-estimation noise
    Fsum_mc: np.ndarray         # B x M, realized + bootstrap scenario factor sums
    resid_sum_mc: np.ndarray    # B x M x N, scenario residual sums
    beta_noise_mc: np.ndarray   # B x M x N, scenario exposure-estimation noise
    q5_episode: np.ndarray      # B x N
    Q: np.ndarray               # B x N x N
    vix: np.ndarray             # B

    def subset_by_date(self, start: str, end: str, name: Optional[str] = None) -> "EpisodeSet":
        mask = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return self.subset(mask, name=name or f"{self.name}_{start}_{end}")

    def subset(self, mask: np.ndarray, name: Optional[str] = None) -> "EpisodeSet":
        mask = np.asarray(mask, dtype=bool)
        idx = np.where(mask)[0]
        # Build manually. Do NOT call dataclasses.asdict(self) here: it deep-copies
        # all arrays and can cause large memory spikes when running neural models.
        return EpisodeSet(
            name=name or self.name,
            dates=self.dates[idx],
            symbols=list(self.symbols),
            features=self.features[idx],
            W_equal=self.W_equal[idx],
            W_tail=self.W_tail[idx],
            W_corr=self.W_corr[idx],
            W_gls=self.W_gls[idx],
            W_random=self.W_random[idx],
            m_marginal=self.m_marginal[idx],
            m_tail_cvar=self.m_tail_cvar[idx],
            m_centrality=self.m_centrality[idx],
            m_random=self.m_random[idx],
            s0=self.s0[idx],
            pi=self.pi[idx],
            mu=self.mu[idx],
            alpha=self.alpha[idx],
            k_cost=self.k_cost[idx],
            h_cost=self.h_cost[idx],
            gamma=self.gamma[idx],
            sigma_episode=self.sigma_episode[idx],
            varF_episode=self.varF_episode[idx],
            Fsum=self.Fsum[idx],
            resid_sum=self.resid_sum[idx],
            beta_noise=self.beta_noise[idx],
            Fsum_mc=self.Fsum_mc[idx],
            resid_sum_mc=self.resid_sum_mc[idx],
            beta_noise_mc=self.beta_noise_mc[idx],
            q5_episode=self.q5_episode[idx],
            Q=self.Q[idx],
            vix=self.vix[idx],
        )

    @property
    def B(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.symbols)


def build_tail_graph(Rwin: np.ndarray, q: float = 0.05, eps: float = 1e-12) -> np.ndarray:
    thresholds = np.nanquantile(Rwin, q, axis=0)
    bad = (Rwin < thresholds[None, :]).astype(float)
    A = bad.T @ bad / max(1, Rwin.shape[0])
    np.fill_diagonal(A, 0.0)
    A[A < eps] = 0.0
    return A


def build_corr_graph(Rwin: np.ndarray) -> np.ndarray:
    C = np.corrcoef(np.nan_to_num(Rwin, nan=0.0), rowvar=False)
    A = np.abs(C)
    A[~np.isfinite(A)] = 0.0
    np.fill_diagonal(A, 0.0)
    return A


def build_gls_weights(Rwin: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Covariance-optimal peer-filtering weights for each row i."""
    X = np.nan_to_num(Rwin - np.nanmean(Rwin, axis=0, keepdims=True), nan=0.0)
    cov = np.cov(X, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    n = cov.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        S = cov[np.ix_(idx, idx)] + ridge * np.eye(n - 1)
        c = cov[idx, i]
        ones = np.ones(n - 1)
        Sinv_c = safe_solve(S, c, ridge=ridge)
        Sinv_1 = safe_solve(S, ones, ridge=ridge)
        denom = float(ones @ Sinv_1)
        if abs(denom) < 1e-12:
            w = np.ones(n - 1) / (n - 1)
        else:
            lam = (ones @ Sinv_c - 1.0) / denom
            w = Sinv_c - lam * Sinv_1
        # Winsorize extreme negative/positive GLS weights for numerical stability.
        w = np.clip(w, -2.0, 2.0)
        # Re-normalize to sum 1.
        s = w.sum()
        if abs(s) < 1e-8:
            w = np.ones(n - 1) / (n - 1)
        else:
            w = w / s
        W[i, idx] = w
    np.fill_diagonal(W, 0.0)
    return W


def random_permuted_weights(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = W.shape[0]
    perm = rng.permutation(n)
    Wp = W[np.ix_(perm, perm)]
    # Keep original node labels but permute graph structure.
    return row_normalize(Wp)


def compute_rolling_beta(Rwin: np.ndarray, Fwin: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Fc = Fwin - np.nanmean(Fwin)
    Rc = Rwin - np.nanmean(Rwin, axis=0, keepdims=True)
    varF = float(np.nanvar(Fc)) + eps
    beta = np.nanmean(Rc * Fc[:, None], axis=0) / varF
    return np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)


def drawdown_from_window(Rwin: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1.0 + np.nan_to_num(Rwin, nan=0.0), axis=0)
    peak = np.maximum.accumulate(wealth, axis=0)
    dd = wealth / np.maximum(peak, 1e-12) - 1.0
    return np.nanmin(dd, axis=0)


def standardize_feature_sets(train: EpisodeSet, val: EpisodeSet, test: EpisodeSet) -> Tuple[EpisodeSet, EpisodeSet, EpisodeSet, Tuple[np.ndarray, np.ndarray]]:
    mu = np.nanmean(train.features.reshape(-1, train.features.shape[-1]), axis=0)
    sd = np.nanstd(train.features.reshape(-1, train.features.shape[-1]), axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)

    def apply(es: EpisodeSet) -> EpisodeSet:
        out = es.subset(np.ones(es.B, dtype=bool), name=es.name)
        out.features = (out.features - mu[None, None, :]) / sd[None, None, :]
        out.features = np.nan_to_num(out.features, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    return apply(train), apply(val), apply(test), (mu, sd)


def build_episode_set(
    name: str,
    returns: pd.DataFrame,
    factor: pd.Series,
    vix: Optional[pd.Series],
    cfg: ExperimentConfig,
    start_date: str,
    end_date: str,
    rng: np.random.Generator,
) -> EpisodeSet:
    """Build rolling episode states from return/factor data."""
    # Align and clean.
    common_idx = returns.index.intersection(factor.index)
    returns = returns.loc[common_idx].sort_index()
    factor = factor.loc[common_idx].sort_index()
    if vix is None:
        vix_aligned = pd.Series(np.nan, index=returns.index, name="VIXCLS")
    else:
        vix_aligned = vix.reindex(returns.index).ffill().bfill()

    # Restrict after overall start but keep enough history before selected episode dates.
    R = returns.values.astype(float)
    F = factor.values.astype(float)
    dates = returns.index
    symbols = list(returns.columns)
    n = len(symbols)

    L = cfg.window
    H = cfg.episode_horizon
    step = cfg.step
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    arrays: Dict[str, List[Any]] = {k: [] for k in [
        "dates", "features", "W_equal", "W_tail", "W_corr", "W_gls", "W_random",
        "m_marginal", "m_tail_cvar", "m_centrality", "m_random", "s0", "pi", "mu", "alpha", "k_cost", "h_cost",
        "gamma", "sigma_episode", "varF_episode", "Fsum", "resid_sum", "beta_noise",
        "Fsum_mc", "resid_sum_mc", "beta_noise_mc", "q5_episode", "Q", "vix",
    ]}

    W_equal_const = make_equal_weights(n)

    # Find first valid idx with L history and H future; step by cfg.step.
    for idx in range(L, len(dates) - H, step):
        d = dates[idx]
        if d < start_ts or d > end_ts:
            continue
        Rwin = R[idx - L: idx]
        Rfut = R[idx: idx + H]
        Fwin = F[idx - L: idx]
        Ffut = F[idx: idx + H]
        if np.isnan(Rwin).any() or np.isnan(Rfut).any() or np.isnan(Fwin).any() or np.isnan(Ffut).any():
            continue

        beta = compute_rolling_beta(Rwin, Fwin)
        # Positive natural systematic exposure; normalize around 1.
        s0 = np.clip(beta, 0.0, 2.5)
        if np.nanmean(s0) > 1e-8:
            s0 = s0 / np.nanmean(s0)
        s0 = np.clip(s0, 0.05, cfg.s_max)

        f_mean = float(np.nanmean(Fwin))
        f_center = Fwin - f_mean
        f_var_daily = float(np.nanvar(f_center)) + 1e-10
        varF_episode = f_var_daily * H
        Fsum = float(np.sum(Ffut - f_mean))

        r_mean = np.nanmean(Rwin, axis=0)
        resid_win = Rwin - r_mean[None, :] - (Fwin - f_mean)[:, None] * beta[None, :]
        resid_fut = Rfut - r_mean[None, :] - (Ffut - f_mean)[:, None] * beta[None, :]
        resid_sum = np.sum(resid_fut, axis=0)
        sigma_episode = np.nanstd(resid_win, axis=0) * math.sqrt(H)
        sigma_episode = np.clip(sigma_episode, 1e-4, None)

        denom = float(np.sum((Ffut - np.nanmean(Ffut)) ** 2)) + 1e-10
        beta_noise = np.sum(resid_fut * (Ffut - np.nanmean(Ffut))[:, None], axis=0) / denom
        beta_noise = np.clip(beta_noise, -0.50, 0.50)

        # Monte Carlo / historical-block bootstrap scenarios for robust tail evaluation.
        # Scenario 0 is the realized forward path when include_realized_path=True; the
        # remaining scenarios are H-day blocks sampled from the pre-contract window.
        M = max(1, int(cfg.mc_paths))
        Fsum_mc = np.empty(M, dtype=float)
        resid_sum_mc = np.empty((M, n), dtype=float)
        beta_noise_mc = np.empty((M, n), dtype=float)
        cursor = 0
        if cfg.include_realized_path:
            Fsum_mc[0] = Fsum
            resid_sum_mc[0] = resid_sum
            beta_noise_mc[0] = beta_noise
            cursor = 1
        max_start = max(1, L - H + 1)
        f_center_win = Fwin - f_mean
        for m_idx in range(cursor, M):
            j = int(rng.integers(0, max_start))
            f_block = f_center_win[j:j + H]
            r_block = resid_win[j:j + H]
            Fsum_mc[m_idx] = float(np.sum(f_block))
            resid_sum_mc[m_idx] = np.sum(r_block, axis=0)
            den_block = float(np.sum((f_block - np.nanmean(f_block)) ** 2)) + 1e-10
            beta_noise_mc[m_idx] = np.clip(
                np.sum(r_block * (f_block - np.nanmean(f_block))[:, None], axis=0) / den_block,
                -0.50,
                0.50,
            )

        # Expected episode drift; heavily shrink empirical window means to avoid noise domination.
        mu = 0.20 * r_mean * H
        # Systematic exposure premium: positive base with mild beta heterogeneity.
        beta_norm = (s0 - np.mean(s0)) / (np.std(s0) + 1e-8)
        pi = cfg.pi0 + cfg.pi_beta_mult * beta_norm
        pi = np.clip(pi, cfg.pi0 * 0.40, cfg.pi0 * 1.80)

        # Graphs.
        A_tail = build_tail_graph(Rwin, q=cfg.tail_q)
        A_corr = build_corr_graph(Rwin)
        W_tail = row_normalize(A_tail)
        W_corr = row_normalize(A_corr)
        W_gls = build_gls_weights(Rwin, ridge=cfg.ridge)
        W_random = random_permuted_weights(W_tail, rng)

        # Graph systemic-risk matrix and scores.
        D_tail = np.diag(A_tail.sum(axis=1))
        Q = D_tail + A_tail + 1e-3 * np.eye(n)
        # Legacy marginal score: graph propagation of natural beta exposure.
        m_raw = Q @ s0
        m_pos = np.maximum(m_raw, 0.0)
        m_marginal = m_pos / (np.mean(m_pos) + 1e-8)
        # Naive centrality score.
        degree = A_tail.sum(axis=1)
        m_cent = degree / (np.mean(degree) + 1e-8)
        m_cent = np.nan_to_num(m_cent, nan=1.0, posinf=1.0, neginf=1.0)
        # Tail-event marginal contribution score. This is intentionally more
        # financial than centrality: it asks which nodes lose money when the
        # aggregate portfolio is in the rolling worst tail, then propagates that
        # contribution through the tail graph.
        agg_loss = -np.nanmean(Rwin, axis=1)
        tail_cut = np.nanquantile(agg_loss, 1.0 - cfg.tail_q)
        tail_mask = agg_loss >= tail_cut
        if int(np.sum(tail_mask)) < 2:
            tail_mask = agg_loss >= np.nanquantile(agg_loss, 0.90)
        tail_contrib = np.nanmean(np.maximum(-Rwin[tail_mask], 0.0), axis=0)
        tail_contrib = np.nan_to_num(tail_contrib, nan=0.0, posinf=0.0, neginf=0.0)
        tail_contrib = tail_contrib / (np.mean(tail_contrib) + 1e-8)
        m_tail_raw = Q @ tail_contrib
        m_tail_pos = np.maximum(m_tail_raw, 0.0)
        m_tail_cvar = m_tail_pos / (np.mean(m_tail_pos) + 1e-8)
        # Clean random negative-control score: permute the graph-risk score across nodes.
        m_random = m_tail_cvar[rng.permutation(n)]

        # Graph-propagated tail-shock generation. Earlier versions generated
        # future residual shocks by factor/residual bootstrap only, so the true
        # graph and random graph faced nearly identical tail distributions. This
        # transformation makes the simulated downside distribution depend on the
        # true public tail graph while all methods are evaluated on the same
        # graph-dependent environment.
        rho_prop = float(getattr(cfg, "graph_propagation_rho", 0.0))
        if rho_prop > 1e-12:
            rho_prop = min(max(rho_prop, 0.0), 0.95)
            Pprop = safe_solve(np.eye(n) - rho_prop * W_tail, np.eye(n), ridge=cfg.ridge)
            resid_sum = Pprop @ resid_sum
            resid_sum_mc = resid_sum_mc @ Pprop.T
            beta_noise = np.clip(Pprop @ beta_noise, -0.50, 0.50)
            beta_noise_mc = np.clip(beta_noise_mc @ Pprop.T, -0.50, 0.50)

        # Episode downside threshold from rolling H-day sums.
        roll_h = rolling_sum_matrix(Rwin, H)
        if roll_h.shape[0] > 0:
            q5_episode = np.nanquantile(roll_h, cfg.tail_q, axis=0)
        else:
            q5_episode = np.nanquantile(Rwin, cfg.tail_q, axis=0) * math.sqrt(H)

        dd = drawdown_from_window(Rwin)
        tail_loss = -np.nanquantile(Rwin, cfg.tail_q, axis=0)
        vol = np.nanstd(Rwin, axis=0) * math.sqrt(252)
        mu_ann = np.nanmean(Rwin, axis=0) * 252
        vix_value = float(vix_aligned.loc[d]) if d in vix_aligned.index else np.nan
        vix_scaled = np.nan_to_num(vix_value / 100.0, nan=0.20)

        # Node features. VIX repeated as node feature because the GNN is node-level.
        feats = np.column_stack([
            mu_ann,
            vol,
            beta,
            s0,
            dd,
            tail_loss,
            m_cent,
            m_tail_cvar,
            np.full(n, vix_scaled),
        ])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        arrays["dates"].append(d)
        arrays["features"].append(feats)
        arrays["W_equal"].append(W_equal_const)
        arrays["W_tail"].append(W_tail)
        arrays["W_corr"].append(W_corr)
        arrays["W_gls"].append(W_gls)
        arrays["W_random"].append(W_random)
        arrays["m_marginal"].append(m_marginal)
        arrays["m_tail_cvar"].append(m_tail_cvar)
        arrays["m_centrality"].append(m_cent)
        arrays["m_random"].append(m_random)
        arrays["s0"].append(s0)
        arrays["pi"].append(pi)
        arrays["mu"].append(mu)
        arrays["alpha"].append(np.full(n, cfg.alpha0))
        arrays["k_cost"].append(np.full(n, cfg.k_cost))
        arrays["h_cost"].append(np.full(n, cfg.h_cost))
        arrays["gamma"].append(np.full(n, cfg.gamma_agent))
        arrays["sigma_episode"].append(sigma_episode)
        arrays["varF_episode"].append(varF_episode)
        arrays["Fsum"].append(Fsum)
        arrays["resid_sum"].append(resid_sum)
        arrays["beta_noise"].append(beta_noise)
        arrays["Fsum_mc"].append(Fsum_mc)
        arrays["resid_sum_mc"].append(resid_sum_mc)
        arrays["beta_noise_mc"].append(beta_noise_mc)
        arrays["q5_episode"].append(q5_episode)
        arrays["Q"].append(Q)
        arrays["vix"].append(vix_value)

    if not arrays["dates"]:
        raise ValueError(f"No episodes built for {name}. Check date ranges and missing data.")

    def stack(key: str) -> np.ndarray:
        return np.asarray(arrays[key], dtype=float)

    return EpisodeSet(
        name=name,
        dates=pd.DatetimeIndex(arrays["dates"]),
        symbols=symbols,
        features=stack("features"),
        W_equal=stack("W_equal"),
        W_tail=stack("W_tail"),
        W_corr=stack("W_corr"),
        W_gls=stack("W_gls"),
        W_random=stack("W_random"),
        m_marginal=stack("m_marginal"),
        m_tail_cvar=stack("m_tail_cvar"),
        m_centrality=stack("m_centrality"),
        m_random=stack("m_random"),
        s0=stack("s0"),
        pi=stack("pi"),
        mu=stack("mu"),
        alpha=stack("alpha"),
        k_cost=stack("k_cost"),
        h_cost=stack("h_cost"),
        gamma=stack("gamma"),
        sigma_episode=stack("sigma_episode"),
        varF_episode=stack("varF_episode"),
        Fsum=stack("Fsum"),
        resid_sum=stack("resid_sum"),
        beta_noise=stack("beta_noise"),
        Fsum_mc=stack("Fsum_mc"),
        resid_sum_mc=stack("resid_sum_mc"),
        beta_noise_mc=stack("beta_noise_mc"),
        q5_episode=stack("q5_episode"),
        Q=stack("Q"),
        vix=stack("vix"),
    )


# -----------------------------------------------------------------------------
# Contract environment: numpy implementation
# -----------------------------------------------------------------------------

@dataclass
class MethodSpec:
    name: str
    weight_kind: str = "equal"       # equal, tail, corr, gls, random
    penalty_kind: str = "none"       # none, marginal, centrality
    allow_theta: bool = True
    allow_eta: bool = True
    allow_lambda: bool = True
    force_positive_eta: bool = False
    force_positive_lambda: bool = False
    is_neural: bool = False


def get_weights(es: EpisodeSet, kind: str) -> np.ndarray:
    if kind == "equal":
        return es.W_equal
    if kind == "tail":
        return es.W_tail
    if kind == "corr":
        return es.W_corr
    if kind == "gls":
        return es.W_gls
    if kind == "random":
        return es.W_random
    raise ValueError(f"Unknown weight_kind={kind}")


def get_penalty_score(es: EpisodeSet, kind: str) -> np.ndarray:
    if kind == "none":
        return np.zeros_like(es.m_marginal)
    if kind == "marginal":
        return es.m_marginal
    if kind in {"tail_cvar", "cvar", "tail"}:
        return es.m_tail_cvar
    if kind == "centrality":
        return es.m_centrality
    if kind == "random":
        return es.m_random
    raise ValueError(f"Unknown penalty_kind={kind}")


def solve_best_response_np(
    theta: np.ndarray,
    eta: np.ndarray,
    lambd: np.ndarray,
    W: np.ndarray,
    m: np.ndarray,
    es: EpisodeSet,
    bidx: int,
    cfg: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    z = theta + eta
    h = es.h_cost[bidx]
    gamma = es.gamma[bidx]
    varF = es.varF_episode[bidx]
    s0 = es.s0[bidx]
    pi = es.pi[bidx]
    alpha = es.alpha[bidx]
    k = es.k_cost[bidx]
    # A = diag(h + gamma*varF*z^2) - gamma*varF*diag(z*eta) W
    A = np.diag(h + gamma * varF * z * z) - (gamma * varF) * ((z * eta)[:, None] * W)
    rhs = h * s0 + z * pi - lambd * m
    s = safe_solve(A, rhs, ridge=cfg.ridge)
    s = np.clip(s, 0.0, cfg.s_max)
    e = alpha * z / np.maximum(k, 1e-12)
    e = np.clip(e, 0.0, cfg.e_max)
    return e, s


def evaluate_fixed_method(
    es: EpisodeSet,
    spec: MethodSpec,
    coeffs: Tuple[float, float, float],
    cfg: ExperimentConfig,
    return_episode_details: bool = False,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    """Evaluate a fixed contract on MC/bootstrap scenario paths.

    The previous single-realized-path evaluation made co-crash and CVaR weak
    evidence. Here each decision state is evaluated over es.Fsum_mc and
    es.resid_sum_mc, so downside risk reacts to the contract-induced exposure s*.
    """
    theta0, eta0, lambda0 = coeffs
    B, N = es.B, es.N
    W_all = get_weights(es, spec.weight_kind)
    m_all = get_penalty_score(es, spec.penalty_kind)

    pi_paths_all, pay_paths_all, co_paths_all = [], [], []
    crowd_list, effort_list, sexp_list, ir_gap_list = [], [], [], []
    theta_list, eta_list, lambda_list = [], [], []
    rows = []

    for b in range(B):
        W = W_all[b]
        m = m_all[b]
        theta = np.full(N, theta0 if spec.allow_theta else 0.0)
        eta = np.full(N, eta0 if spec.allow_eta else 0.0)
        lambd = np.full(N, lambda0 if spec.allow_lambda else 0.0)
        if spec.penalty_kind == "none":
            lambd[:] = 0.0
        if spec.weight_kind == "equal" and not spec.allow_eta:
            eta[:] = 0.0

        e, s_vec = solve_best_response_np(theta, eta, lambd, W, m, es, b, cfg)
        z = theta + eta
        EX = es.mu[b] + es.alpha[b] * e + es.pi[b] * s_vec
        bench_EX = W @ EX

        # IR base payment uses expected CE, not realized scenarios.
        var_proxy = (z ** 2) * (s_vec ** 2 * es.varF_episode[b] + es.sigma_episode[b] ** 2)
        cost_effort = 0.5 * es.k_cost[b] * e ** 2
        cost_exposure = 0.5 * es.h_cost[b] * (s_vec - es.s0[b]) ** 2
        xi_no_b_exp = theta * EX + eta * (EX - bench_EX) - lambd * m * s_vec
        CE_no_b = xi_no_b_exp - cost_effort - cost_exposure - 0.5 * es.gamma[b] * var_proxy
        base = cfg.reservation_utility - CE_no_b

        F_paths = es.Fsum_mc[b]
        resid_paths = es.resid_sum_mc[b]
        beta_noise_paths = es.beta_noise_mc[b]
        M = F_paths.shape[0]
        X = EX[None, :] + s_vec[None, :] * F_paths[:, None] + resid_paths
        hat_s = np.clip(s_vec[None, :] + beta_noise_paths, 0.0, cfg.s_max)
        bench_X = X @ W.T
        xi = base[None, :] + theta[None, :] * X + eta[None, :] * (X - bench_X) - lambd[None, :] * m[None, :] * hat_s
        Pi_paths = np.sum(X, axis=1) - np.sum(xi, axis=1)
        pay_paths = np.sum(xi, axis=1)
        co_paths = (np.sum(X < es.q5_episode[b][None, :], axis=1) >= math.ceil(cfg.co_crash_frac * N)).astype(float)

        crowd = float(s_vec @ es.Q[b] @ s_vec / N)
        effort = float(np.mean(e))
        sexp = float(np.mean(s_vec))
        ir_gap = float(np.min(CE_no_b + base - cfg.reservation_utility))

        pi_paths_all.append(Pi_paths)
        pay_paths_all.append(pay_paths)
        co_paths_all.append(co_paths)
        crowd_list.append(crowd)
        effort_list.append(effort)
        sexp_list.append(sexp)
        ir_gap_list.append(ir_gap)
        theta_list.append(float(np.mean(theta)))
        eta_list.append(float(np.mean(eta)))
        lambda_list.append(float(np.mean(lambd)))

        if return_episode_details:
            rows.append({
                "date": es.dates[b],
                "method": spec.name,
                "display_method": paper_method_name(spec.name),
                "principal_payoff": float(np.mean(Pi_paths)),
                "principal_CVaR95_loss_episode": cvar_np(-Pi_paths, 0.95),
                "total_payment": float(np.mean(pay_paths)),
                "crowding": crowd,
                "mean_effort": effort,
                "mean_systematic_exposure": sexp,
                "co_crash": float(np.mean(co_paths)),
                "min_ir_gap": ir_gap,
                "theta": float(np.mean(theta)),
                "eta": float(np.mean(eta)),
                "lambda": float(np.mean(lambd)),
                "M_paths": float(M),
            })

    pi_arr = np.concatenate(pi_paths_all) if pi_paths_all else np.array([])
    pay_arr = np.concatenate(pay_paths_all) if pay_paths_all else np.array([])
    co_arr = np.concatenate(co_paths_all) if co_paths_all else np.array([])
    loss = -pi_arr
    avg_crowding = float(np.mean(crowd_list))
    avg_effort = float(np.mean(effort_list))
    principal_ce = objective_from_pi(pi_arr, pay_arr, cfg)
    policy_obj = policy_objective_score(principal_ce, avg_crowding, avg_effort, cfg)
    metrics = {
        "method": spec.name,
        "display_method": paper_method_name(spec.name),
        "B": float(B),
        "M_paths": float(es.Fsum_mc.shape[1]) if hasattr(es, "Fsum_mc") else 1.0,
        "principal_CE": principal_ce,
        "policy_objective": policy_obj,
        "mean_payoff": float(np.mean(pi_arr)),
        "std_payoff": float(np.std(pi_arr)),
        "CVaR95_loss": cvar_np(loss, 0.95),
        "CVaR99_loss": cvar_np(loss, 0.99),
        "mean_payment": float(np.mean(pay_arr)),
        "std_payment": float(np.std(pay_arr)),
        "crowding": avg_crowding,
        "co_crash_freq": float(np.mean(co_arr)),
        "mean_effort": avg_effort,
        "mean_systematic_exposure": float(np.mean(sexp_list)),
        "IR_violation_rate": float(np.mean(np.asarray(ir_gap_list) < -1e-8)),
        "min_IR_gap": float(np.min(ir_gap_list)),
        "theta": float(np.mean(theta_list)),
        "eta": float(np.mean(eta_list)),
        "lambda": float(np.mean(lambda_list)),
    }
    details = pd.DataFrame(rows) if return_episode_details else None
    return metrics, details


def grid_search_method(
    train: EpisodeSet,
    spec: MethodSpec,
    cfg: ExperimentConfig,
) -> Tuple[Tuple[float, float, float], pd.DataFrame]:
    theta_grid = cfg.theta_grid if spec.allow_theta else (0.0,)
    eta_grid = cfg.eta_grid if spec.allow_eta else (0.0,)
    lambda_grid = cfg.lambda_grid if (spec.allow_lambda and spec.penalty_kind != "none") else (0.0,)

    # Some methods should not use eta/lambda.
    if spec.name.lower().startswith("fixed"):
        theta_grid, eta_grid, lambda_grid = (0.0,), (0.0,), (0.0,)
    if spec.name.lower() == "linearpnl":
        eta_grid, lambda_grid = (0.0,), (0.0,)

    rows = []
    best_obj = -np.inf
    best = (0.0, 0.0, 0.0)
    for th in theta_grid:
        for et in eta_grid:
            for la in lambda_grid:
                if spec.name.lower().startswith("fixed") and (th != 0 or et != 0 or la != 0):
                    continue
                if spec.force_positive_eta and et <= 0.0:
                    continue
                if spec.force_positive_lambda and la <= 0.0:
                    continue
                metrics, _ = evaluate_fixed_method(train, spec, (float(th), float(et), float(la)), cfg)
                row = dict(metrics)
                row.update({"theta_grid": th, "eta_grid": et, "lambda_grid": la})
                rows.append(row)
                obj = metrics.get("policy_objective", metrics["principal_CE"])
                if obj > best_obj:
                    best_obj = obj
                    best = (float(th), float(et), float(la))
    return best, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Neural contract layer: torch implementation
# -----------------------------------------------------------------------------

class GraphContractNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, cfg: ExperimentConfig, use_graph: bool = True):
        super().__init__()
        self.use_graph = use_graph
        self.cfg = cfg
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        # Outputs: effective incentive z, split rho, systemic penalty lambda.
        # theta and eta are constructed from z so that theta+eta is hard-bounded.
        self.out = nn.Linear(hidden_dim, 3)
        self.act = nn.ReLU()

    def forward(self, x: "torch.Tensor", W_message: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        # x: B x N x D. W_message is used only for message passing.
        if self.use_graph:
            eye = torch.eye(W_message.shape[-1], device=W_message.device, dtype=W_message.dtype).unsqueeze(0)
            h = torch.bmm(W_message + eye, x)
        else:
            h = x
        h = self.act(self.lin1(h))
        if self.use_graph:
            eye = torch.eye(W_message.shape[-1], device=W_message.device, dtype=W_message.dtype).unsqueeze(0)
            h = torch.bmm(W_message + eye, h)
        h = self.act(self.lin2(h))
        raw = self.out(h)

        # Hard effort constraint. Since e_i*=alpha_i(theta_i+eta_i)/k_i,
        # z_min=0.80 with alpha0=0.05 and k=1 forces effort >= 0.04.
        z_min = float(max(0.0, self.cfg.hard_effort_z_min))
        z_cap = float(max(z_min + 1e-6, min(self.cfg.hard_effort_z_cap, self.cfg.theta_cap + self.cfg.eta_cap)))
        z = z_min + (z_cap - z_min) * torch.sigmoid(raw[..., 0])

        # Split z into theta and eta while respecting individual caps.
        eta_low = torch.clamp(z - self.cfg.theta_cap, min=0.0)
        eta_high = torch.minimum(torch.full_like(z, self.cfg.eta_cap), z)
        split = torch.sigmoid(raw[..., 1])
        eta = eta_low + (eta_high - eta_low) * split
        theta = z - eta

        lambd = self.cfg.lambda_cap * torch.sigmoid(raw[..., 2])
        return theta, eta, lambd


class UnconstrainedContractNet(nn.Module):
    """Same backbone, but no hard-effort output parameterization.

    This is used only for the post-hoc projection ablation. During training it
    emits ordinary bounded contract coefficients; at evaluation we project
    theta+eta to the same hard-effort feasible set used by GraphSignal-ICL.
    """
    def __init__(self, in_dim: int, hidden_dim: int, cfg: ExperimentConfig, use_graph: bool = True):
        super().__init__()
        self.use_graph = use_graph
        self.cfg = cfg
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 3)
        self.act = nn.ReLU()

    def forward(self, x: "torch.Tensor", W_message: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if self.use_graph:
            eye = torch.eye(W_message.shape[-1], device=W_message.device, dtype=W_message.dtype).unsqueeze(0)
            h = torch.bmm(W_message + eye, x)
        else:
            h = x
        h = self.act(self.lin1(h))
        if self.use_graph:
            eye = torch.eye(W_message.shape[-1], device=W_message.device, dtype=W_message.dtype).unsqueeze(0)
            h = torch.bmm(W_message + eye, h)
        h = self.act(self.lin2(h))
        raw = self.out(h)
        theta = self.cfg.theta_cap * torch.sigmoid(raw[..., 0])
        eta = self.cfg.eta_cap * torch.sigmoid(raw[..., 1])
        lambd = self.cfg.lambda_cap * torch.sigmoid(raw[..., 2])
        return theta, eta, lambd


def project_to_hard_effort_torch(theta: "torch.Tensor", eta: "torch.Tensor", cfg: ExperimentConfig) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Project coefficients onto theta+eta >= z_min with caps.

    The split between theta and eta is preserved as much as possible. This gives
    a clean post-hoc projection baseline against constrained-during-training.
    """
    z_old = torch.clamp(theta + eta, min=1e-8)
    z_min = float(max(0.0, cfg.hard_effort_z_min))
    z_cap = float(max(z_min + 1e-6, min(cfg.hard_effort_z_cap, cfg.theta_cap + cfg.eta_cap)))
    z_new = torch.clamp(z_old, min=z_min, max=z_cap)
    split = torch.clamp(eta / z_old, 0.0, 1.0)
    eta_low = torch.clamp(z_new - cfg.theta_cap, min=0.0)
    eta_high = torch.minimum(torch.full_like(z_new, cfg.eta_cap), z_new)
    eta_new = eta_low + (eta_high - eta_low) * split
    theta_new = z_new - eta_new
    return theta_new, eta_new

def torchify_episode_set(es: EpisodeSet, device: str = "cpu") -> Dict[str, "torch.Tensor"]:
    def t(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=device)
    return {
        "features": t(es.features),
        "W_equal": t(es.W_equal),
        "W_tail": t(es.W_tail),
        "W_corr": t(es.W_corr),
        "W_gls": t(es.W_gls),
        "W_random": t(es.W_random),
        "m_marginal": t(es.m_marginal),
        "m_tail_cvar": t(es.m_tail_cvar),
        "m_centrality": t(es.m_centrality),
        "m_random": t(es.m_random),
        "s0": t(es.s0),
        "pi": t(es.pi),
        "mu": t(es.mu),
        "alpha": t(es.alpha),
        "k_cost": t(es.k_cost),
        "h_cost": t(es.h_cost),
        "gamma": t(es.gamma),
        "sigma_episode": t(es.sigma_episode),
        "varF_episode": t(es.varF_episode),
        "Fsum": t(es.Fsum),
        "resid_sum": t(es.resid_sum),
        "beta_noise": t(es.beta_noise),
        "Fsum_mc": t(es.Fsum_mc),
        "resid_sum_mc": t(es.resid_sum_mc),
        "beta_noise_mc": t(es.beta_noise_mc),
        "q5_episode": t(es.q5_episode),
        "Q": t(es.Q),
    }


def contract_forward_torch(
    data: Dict[str, "torch.Tensor"],
    theta: "torch.Tensor",
    eta: "torch.Tensor",
    lambd: "torch.Tensor",
    W: "torch.Tensor",
    m: "torch.Tensor",
    cfg: ExperimentConfig,
) -> Dict[str, "torch.Tensor"]:
    B, N = theta.shape
    z = theta + eta
    h = data["h_cost"]
    gamma = data["gamma"]
    varF = data["varF_episode"]

    rhs = h * data["s0"] + z * data["pi"] - lambd * m
    if cfg.neural_exact_ic:
        eye = torch.eye(N, device=theta.device, dtype=theta.dtype).unsqueeze(0)
        diag = h + gamma * varF[:, None] * z * z
        A = torch.diag_embed(diag) - (gamma * varF[:, None])[:, :, None] * ((z * eta)[:, :, None] * W)
        A = A + cfg.ridge * eye
        s = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
    else:
        denom = h + gamma * varF[:, None] * torch.clamp(theta * z, min=1e-8)
        s = rhs / torch.clamp(denom, min=1e-8)
    s = torch.clamp(s, 0.0, cfg.s_max)
    e = torch.clamp(data["alpha"] * z / torch.clamp(data["k_cost"], min=1e-8), 0.0, cfg.e_max)

    EX = data["mu"] + data["alpha"] * e + data["pi"] * s
    bench_EX = torch.bmm(W, EX.unsqueeze(-1)).squeeze(-1)
    var_proxy = (z ** 2) * (s ** 2 * varF[:, None] + data["sigma_episode"] ** 2)
    cost_effort = 0.5 * data["k_cost"] * e ** 2
    cost_exposure = 0.5 * data["h_cost"] * (s - data["s0"]) ** 2
    xi_no_b_exp = theta * EX + eta * (EX - bench_EX) - lambd * m * s
    CE_no_b = xi_no_b_exp - cost_effort - cost_exposure - 0.5 * data["gamma"] * var_proxy
    base = cfg.reservation_utility - CE_no_b

    F_paths = data["Fsum_mc"]              # B x M
    resid_paths = data["resid_sum_mc"]     # B x M x N
    beta_noise_paths = data["beta_noise_mc"]
    X = EX[:, None, :] + s[:, None, :] * F_paths[:, :, None] + resid_paths
    hat_s = torch.clamp(s[:, None, :] + beta_noise_paths, 0.0, cfg.s_max)
    bench_X = torch.einsum("bij,bmj->bmi", W, X)
    xi = base[:, None, :] + theta[:, None, :] * X + eta[:, None, :] * (X - bench_X) - lambd[:, None, :] * m[:, None, :] * hat_s
    Pi_paths = X.sum(dim=2) - xi.sum(dim=2)       # B x M
    payments_paths = xi.sum(dim=2)                # B x M
    crowd = torch.einsum("bi,bij,bj->b", s, data["Q"], s) / N
    co_crash_paths = (torch.sum((X < data["q5_episode"][:, None, :]).float(), dim=2) >= math.ceil(cfg.co_crash_frac * N)).float()
    return {
        "Pi": Pi_paths.reshape(-1),
        "Pi_paths": Pi_paths,
        "payments": payments_paths.reshape(-1),
        "payments_paths": payments_paths,
        "crowding": crowd,
        "co_crash": co_crash_paths.reshape(-1),
        "co_crash_paths": co_crash_paths,
        "effort": e.mean(dim=1),
        "sexp": s.mean(dim=1),
        "theta": theta.mean(dim=1),
        "eta": eta.mean(dim=1),
        "lambda": lambd.mean(dim=1),
        "IR_gap": (CE_no_b + base - cfg.reservation_utility).min(dim=1).values,
    }


def cvar_torch(loss: "torch.Tensor", alpha: float) -> "torch.Tensor":
    B = loss.shape[0]
    k = max(1, int(math.ceil((1.0 - alpha) * B)))
    return torch.topk(loss, k=k, largest=True).values.mean()


def neural_objective(out: Dict[str, "torch.Tensor"], cfg: ExperimentConfig) -> "torch.Tensor":
    """Risk-sensitive neural objective with explicit graph-claim constraints.

    The two added terms address the previous failure mode: a neural model could
    win by becoming a low-effort near-linear contract and ignoring graph-risk
    terms. We therefore penalize model-implied crowding and effort shortfall.
    """
    Pi = out["Pi"]
    payments = out["payments"]
    loss = -Pi
    obj = Pi.mean() - cfg.cvar_kappa * cvar_torch(loss, cfg.cvar_alpha)
    obj = obj - cfg.payment_var_weight * payments.std(unbiased=False)
    obj = obj - cfg.crowding_obj_weight * out["crowding"].mean()
    if cfg.effort_target > 0:
        target = torch.tensor(float(cfg.effort_target), device=Pi.device, dtype=Pi.dtype)
        effort_gap = torch.relu(target - out["effort"].mean())
        obj = obj - cfg.effort_target_weight * effort_gap.pow(2)
    # Smoothness/turnover in coefficients across sorted episodes.
    if Pi.shape[0] > 1:
        turn = (
            (out["theta"][1:] - out["theta"][:-1]).pow(2).mean()
            + (out["eta"][1:] - out["eta"][:-1]).pow(2).mean()
            + (out["lambda"][1:] - out["lambda"][:-1]).pow(2).mean()
        )
        obj = obj - cfg.turnover_weight * turn
    return obj


def feature_indices(mode: str) -> List[int]:
    """Feature modes for clean graph ablations.

    Full feature vector is [mu_ann, vol, beta, s0, drawdown, tail_loss,
    centrality_score, tail_cvar_score, vix]. Raw mode removes graph-derived
    scores so RawNodeMLP and clean RandomGNN cannot leak true graph information.
    Random-score mode keeps the same dimensionality as GraphSignal-ICL but
    replaces graph-derived risk-score columns with the randomized score.
    """
    if mode == "raw":
        return [0, 1, 2, 3, 4, 5, 8]
    if mode in {"graph_scores", "full"}:
        return list(range(9))
    if mode == "risk_scores_only":
        return [6, 7]
    if mode == "random_scores":
        return list(range(9))
    raise ValueError(f"Unknown feature_mode={mode}")


def select_torch_features(data: Dict[str, "torch.Tensor"], mode: str) -> "torch.Tensor":
    """Select feature tensors, with a same-architecture random-signal control."""
    if mode == "random_scores":
        x = data["features"].clone()
        x[..., 6] = data["m_random"]
        x[..., 7] = data["m_random"]
        return x
    cols = feature_indices(mode)
    return data["features"][..., cols]


def choose_torch_W(data: Dict[str, "torch.Tensor"], kind: str) -> "torch.Tensor":
    if kind == "equal":
        return data["W_equal"]
    if kind == "tail":
        return data["W_tail"]
    if kind == "corr":
        return data["W_corr"]
    if kind == "gls":
        return data["W_gls"]
    if kind == "random":
        return data["W_random"]
    raise ValueError(f"Unknown torch weight kind={kind}")


def choose_torch_m(data: Dict[str, "torch.Tensor"], kind: str) -> "torch.Tensor":
    if kind == "none":
        return torch.zeros_like(data["m_tail_cvar"])
    if kind in {"tail_cvar", "cvar", "tail"}:
        return data["m_tail_cvar"]
    if kind == "marginal":
        return data["m_marginal"]
    if kind == "centrality":
        return data["m_centrality"]
    if kind == "random":
        return data["m_random"]
    raise ValueError(f"Unknown torch penalty kind={kind}")


def train_neural_contract(
    train: EpisodeSet,
    val: EpisodeSet,
    test: EpisodeSet,
    cfg: ExperimentConfig,
    out_dir: Path,
    use_graph: bool = True,
    graph_kind: str = "tail",
    benchmark_kind: str = "gls",
    penalty_kind: str = "tail_cvar",
    feature_mode: str = "raw",
    label: str = "FullGNN",
    constrained_output: bool = True,
    posthoc_project: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame, Optional[pd.DataFrame]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available. Use --skip_neural.")
    device = "cpu"
    train_t = torchify_episode_set(train, device=device)
    val_t = torchify_episode_set(val, device=device)
    test_t = torchify_episode_set(test, device=device)

    train_x = select_torch_features(train_t, feature_mode)
    val_x = select_torch_features(val_t, feature_mode)
    test_x = select_torch_features(test_t, feature_mode)

    in_dim = train_x.shape[-1]
    net_cls = GraphContractNet if constrained_output else UnconstrainedContractNet
    net = net_cls(in_dim, cfg.hidden_dim, cfg, use_graph=use_graph).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1e18
    best_state = None
    best_epoch = 0
    log_rows = []
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        net.train()
        Wmsg_tr = choose_torch_W(train_t, graph_kind)
        Wbench_tr = choose_torch_W(train_t, benchmark_kind)
        mtr = choose_torch_m(train_t, penalty_kind)
        theta, eta, lambd = net(train_x, Wmsg_tr)
        if penalty_kind == "none":
            lambd = torch.zeros_like(lambd)
        out = contract_forward_torch(train_t, theta, eta, lambd, Wbench_tr, mtr, cfg)
        obj = neural_objective(out, cfg)
        loss = -obj
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            net.eval()
            with torch.no_grad():
                Wmsg_v = choose_torch_W(val_t, graph_kind)
                Wbench_v = choose_torch_W(val_t, benchmark_kind)
                mv = choose_torch_m(val_t, penalty_kind)
                thv, etv, lav = net(val_x, Wmsg_v)
                if penalty_kind == "none":
                    lav = torch.zeros_like(lav)
                outv = contract_forward_torch(val_t, thv, etv, lav, Wbench_v, mv, cfg)
                val_obj = float(neural_objective(outv, cfg).cpu())
                train_obj = float(obj.detach().cpu())
                log_rows.append({"label": label, "epoch": epoch, "train_obj": train_obj, "val_obj": val_obj})
                if val_obj > best_val + 1e-10:
                    best_val = val_obj
                    best_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 5
                if no_improve >= cfg.patience:
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save(net.state_dict(), out_dir / f"{label}_model.pt")

    net.eval()
    with torch.no_grad():
        Wmsg_test = choose_torch_W(test_t, graph_kind)
        Wbench_test = choose_torch_W(test_t, benchmark_kind)
        mtest = choose_torch_m(test_t, penalty_kind)
        th, et, la = net(test_x, Wmsg_test)
        if posthoc_project:
            th, et = project_to_hard_effort_torch(th, et, cfg)
        if penalty_kind == "none":
            la = torch.zeros_like(la)
        outt = contract_forward_torch(test_t, th, et, la, Wbench_test, mtest, cfg)
        Pi = outt["Pi"].cpu().numpy()
        payments = outt["payments"].cpu().numpy()
        loss = -Pi
        avg_crowding = float(outt["crowding"].cpu().numpy().mean())
        avg_effort = float(outt["effort"].cpu().numpy().mean())
        principal_ce = objective_from_pi(Pi, payments, cfg)
        policy_obj = policy_objective_score(principal_ce, avg_crowding, avg_effort, cfg)
        metrics = {
            "method": label,
            "display_method": paper_method_name(label),
            "B": float(test.B),
            "M_paths": float(test.Fsum_mc.shape[1]),
            "principal_CE": principal_ce,
            "policy_objective": policy_obj,
            "mean_payoff": float(Pi.mean()),
            "std_payoff": float(Pi.std()),
            "CVaR95_loss": cvar_np(loss, 0.95),
            "CVaR99_loss": cvar_np(loss, 0.99),
            "mean_payment": float(payments.mean()),
            "std_payment": float(payments.std()),
            "crowding": avg_crowding,
            "co_crash_freq": float(outt["co_crash"].cpu().numpy().mean()),
            "mean_effort": avg_effort,
            "mean_systematic_exposure": float(outt["sexp"].cpu().numpy().mean()),
            "IR_violation_rate": float((outt["IR_gap"].cpu().numpy() < -1e-8).mean()),
            "min_IR_gap": float(outt["IR_gap"].cpu().numpy().min()),
            "theta": float(outt["theta"].cpu().numpy().mean()),
            "eta": float(outt["eta"].cpu().numpy().mean()),
            "lambda": float(outt["lambda"].cpu().numpy().mean()),
            "best_epoch": float(best_epoch),
            "best_val_obj": float(best_val),
            "feature_mode": feature_mode,
            "message_graph_kind": graph_kind,
            "benchmark_kind": benchmark_kind,
            "penalty_kind": penalty_kind,
            "constrained_during_training": bool(constrained_output),
            "posthoc_projection": bool(posthoc_project),
            "hard_effort_z_min": float(cfg.hard_effort_z_min),
        }
        Pi_paths = outt["Pi_paths"].cpu().numpy()
        payments_paths = outt["payments_paths"].cpu().numpy()
        co_paths = outt["co_crash_paths"].cpu().numpy()
        details = pd.DataFrame({
            "date": test.dates,
            "method": label,
            "display_method": paper_method_name(label),
            "principal_payoff": Pi_paths.mean(axis=1),
            "principal_CVaR95_loss_episode": [cvar_np(-row, 0.95) for row in Pi_paths],
            "total_payment": payments_paths.mean(axis=1),
            "crowding": outt["crowding"].cpu().numpy(),
            "mean_effort": outt["effort"].cpu().numpy(),
            "mean_systematic_exposure": outt["sexp"].cpu().numpy(),
            "co_crash": co_paths.mean(axis=1),
            "theta": outt["theta"].cpu().numpy(),
            "eta": outt["eta"].cpu().numpy(),
            "lambda": outt["lambda"].cpu().numpy(),
        })
    return metrics, pd.DataFrame(log_rows), details


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).split(','):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def run_zmin_sensitivity(
    train: EpisodeSet,
    val: EpisodeSet,
    test: EpisodeSet,
    cfg: ExperimentConfig,
    out_dir: Path,
    z_values: Sequence[float],
    epochs: int,
) -> pd.DataFrame:
    """Run z_min sensitivity for GraphSignal-ICL only."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available; cannot run z_min sensitivity.")
    rows, logs, details = [], [], []
    for z in z_values:
        cfg_z = dataclasses.replace(cfg, hard_effort_z_min=float(z), epochs=int(epochs))
        label = f"GraphSignal-ICL_zmin_{str(float(z)).replace('.', 'p')}"
        print(f"Running z_min sensitivity: z_min={z}")
        # Reuse the outer seed so the robustness isolates z_min as much as possible.
        set_seed(cfg.random_seed)
        m, log, det = train_neural_contract(
            train, val, test, cfg_z, out_dir,
            use_graph=False,
            graph_kind="equal",
            benchmark_kind=cfg_z.neural_peer_benchmark,
            penalty_kind="tail_cvar",
            feature_mode="graph_scores",
            label=label,
            constrained_output=True,
            posthoc_project=False,
        )
        m["method"] = "GraphSignal-ICL"
        m["z_min"] = float(z)
        rows.append(m)
        log["z_min"] = float(z)
        logs.append(log)
        det["z_min"] = float(z)
        details.append(det)
    df = pd.DataFrame(rows)
    first_cols = ["z_min", "method", "mean_effort", "principal_CE", "CVaR95_loss", "CVaR99_loss", "crowding", "co_crash_freq", "IR_violation_rate"]
    cols = [c for c in first_cols if c in df.columns] + [c for c in df.columns if c not in first_cols]
    df = df[cols]
    df.to_csv(out_dir / "industry_zmin_sensitivity.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(out_dir / "industry_zmin_sensitivity_training_log.csv", index=False)
    if details:
        pd.concat(details, ignore_index=True).to_csv(out_dir / "industry_zmin_sensitivity_episode_details.csv", index=False)
    return df


def run_projection_ablation(
    train: EpisodeSet,
    val: EpisodeSet,
    test: EpisodeSet,
    cfg: ExperimentConfig,
    out_dir: Path,
    epochs: int,
) -> pd.DataFrame:
    """Compare constrained-during-training vs unconstrained + post-hoc projection."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available; cannot run projection ablation.")
    cfg_p = dataclasses.replace(cfg, epochs=int(epochs))
    rows, logs, details = [], [], []
    specs = [
        dict(label="GraphSignal-ICL_constrained", constrained_output=True, posthoc_project=False),
        dict(label="GraphSignal-ICL_posthoc_projection", constrained_output=False, posthoc_project=True),
    ]
    for sp in specs:
        print(f"Running projection ablation: {sp['label']}")
        set_seed(cfg.random_seed)
        m, log, det = train_neural_contract(
            train, val, test, cfg_p, out_dir,
            use_graph=False,
            graph_kind="equal",
            benchmark_kind=cfg_p.neural_peer_benchmark,
            penalty_kind="tail_cvar",
            feature_mode="graph_scores",
            label=sp["label"],
            constrained_output=bool(sp["constrained_output"]),
            posthoc_project=bool(sp["posthoc_project"]),
        )
        rows.append(m)
        logs.append(log)
        details.append(det)
    df = pd.DataFrame(rows)
    first_cols = ["method", "constrained_during_training", "posthoc_projection", "mean_effort", "principal_CE", "CVaR95_loss", "CVaR99_loss", "crowding", "IR_violation_rate"]
    cols = [c for c in first_cols if c in df.columns] + [c for c in df.columns if c not in first_cols]
    df = df[cols]
    df.to_csv(out_dir / "industry_projection_ablation_results.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(out_dir / "industry_projection_ablation_training_log.csv", index=False)
    if details:
        pd.concat(details, ignore_index=True).to_csv(out_dir / "industry_projection_ablation_episode_details.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Experiments, plotting, and reporting
# -----------------------------------------------------------------------------

def default_method_specs() -> List[MethodSpec]:
    # Strong fixed baselines for the final graph-aware version.
    # Important design choice: the peer benchmark is GLS/equal, not the tail graph.
    # Graph information enters through systemic-risk scores and neural message passing.
    return [
        MethodSpec("FixedSalary", weight_kind="equal", penalty_kind="none", allow_theta=False, allow_eta=False, allow_lambda=False),
        MethodSpec("LinearPnL", weight_kind="equal", penalty_kind="none", allow_theta=True, allow_eta=False, allow_lambda=False),
        MethodSpec("Linear_TailCVaRPenalty", weight_kind="equal", penalty_kind="tail_cvar", allow_theta=True, allow_eta=False, allow_lambda=True, force_positive_lambda=True),
        MethodSpec("EqualRPE", weight_kind="equal", penalty_kind="none", allow_theta=True, allow_eta=True, allow_lambda=False, force_positive_eta=True),
        MethodSpec("GLS_RPE", weight_kind="gls", penalty_kind="none", allow_theta=True, allow_eta=True, allow_lambda=False, force_positive_eta=True),
        MethodSpec("GraphRPE_tail_Diagnostic", weight_kind="tail", penalty_kind="none", allow_theta=True, allow_eta=True, allow_lambda=False, force_positive_eta=True),
        MethodSpec("EqualRPE_TailCVaRPenalty", weight_kind="equal", penalty_kind="tail_cvar", allow_theta=True, allow_eta=True, allow_lambda=True, force_positive_eta=True, force_positive_lambda=True),
        MethodSpec("GLS_RPE_TailCVaRPenalty", weight_kind="gls", penalty_kind="tail_cvar", allow_theta=True, allow_eta=True, allow_lambda=True, force_positive_eta=True, force_positive_lambda=True),
        MethodSpec("GLS_RPE_CentralityPenalty", weight_kind="gls", penalty_kind="centrality", allow_theta=True, allow_eta=True, allow_lambda=True, force_positive_eta=True, force_positive_lambda=True),
        MethodSpec("GLS_RPE_LegacyMarginalPenalty", weight_kind="gls", penalty_kind="marginal", allow_theta=True, allow_eta=True, allow_lambda=True, force_positive_eta=True, force_positive_lambda=True),
        MethodSpec("FullFixed_GLSBenchmark_GraphTailCVaR", weight_kind="gls", penalty_kind="tail_cvar", allow_theta=True, allow_eta=True, allow_lambda=True, force_positive_eta=True, force_positive_lambda=True),
    ]

def run_fixed_baselines(
    train: EpisodeSet,
    val: EpisodeSet,
    test: EpisodeSet,
    cfg: ExperimentConfig,
    out_dir: Path,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    grid_rows = []
    detail_rows = []
    best_param_rows = []

    for spec in default_method_specs():
        print(f"  Grid-searching {prefix}: {spec.name}")
        best, grid_df = grid_search_method(train, spec, cfg)
        grid_df["dataset"] = prefix
        grid_rows.append(grid_df)
        metrics, details = evaluate_fixed_method(test, spec, best, cfg, return_episode_details=True)
        metrics["dataset"] = prefix
        metrics["best_theta"] = best[0]
        metrics["best_eta"] = best[1]
        metrics["best_lambda"] = best[2]
        rows.append(metrics)
        best_param_rows.append({
            "dataset": prefix,
            "method": spec.name,
            "display_method": paper_method_name(spec.name),
            "theta": best[0],
            "eta": best[1],
            "lambda": best[2],
        })
        if details is not None:
            details["dataset"] = prefix
            detail_rows.append(details)

    results = pd.DataFrame(rows)
    grid_all = pd.concat(grid_rows, ignore_index=True) if grid_rows else pd.DataFrame()
    details_all = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    best_params = pd.DataFrame(best_param_rows)

    write_display_csv(results, out_dir / f"{prefix}_fixed_baseline_results.csv")
    write_display_csv(grid_all, out_dir / f"{prefix}_grid_search_all.csv")
    write_display_csv(best_params, out_dir / f"{prefix}_best_fixed_params.csv")
    write_display_csv(details_all, out_dir / f"{prefix}_episode_details_fixed.csv")
    return results, grid_all, details_all



def make_target_effort_frontier(train: EpisodeSet, test: EpisodeSet, cfg: ExperimentConfig, out_dir: Path, prefix: str) -> pd.DataFrame:
    """Choose each method's coefficients subject to the same minimum effort.

    This prevents a weak but overly conservative contract from winning simply
    because it induces little effort/risk. It directly supports the paper's
    risk-incentive frontier claim.
    """
    specs = [
        MethodSpec("LinearPnL", weight_kind="equal", penalty_kind="none", allow_theta=True, allow_eta=False, allow_lambda=False),
        MethodSpec("Linear_TailCVaRPenalty", weight_kind="equal", penalty_kind="tail_cvar", allow_theta=True, allow_eta=False, allow_lambda=True, force_positive_lambda=True),
        MethodSpec("EqualRPE", weight_kind="equal", penalty_kind="none", allow_theta=True, allow_eta=True, allow_lambda=False, force_positive_eta=True),
        MethodSpec("GLS_RPE", weight_kind="gls", penalty_kind="none", allow_theta=True, allow_eta=True, allow_lambda=False, force_positive_eta=True),
        MethodSpec("GraphRPE_tail_Diagnostic", weight_kind="tail", penalty_kind="none", allow_theta=True, allow_eta=True, allow_lambda=False, force_positive_eta=True),
        MethodSpec("EqualRPE_TailCVaRPenalty", weight_kind="equal", penalty_kind="tail_cvar", allow_theta=True, allow_eta=True, allow_lambda=True, force_positive_eta=True, force_positive_lambda=True),
        MethodSpec("FullFixed_GLSBenchmark_GraphTailCVaR", weight_kind="gls", penalty_kind="tail_cvar", allow_theta=True, allow_eta=True, allow_lambda=True, force_positive_eta=True, force_positive_lambda=True),
    ]
    rows = []
    for target in cfg.effort_targets:
        for sp in specs:
            _, grid_df = grid_search_method(train, sp, cfg)
            if grid_df.empty:
                continue
            feas = grid_df[grid_df["mean_effort"] >= target].copy()
            feasible = True
            if feas.empty:
                # Fall back to the closest available effort so the table is still informative.
                feasible = False
                grid_df = grid_df.copy()
                grid_df["effort_gap_abs"] = (grid_df["mean_effort"] - target).abs()
                pick = grid_df.sort_values(["effort_gap_abs", "principal_CE"], ascending=[True, False]).iloc[0]
            else:
                pick = feas.sort_values("principal_CE", ascending=False).iloc[0]
            coeff = (float(pick["theta_grid"]), float(pick["eta_grid"]), float(pick["lambda_grid"]))
            metrics, _ = evaluate_fixed_method(test, sp, coeff, cfg, return_episode_details=False)
            metrics.update({
                "dataset": prefix,
                "effort_target": float(target),
                "target_feasible_on_train": bool(feasible),
                "theta_used": coeff[0],
                "eta_used": coeff[1],
                "lambda_used": coeff[2],
            })
            rows.append(metrics)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{prefix}_target_effort_frontier.csv", index=False)
    return df


def make_ablation_table(test: EpisodeSet, cfg: ExperimentConfig, out_dir: Path, prefix: str, reference_coeffs: Tuple[float, float, float]) -> pd.DataFrame:
    # Final ablation: peer benchmark is GLS/equal; graph is used only as risk score.
    th, et, la = reference_coeffs
    specs = [
        MethodSpec("Ablation_Full_GLS_tailCVaR", weight_kind="gls", penalty_kind="tail_cvar", allow_theta=True, allow_eta=True, allow_lambda=True),
        MethodSpec("Ablation_EqualPeer_tailCVaR", weight_kind="equal", penalty_kind="tail_cvar", allow_theta=True, allow_eta=True, allow_lambda=True),
        MethodSpec("Ablation_GLS_NoPenalty", weight_kind="gls", penalty_kind="none", allow_theta=True, allow_eta=True, allow_lambda=False),
        MethodSpec("Ablation_GLS_CentralityPenalty", weight_kind="gls", penalty_kind="centrality", allow_theta=True, allow_eta=True, allow_lambda=True),
        MethodSpec("Ablation_GLS_LegacyMarginalPenalty", weight_kind="gls", penalty_kind="marginal", allow_theta=True, allow_eta=True, allow_lambda=True),
        MethodSpec("Ablation_TailPeer_tailCVaR_Diagnostic", weight_kind="tail", penalty_kind="tail_cvar", allow_theta=True, allow_eta=True, allow_lambda=True),
        MethodSpec("Ablation_RandomPeer_randomPenalty", weight_kind="random", penalty_kind="random", allow_theta=True, allow_eta=True, allow_lambda=True),
        MethodSpec("Ablation_NoRPE_GLSPenaltyOnly", weight_kind="gls", penalty_kind="tail_cvar", allow_theta=True, allow_eta=False, allow_lambda=True),
    ]
    rows = []
    for sp in specs:
        coeff = (th, et if sp.allow_eta else 0.0, la if sp.allow_lambda and sp.penalty_kind != "none" else 0.0)
        metrics, _ = evaluate_fixed_method(test, sp, coeff, cfg, return_episode_details=False)
        metrics["dataset"] = prefix
        metrics["theta_used"] = coeff[0]
        metrics["eta_used"] = coeff[1]
        metrics["lambda_used"] = coeff[2]
        rows.append(metrics)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{prefix}_ablation_results.csv", index=False)
    return df

def run_synthetic_mechanism(cfg: ExperimentConfig, out_dir: Path) -> pd.DataFrame:
    """Mechanism sanity check: eta increases crowding, lambda decreases crowding."""
    rng = np.random.default_rng(cfg.random_seed)
    N = 30
    B = 1
    A = rng.random((N, N))
    A = (A + A.T) / 2
    A[A < 0.75] = 0.0
    np.fill_diagonal(A, 0.0)
    W = row_normalize(A)
    D = np.diag(A.sum(axis=1))
    Q = D + A + 1e-3 * np.eye(N)
    degree = A.sum(axis=1)
    m = degree / (degree.mean() + 1e-8)

    # Build a one-episode EpisodeSet manually.
    es = EpisodeSet(
        name="synthetic",
        dates=pd.DatetimeIndex([pd.Timestamp("2000-01-01")]),
        symbols=[f"A{i}" for i in range(N)],
        features=np.zeros((1, N, 9)),
        W_equal=np.expand_dims(make_equal_weights(N), 0),
        W_tail=np.expand_dims(W, 0),
        W_corr=np.expand_dims(W, 0),
        W_gls=np.expand_dims(W, 0),
        W_random=np.expand_dims(random_permuted_weights(W, rng), 0),
        m_marginal=np.expand_dims(m, 0),
        m_tail_cvar=np.expand_dims(m, 0),
        m_centrality=np.expand_dims(m, 0),
        m_random=np.expand_dims(m[rng.permutation(N)], 0),
        s0=np.ones((1, N)),
        pi=np.full((1, N), cfg.pi0),
        mu=np.zeros((1, N)),
        alpha=np.full((1, N), cfg.alpha0),
        k_cost=np.full((1, N), cfg.k_cost),
        h_cost=np.full((1, N), cfg.h_cost),
        gamma=np.full((1, N), cfg.gamma_agent),
        sigma_episode=np.full((1, N), 0.04),
        varF_episode=np.array([0.04 ** 2]),
        Fsum=np.array([0.0]),
        resid_sum=np.zeros((1, N)),
        beta_noise=np.zeros((1, N)),
        Fsum_mc=np.zeros((1, max(1, int(cfg.mc_paths)))),
        resid_sum_mc=np.zeros((1, max(1, int(cfg.mc_paths)), N)),
        beta_noise_mc=np.zeros((1, max(1, int(cfg.mc_paths)), N)),
        q5_episode=np.full((1, N), -0.05),
        Q=np.expand_dims(Q, 0),
        vix=np.array([20.0]),
    )

    rows = []
    theta = 0.20
    for eta in np.linspace(0.0, 0.80, 21):
        sp = MethodSpec("eta_sweep", "tail", "none", True, True, False)
        metrics, _ = evaluate_fixed_method(es, sp, (theta, float(eta), 0.0), cfg)
        rows.append({"sweep": "eta", "theta": theta, "eta": eta, "lambda": 0.0, **metrics})
    for lambd in np.linspace(0.0, 0.12, 21):
        sp = MethodSpec("lambda_sweep", "tail", "marginal", True, True, True)
        metrics, _ = evaluate_fixed_method(es, sp, (theta, 0.50, float(lambd)), cfg)
        rows.append({"sweep": "lambda", "theta": theta, "eta": 0.50, "lambda": lambd, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "synthetic_mechanism_sweeps.csv", index=False)
    return df


def plot_results(out_dir: Path, prefix: str, results: pd.DataFrame, ablation: Optional[pd.DataFrame] = None, synthetic: Optional[pd.DataFrame] = None) -> None:
    if plt is None:
        return
    fig_dir = ensure_dir(out_dir / "figures")

    # Main CVaR bar.
    if not results.empty:
        order = results.sort_values("principal_CE", ascending=False)["method"].tolist()
        plot_df = results.set_index("method").loc[order].reset_index()
        plt.figure(figsize=(11, 5))
        plt.bar(plot_df["method"], plot_df["CVaR95_loss"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("CVaR95 of principal loss")
        plt.title(f"{prefix}: downside risk by contract")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_cvar95_bar.png", dpi=180)
        plt.close()

        # Risk-return frontier.
        plt.figure(figsize=(7, 5))
        plt.scatter(results["CVaR95_loss"], results["mean_payoff"])
        for _, r in results.iterrows():
            plt.annotate(r["method"], (r["CVaR95_loss"], r["mean_payoff"]), fontsize=8)
        plt.xlabel("CVaR95 principal loss ↓")
        plt.ylabel("Mean principal payoff ↑")
        plt.title(f"{prefix}: risk-return frontier")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_risk_return_frontier.png", dpi=180)
        plt.close()

        # Crowding vs effort.
        plt.figure(figsize=(7, 5))
        plt.scatter(results["crowding"], results["mean_effort"])
        for _, r in results.iterrows():
            plt.annotate(r["method"], (r["crowding"], r["mean_effort"]), fontsize=8)
        plt.xlabel("Crowding s'Q s / N ↓")
        plt.ylabel("Mean effort ↑")
        plt.title(f"{prefix}: incentive-risk tradeoff")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_crowding_effort.png", dpi=180)
        plt.close()

    if ablation is not None and not ablation.empty:
        plt.figure(figsize=(10, 5))
        plot_df = ablation.sort_values("principal_CE", ascending=False)
        plt.bar(plot_df["method"], plot_df["principal_CE"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Principal CE")
        plt.title(f"{prefix}: ablation")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_ablation_principal_CE.png", dpi=180)
        plt.close()

    if synthetic is not None and not synthetic.empty:
        for sweep, xcol in [("eta", "eta"), ("lambda", "lambda")]:
            sub = synthetic[synthetic["sweep"] == sweep]
            if sub.empty:
                continue
            plt.figure(figsize=(6, 4))
            plt.plot(sub[xcol], sub["crowding"], marker="o")
            plt.xlabel(xcol)
            plt.ylabel("Crowding s'Q s / N")
            title = "RPE distortion: eta increases crowding" if sweep == "eta" else "Systemic penalty correction: lambda reduces crowding"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_dir / f"synthetic_{sweep}_sweep.png", dpi=180)
            plt.close()


def summarize_data(out_dir: Path, summary: Dict[str, Any]) -> None:
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    lines = []
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    (out_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def load_all_data(data_dir: Path) -> Dict[str, Any]:
    industry_path = find_file(data_dir, ["49_Industry_Portfolios_Daily.csv", "49_Industry_Portfolios_daily", "49_industry"])
    ff3_path = find_file(data_dir, ["F-F_Research_Data_Factors_daily.csv", "F-F_Research_Data_Factors_daily", "Factors_daily"])
    ff5_path = find_file(data_dir, ["F-F_Research_Data_5_Factors_2x3_daily.csv", "5_Factors_2x3_daily"], required=False)
    mom_path = find_file(data_dir, ["F-F_Momentum_Factor_daily.csv", "Momentum_Factor_daily"], required=False)
    vix_path = find_file(data_dir, ["VIXCLS", "vixcls"], required=False)
    vxv_path = find_file(data_dir, ["VXVCLS", "vxvcls"], required=False)
    etf_path = find_file(data_dir, ["master_daily_features_macro_dailyonly", "master_daily_features"], required=False)

    industry = load_industry_returns(industry_path, section="value")
    ff3 = parse_french_factor_file(ff3_path)
    ff5 = parse_french_factor_file(ff5_path) if ff5_path is not None else None
    mom = parse_french_factor_file(mom_path) if mom_path is not None else None
    vix = load_vix_file(vix_path, "VIXCLS") if vix_path is not None else None
    vxv = load_vix_file(vxv_path, "VXVCLS") if vxv_path is not None else None

    if etf_path is not None:
        etf_prices, etf_macro = load_etf_wide_prices(etf_path)
    else:
        etf_prices, etf_macro = None, None

    return {
        "paths": {
            "industry": str(industry_path),
            "ff3": str(ff3_path),
            "ff5": str(ff5_path) if ff5_path else None,
            "mom": str(mom_path) if mom_path else None,
            "vix": str(vix_path) if vix_path else None,
            "vxv": str(vxv_path) if vxv_path else None,
            "etf": str(etf_path) if etf_path else None,
        },
        "industry": industry,
        "ff3": ff3,
        "ff5": ff5,
        "mom": mom,
        "vix": vix,
        "vxv": vxv,
        "etf_prices": etf_prices,
        "etf_macro": etf_macro,
    }


def prepare_industry_returns(data: Dict[str, Any], cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    industry = data["industry"].copy()
    ff3 = data["ff3"].copy()
    common = industry.index.intersection(ff3.index)
    industry = industry.loc[common]
    ff3 = ff3.loc[common]
    # Use excess industry returns net RF when available.
    rf = ff3["RF"] if "RF" in ff3.columns else 0.0
    returns = industry.sub(rf, axis=0)
    # Keep dates from 1990 onward, and complete cases.
    returns = returns.loc[pd.Timestamp(cfg.industry_start):]
    returns = returns.dropna(axis=1, how="any")
    factor = ff3.loc[returns.index, "Mkt-RF"]
    vix = data["vix"]["VIXCLS"] if data.get("vix") is not None else None
    return returns, factor, vix


def prepare_etf_returns(data: Dict[str, Any], cfg: ExperimentConfig) -> Optional[Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]]:
    prices = data.get("etf_prices")
    if prices is None:
        return None
    # Use 20 ETFs from 2015-10-08 onward to keep XLRE, then pct_change.
    prices = prices.loc[pd.Timestamp(cfg.etf_start):].copy()
    prices = prices.dropna(axis=1, how="any")
    returns = prices.pct_change().dropna(how="any")
    if returns.shape[1] < 6:
        return None
    ff3 = data["ff3"]
    common = returns.index.intersection(ff3.index)
    returns = returns.loc[common]
    factor = ff3.loc[common, "Mkt-RF"]
    # If RF exists, subtract it. It is tiny but keeps the units consistent.
    if "RF" in ff3.columns:
        returns = returns.sub(ff3.loc[common, "RF"], axis=0)
    vix = data["vix"]["VIXCLS"] if data.get("vix") is not None else None
    return returns, factor, vix


def run_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    tables_dir = ensure_dir(out_dir / "tables")
    ensure_dir(out_dir / "figures")

    cfg = ExperimentConfig(
        quick=args.quick,
        epochs=args.epochs,
        step=args.step if args.step is not None else ExperimentConfig.step,
        episode_horizon=args.horizon,
        window=args.window,
        random_seed=args.seed,
        mc_paths=args.mc_paths,
        neural_exact_ic=args.exact_neural_ic,
        torch_threads=args.torch_threads,
        crowding_obj_weight=args.crowding_obj_weight,
        effort_target=args.effort_target,
        effort_target_weight=args.effort_target_weight,
        graph_propagation_rho=args.graph_propagation_rho,
        hard_effort_z_min=args.hard_effort_z_min,
        hard_effort_z_cap=args.hard_effort_z_cap,
        neural_peer_benchmark=args.neural_peer_benchmark,
    )
    cfg.apply_quick()
    if TORCH_AVAILABLE and cfg.torch_threads and cfg.torch_threads > 0:
        try:
            torch.set_num_threads(int(cfg.torch_threads))
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    set_seed(cfg.random_seed)
    rng = np.random.default_rng(cfg.random_seed)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print("Loading data...")
    data = load_all_data(data_dir)
    returns, factor, vix = prepare_industry_returns(data, cfg)
    print(f"Industry returns: {returns.shape}, {returns.index.min().date()} to {returns.index.max().date()}")

    # Build all industry episodes in the full requested range, then split.
    print("Building industry rolling episodes...")
    industry_es = build_episode_set(
        name="industry49",
        returns=returns,
        factor=factor,
        vix=vix,
        cfg=cfg,
        start_date=cfg.industry_start,
        end_date=cfg.test_end,
        rng=rng,
    )
    train = industry_es.subset_by_date(cfg.industry_start, cfg.train_end, "industry_train")
    val = industry_es.subset_by_date(cfg.val_start, cfg.val_end, "industry_val")
    test = industry_es.subset_by_date(cfg.test_start, cfg.test_end, "industry_test")
    train, val, test, feat_scaler = standardize_feature_sets(train, val, test)
    print(f"Industry episodes: train={train.B}, val={val.B}, test={test.B}, N={test.N}")

    summary = {
        "data_paths": data["paths"],
        "config": dataclasses.asdict(cfg),
        "industry_returns_shape": returns.shape,
        "industry_date_range": (str(returns.index.min().date()), str(returns.index.max().date())),
        "industry_episodes": {"train": train.B, "val": val.B, "test": test.B, "N": test.N},
        "torch_available": TORCH_AVAILABLE,
    }

    # Optional reviewer-robustness-only mode. This skips fixed baselines, ETF,
    # plots, and stress summaries; it trains only the requested GraphSignal-ICL
    # robustness checks. Use this for the compact z_min and projection tables.
    if getattr(args, "robustness_only", False):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for robustness_only experiments.")
        if getattr(args, "run_zmin_sensitivity", False):
            z_values = parse_float_list(args.zmin_values)
            z_epochs = min(int(args.zmin_epochs), int(cfg.epochs)) if cfg.quick else int(args.zmin_epochs)
            run_zmin_sensitivity(train, val, test, cfg, tables_dir, z_values, epochs=z_epochs)
        if getattr(args, "run_projection_ablation", False):
            p_epochs = min(int(args.projection_epochs), int(cfg.epochs)) if cfg.quick else int(args.projection_epochs)
            run_projection_ablation(train, val, test, cfg, tables_dir, epochs=p_epochs)
        summarize_data(out_dir, summary)
        print("\nDone robustness-only run.")
        return

    # Synthetic mechanism check.
    print("Running synthetic mechanism sweeps...")
    synthetic_df = run_synthetic_mechanism(cfg, tables_dir)

    # Industry fixed baselines.
    print("Running industry fixed baselines...")
    industry_results, industry_grid, industry_details = run_fixed_baselines(train, val, test, cfg, tables_dir, prefix="industry")

    # Ablation based on full fixed graph marginal best parameters.
    full_row = industry_results[industry_results["method"] == "FullFixed_GLSBenchmark_GraphTailCVaR"]
    if len(full_row):
        ref = (float(full_row["best_theta"].iloc[0]), float(full_row["best_eta"].iloc[0]), float(full_row["best_lambda"].iloc[0]))
    else:
        ref = (0.30, 0.30, 0.035)
    industry_ablation = make_ablation_table(test, cfg, tables_dir, "industry", ref)
    print("Writing target-effort frontier table...")
    industry_effort_frontier = make_target_effort_frontier(train, test, cfg, tables_dir, "industry")

    # Neural baselines.
    neural_rows = []
    neural_logs = []
    neural_details = []
    if args.run_neural and not args.skip_neural:
        if not TORCH_AVAILABLE:
            print("PyTorch unavailable; skipping neural contracts.")
        else:
            neural_specs = [
                # Pure no-graph negative control: raw node features, equal benchmark, no graph penalty.
                dict(label="RawNodeMLP_GLS_NoGraph_HardEffort", use_graph=False, graph_kind="equal", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="none", feature_mode="raw"),
                # Graph-score-only ablation: sees the graph tail-risk score as a state feature,
                # but removes the contract penalty term by forcing lambda to zero.
                # This isolates graph-state conditioning from the systemic-penalty mechanism.
                dict(label="GraphSignal-ICL-NoPenalty", use_graph=False, graph_kind="equal", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="none", feature_mode="graph_scores"),
                # Main model GraphSignal-ICL: node-level graph-risk-signal contract learner; sees tail-CVaR graph risk score and uses it in the systemic penalty term.
                dict(label="GraphSignal-ICL", use_graph=False, graph_kind="equal", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="tail_cvar", feature_mode="graph_scores"),
                # Same-architecture random-signal control: no message passing, same decision rule,
                # but graph-risk feature columns and the explicit penalty score are randomized.
                dict(label="RandomSignal-ICL", use_graph=False, graph_kind="equal", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="random", feature_mode="random_scores"),
                # GNN ablation: raw features + true tail graph message passing + tail-CVaR penalty.
                dict(label="FullGNN_TailMessage_GLSBenchmark_HardEffort", use_graph=True, graph_kind="tail", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="tail_cvar", feature_mode="raw"),
                # Clean random negative control: raw features + permuted graph + permuted penalty score.
                dict(label="RandomGraphGNN_GLSBenchmark_HardEffort", use_graph=True, graph_kind="random", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="random", feature_mode="raw"),
            ]
            for ns in neural_specs:
                print(f"Training {ns['label']} contract...")
                m, log, det = train_neural_contract(train, val, test, cfg, tables_dir, **ns)
                m["dataset"] = "industry"
                neural_rows.append(m); neural_logs.append(log); neural_details.append(det)

    neural_results = pd.DataFrame(neural_rows)
    if not neural_results.empty:
        neural_results.to_csv(tables_dir / "industry_neural_results.csv", index=False)
        pd.concat(neural_logs, ignore_index=True).to_csv(tables_dir / "industry_neural_training_log.csv", index=False)
        pd.concat(neural_details, ignore_index=True).to_csv(tables_dir / "industry_episode_details_neural.csv", index=False)
        industry_all = pd.concat([industry_results, neural_results], ignore_index=True)
    else:
        industry_all = industry_results.copy()
    industry_all.to_csv(tables_dir / "industry_main_results_all.csv", index=False)

    # Two compact reviewer-facing robustness checks. They are intentionally
    # limited to GraphSignal-ICL, so the main experimental structure is unchanged.
    if getattr(args, "run_zmin_sensitivity", False):
        if TORCH_AVAILABLE:
            z_values = parse_float_list(args.zmin_values)
            z_epochs = min(int(args.zmin_epochs), int(cfg.epochs)) if cfg.quick else int(args.zmin_epochs)
            run_zmin_sensitivity(train, val, test, cfg, tables_dir, z_values, epochs=z_epochs)
        else:
            print("PyTorch unavailable; skipping z_min sensitivity.")
    if getattr(args, "run_projection_ablation", False):
        if TORCH_AVAILABLE:
            p_epochs = min(int(args.projection_epochs), int(cfg.epochs)) if cfg.quick else int(args.projection_epochs)
            run_projection_ablation(train, val, test, cfg, tables_dir, epochs=p_epochs)
        else:
            print("PyTorch unavailable; skipping projection ablation.")

    # ETF robustness.
    etf_results_all = pd.DataFrame()
    if not args.skip_etf:
        etf_prepared = prepare_etf_returns(data, cfg)
        if etf_prepared is not None:
            etf_returns, etf_factor, etf_vix = etf_prepared
            print(f"ETF returns: {etf_returns.shape}, {etf_returns.index.min().date()} to {etf_returns.index.max().date()}")
            try:
                etf_es = build_episode_set(
                    name="ETF",
                    returns=etf_returns,
                    factor=etf_factor,
                    vix=etf_vix,
                    cfg=cfg,
                    start_date=cfg.etf_start,
                    end_date=cfg.etf_test_end,
                    rng=rng,
                )
                etf_train = etf_es.subset_by_date(cfg.etf_start, cfg.etf_train_end, "etf_train")
                etf_val = etf_es.subset_by_date(cfg.etf_val_start, cfg.etf_val_end, "etf_val")
                etf_test = etf_es.subset_by_date(cfg.etf_test_start, cfg.etf_test_end, "etf_test")
                etf_train, etf_val, etf_test, _ = standardize_feature_sets(etf_train, etf_val, etf_test)
                print(f"ETF episodes: train={etf_train.B}, val={etf_val.B}, test={etf_test.B}, N={etf_test.N}")
                etf_results, etf_grid, etf_details = run_fixed_baselines(etf_train, etf_val, etf_test, cfg, tables_dir, prefix="etf")
                etf_results_all = etf_results
                if args.run_neural and not args.skip_neural and TORCH_AVAILABLE and etf_train.B > 5 and etf_val.B > 2:
                    print("Training ETF clean neural graph controls...")
                    etf_specs = [
                        dict(label="ETF_RawNodeMLP_GLS_NoGraph_HardEffort", use_graph=False, graph_kind="equal", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="none", feature_mode="raw"),
                        dict(label="ETF_GraphSignal-ICL-NoPenalty", use_graph=False, graph_kind="equal", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="none", feature_mode="graph_scores"),
                        dict(label="ETF_GraphSignal-ICL", use_graph=False, graph_kind="equal", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="tail_cvar", feature_mode="graph_scores"),
                        dict(label="ETF_RandomSignal-ICL", use_graph=False, graph_kind="equal", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="random", feature_mode="random_scores"),
                        dict(label="ETF_FullGNN_TailMessage_GLSBenchmark_HardEffort", use_graph=True, graph_kind="tail", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="tail_cvar", feature_mode="raw"),
                        dict(label="ETF_RandomGraphGNN_GLSBenchmark_HardEffort", use_graph=True, graph_kind="random", benchmark_kind=cfg.neural_peer_benchmark, penalty_kind="random", feature_mode="raw"),
                    ]
                    etf_neural_rows, etf_neural_logs, etf_neural_details = [], [], []
                    for ns in etf_specs:
                        print(f"Training {ns['label']} contract...")
                        etf_m, etf_log, etf_det = train_neural_contract(etf_train, etf_val, etf_test, cfg, tables_dir, **ns)
                        etf_m["dataset"] = "etf"
                        etf_neural_rows.append(etf_m)
                        etf_neural_logs.append(etf_log)
                        etf_neural_details.append(etf_det)
                    etf_neural = pd.DataFrame(etf_neural_rows)
                    etf_neural.to_csv(tables_dir / "etf_neural_results.csv", index=False)
                    pd.concat(etf_neural_logs, ignore_index=True).to_csv(tables_dir / "etf_neural_training_log.csv", index=False)
                    pd.concat(etf_neural_details, ignore_index=True).to_csv(tables_dir / "etf_episode_details_neural.csv", index=False)
                    etf_results_all = pd.concat([etf_results, etf_neural], ignore_index=True)
                etf_results_all.to_csv(tables_dir / "etf_main_results_all.csv", index=False)
                summary["etf_returns_shape"] = etf_returns.shape
                summary["etf_episodes"] = {"train": etf_train.B, "val": etf_val.B, "test": etf_test.B, "N": etf_test.N}
            except Exception as e:
                print(f"ETF robustness failed: {e}")
                summary["etf_error"] = str(e)
        else:
            print("ETF data not available or insufficient; skipping ETF robustness.")

    # Stress-period summaries for industry.
    print("Writing stress-period summaries...")
    # Use detailed rows from fixed + neural if available.
    detail_sources = [industry_details]
    if neural_details:
        detail_sources += neural_details
    detail_all = pd.concat([d for d in detail_sources if d is not None and not d.empty], ignore_index=True)
    if not detail_all.empty:
        detail_all["date"] = pd.to_datetime(detail_all["date"])
        stress_defs = {
            "GFC_2008_2009": ("2008-01-01", "2009-12-31"),
            "COVID_2020": ("2020-02-01", "2020-06-30"),
            "Rates_2022": ("2022-01-01", "2022-12-31"),
        }
        stress_rows = []
        for label, (s, e) in stress_defs.items():
            sub = detail_all[(detail_all["date"] >= pd.Timestamp(s)) & (detail_all["date"] <= pd.Timestamp(e))]
            if sub.empty:
                continue
            for method, g in sub.groupby("method"):
                pi = g["principal_payoff"].values
                stress_rows.append({
                    "stress_period": label,
                    "method": method,
                    "B": len(g),
                    "mean_payoff": float(np.mean(pi)),
                    "CVaR95_loss": cvar_np(-pi, 0.95),
                    "crowding": float(g["crowding"].mean()),
                    "co_crash_freq": float(g["co_crash"].mean()),
                    "mean_effort": float(g["mean_effort"].mean()),
                })
        # High-VIX regime is the cleanest stress subset for graph evidence.
        vix_s = pd.Series(test.vix, index=test.dates).replace([np.inf, -np.inf], np.nan).dropna()
        if not vix_s.empty:
            cut = float(np.nanquantile(vix_s.values, 0.90))
            high_dates = set(vix_s[vix_s >= cut].index)
            sub = detail_all[detail_all["date"].isin(high_dates)]
            for method, g in sub.groupby("method"):
                pi = g["principal_payoff"].values
                stress_rows.append({
                    "stress_period": "HighVIX_Top10pct",
                    "method": method,
                    "B": len(g),
                    "mean_payoff": float(np.mean(pi)),
                    "CVaR95_loss": cvar_np(-pi, 0.95),
                    "crowding": float(g["crowding"].mean()),
                    "co_crash_freq": float(g["co_crash"].mean()),
                    "mean_effort": float(g["mean_effort"].mean()),
                })
        pd.DataFrame(stress_rows).to_csv(tables_dir / "industry_stress_period_results.csv", index=False)

    # Summary is written before plotting so long plotting/font issues cannot
    # erase the run metadata. Plots are optional and can be skipped with --no_plots.
    summarize_data(out_dir, summary)

    # Figures.
    if not args.no_plots:
        try:
            plot_results(out_dir, "industry", industry_all, industry_ablation, synthetic_df)
            if not etf_results_all.empty:
                plot_results(out_dir, "etf", etf_results_all, None, None)
        except Exception as plot_error:
            print(f"Warning: plotting failed but tables were written: {plot_error}")

    print("\nDone.")
    print(f"Main result table: {tables_dir / 'industry_main_results_all.csv'}")
    print(f"Ablation table:    {tables_dir / 'industry_ablation_results.csv'}")
    print(f"Figures:           {out_dir / 'figures'}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run graph-aware IC contract experiments.")
    p.add_argument("--data_dir", type=str, default=".", help="Folder containing all data files.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output folder.")
    p.add_argument("--quick", action="store_true", help="Fast smoke-test mode with fewer episodes/grid points/epochs.")
    p.add_argument("--run_neural", action="store_true", help="Run NodeOnlyMLP and GNN contract learning.")
    p.add_argument("--skip_neural", action="store_true", help="Force skip neural models even if PyTorch is installed.")
    p.add_argument("--skip_etf", action="store_true", help="Skip ETF robustness.")
    p.add_argument("--robustness_only", action="store_true", help="Run only requested GraphSignal-ICL robustness checks after data/episode construction; skip baselines and ETF.")
    p.add_argument("--epochs", type=int, default=250, help="Neural training epochs.")
    p.add_argument("--step", type=int, default=None, help="Rolling episode step. Default 20, quick mode overrides to 60.")
    p.add_argument("--horizon", type=int, default=20, help="Episode horizon in trading days.")
    p.add_argument("--window", type=int, default=252, help="Rolling lookback window in trading days.")
    p.add_argument("--mc_paths", type=int, default=50, help="Bootstrap scenario paths per decision state for MC tail evaluation.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument("--exact_neural_ic", action="store_true", help="Use exact batched linear-system IC layer for neural contracts. Slower but paper-faithful.")
    p.add_argument("--torch_threads", type=int, default=1, help="Number of PyTorch CPU threads. Default 1 for stable Windows runs.")
    p.add_argument("--crowding_obj_weight", type=float, default=0.10, help="Penalty weight on mean model-implied crowding in policy selection/training.")
    p.add_argument("--effort_target", type=float, default=0.040, help="Minimum mean effort target for neural training; set <=0 to disable.")
    p.add_argument("--effort_target_weight", type=float, default=50.0, help="Squared hinge penalty weight for effort shortfall.")
    p.add_argument("--graph_propagation_rho", type=float, default=0.25, help="Residual shock propagation strength through true tail graph; 0 disables.")
    p.add_argument("--hard_effort_z_min", type=float, default=0.80, help="Hard lower bound on neural effective incentive z=theta+eta. With alpha0=0.05, 0.80 gives effort about 0.04.")
    p.add_argument("--hard_effort_z_cap", type=float, default=1.20, help="Upper bound on neural effective incentive z=theta+eta.")
    p.add_argument("--neural_peer_benchmark", type=str, default="gls", choices=["gls", "equal"], help="Peer benchmark used in neural contracts; graph enters through risk score/message passing, not peer weights.")
    p.add_argument("--seeds", type=str, default="", help="Comma-separated seed list for multi-seed runs, e.g. 1,2,3,4,5. If set, each seed runs in out_dir/seed_<seed> and aggregate CSVs are written.")
    p.add_argument("--run_zmin_sensitivity", action="store_true", help="Run GraphSignal-ICL z_min sensitivity robustness.")
    p.add_argument("--zmin_values", type=str, default="0.6,0.7,0.8,0.9", help="Comma-separated hard-effort z_min values for sensitivity.")
    p.add_argument("--zmin_epochs", type=int, default=150, help="Epochs for each z_min sensitivity model.")
    p.add_argument("--run_projection_ablation", action="store_true", help="Run constrained-training vs unconstrained + post-hoc projection ablation.")
    p.add_argument("--projection_epochs", type=int, default=150, help="Epochs for projection ablation models.")
    p.add_argument("--no_plots", action="store_true", help="Skip matplotlib figures and only write CSV tables/models.")
    return p



# -----------------------------------------------------------------------------
# Multi-seed wrapper and aggregation
# -----------------------------------------------------------------------------

def parse_seed_list(seed_text: str) -> List[int]:
    if not seed_text:
        return []
    out = []
    for part in seed_text.split(','):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def aggregate_table_across_seeds(base_out: Path, seeds: List[int], rel_table: str, group_cols: Sequence[str]) -> None:
    frames = []
    for seed in seeds:
        path = base_out / f"seed_{seed}" / "tables" / rel_table
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        return
    all_df = pd.concat(frames, ignore_index=True)
    stem = Path(rel_table).stem
    all_df = add_display_columns(all_df)
    write_display_csv(all_df, base_out / f"multi_seed_{stem}_all.csv")
    numeric_cols = [c for c in all_df.select_dtypes(include=[np.number]).columns if c not in set(group_cols) | {"seed"}]
    if not numeric_cols:
        return
    agg = all_df.groupby(list(group_cols))[numeric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()
    write_display_csv(agg, base_out / f"multi_seed_{stem}_summary.csv")


REVIEWER_SEED_METRICS = [
    "principal_CE",
    "policy_objective",
    "CVaR95_loss",
    "CVaR99_loss",
    "crowding",
    "co_crash_freq",
    "mean_payoff",
    "mean_effort",
    "mean_systematic_exposure",
]

REVIEWER_EPISODE_METRICS = [
    "principal_payoff",
    "principal_CVaR95_loss_episode",
    "crowding",
    "co_crash",
    "mean_effort",
    "mean_systematic_exposure",
]

REVIEWER_EPS = 1e-12

LOWER_IS_BETTER_METRICS = {
    "CVaR95_loss",
    "CVaR99_loss",
    "crowding",
    "co_crash_freq",
    "principal_CVaR95_loss_episode",
    "co_crash",
    "mean_systematic_exposure",
}

HIGHER_IS_BETTER_METRICS = {
    "principal_CE",
    "policy_objective",
    "mean_payoff",
    "principal_payoff",
    "mean_effort",
}

REVIEWER_COMPARISONS = {
    "industry": [
        ("GraphSignal-ICL", "GraphSignal-ICL-NoPenalty", "placement_vs_feature_only"),
        ("GraphSignal-ICL", "RawNodeMLP_GLS_NoGraph_HardEffort", "placement_vs_raw_node"),
        ("GraphSignal-ICL", "RandomSignal-ICL", "learned_signal_vs_random_signal"),
    ],
    "etf": [
        ("ETF_GraphSignal-ICL", "ETF_GraphSignal-ICL-NoPenalty", "placement_vs_feature_only"),
        ("ETF_GraphSignal-ICL", "ETF_RawNodeMLP_GLS_NoGraph_HardEffort", "placement_vs_raw_node"),
        ("ETF_GraphSignal-ICL", "ETF_RandomSignal-ICL", "learned_signal_vs_random_signal"),
    ],
}


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER_METRICS:
        return "lower_is_better"
    if metric in HIGHER_IS_BETTER_METRICS:
        return "higher_is_better"
    return "direction_unspecified"


def claim_support_from_ci(metric: str, ci_low: float, ci_high: float) -> str:
    direction = metric_direction(metric)
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return "insufficient_evidence"
    if direction == "lower_is_better":
        if ci_high < 0:
            return "supports_treatment"
        if ci_low > 0:
            return "supports_baseline"
        return "mixed_or_not_significant"
    if direction == "higher_is_better":
        if ci_low > 0:
            return "supports_treatment"
        if ci_high < 0:
            return "supports_baseline"
        return "mixed_or_not_significant"
    return "direction_unspecified"


def bootstrap_delta_summary(values: np.ndarray, reps: int = 5000, seed: int = 20260520) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_sign": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(max(1, reps))], dtype=float)
    p_sign = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    return {
        "delta": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_sign": float(min(1.0, p_sign)),
        "n": int(len(values)),
    }


def sign_test_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > REVIEWER_EPS]
    n = int(len(values))
    if n == 0:
        return np.nan
    n_pos = int(np.sum(values > 0.0))
    n_neg = n - n_pos
    k = min(n_pos, n_neg)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * prob))


def wilcoxon_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > REVIEWER_EPS]
    if len(values) < 2:
        return np.nan
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return sign_test_pvalue(values)


def sign_consistency(values: np.ndarray) -> Dict[str, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n_positive": int(np.sum(values > REVIEWER_EPS)),
        "n_negative": int(np.sum(values < -REVIEWER_EPS)),
        "n_zero": int(np.sum(np.abs(values) <= REVIEWER_EPS)),
    }


def add_bh_correction(df: pd.DataFrame, p_col: str = "wilcoxon_p", alpha: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    out["bh_p"] = np.nan
    out["bh_reject_05"] = False
    if out.empty or p_col not in out.columns:
        return out
    p = pd.to_numeric(out[p_col], errors="coerce")
    valid = p.notna() & np.isfinite(p)
    if not valid.any():
        return out
    idx = out.index[valid].to_numpy()
    pvals = p.loc[idx].to_numpy(dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    adj_full = np.empty_like(adj)
    adj_full[order] = adj
    out.loc[idx, "bh_p"] = adj_full
    out.loc[idx, "bh_reject_05"] = adj_full <= alpha
    return out


def read_seed_level_results(base_out: Path, seeds: Sequence[int], dataset: str) -> pd.DataFrame:
    rel = "industry_neural_results.csv" if dataset == "industry" else "etf_neural_results.csv"
    rows = []
    for seed in seeds:
        path = base_out / f"seed_{int(seed)}" / "tables" / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["seed"] = int(seed)
        df["dataset"] = dataset
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def read_episode_level_results(base_out: Path, seeds: Sequence[int], dataset: str) -> pd.DataFrame:
    rel = "industry_episode_details_neural.csv" if dataset == "industry" else "etf_episode_details_neural.csv"
    rows = []
    for seed in seeds:
        path = base_out / f"seed_{int(seed)}" / "tables" / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["seed"] = int(seed)
        df["dataset"] = dataset
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_seed_significance_table(base_out: Path, seeds: Sequence[int], reps: int = 10000) -> pd.DataFrame:
    out_rows = []
    for dataset, comparisons in REVIEWER_COMPARISONS.items():
        df = read_seed_level_results(base_out, seeds, dataset)
        if df.empty:
            continue
        for treatment, baseline, comparison_type in comparisons:
            comparison = f"{treatment} minus {baseline}"
            for metric in REVIEWER_SEED_METRICS:
                if metric not in df.columns:
                    continue
                pivot = df.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
                if treatment not in pivot.columns or baseline not in pivot.columns:
                    continue
                pair = pivot[[treatment, baseline]].dropna()
                deltas = pair[treatment].to_numpy(dtype=float) - pair[baseline].to_numpy(dtype=float)
                summary = bootstrap_delta_summary(deltas, reps=reps, seed=20260520 + len(out_rows))
                signs = sign_consistency(deltas)
                mean_treatment = float(pair[treatment].mean()) if not pair.empty else np.nan
                mean_baseline = float(pair[baseline].mean()) if not pair.empty else np.nan
                out_rows.append(
                    {
                        "dataset": dataset,
                        "comparison_type": comparison_type,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_treatment": mean_treatment,
                        "mean_baseline": mean_baseline,
                        "delta": summary["delta"],
                        "delta_pct": float(summary["delta"] / (abs(mean_baseline) + REVIEWER_EPS)) if np.isfinite(mean_baseline) else np.nan,
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "p_sign": summary["p_sign"],
                        "wilcoxon_p": wilcoxon_pvalue(deltas),
                        "sign_test_p": sign_test_pvalue(deltas),
                        **signs,
                        "n_seeds": summary["n"],
                        "direction": metric_direction(metric),
                        "claim_support": claim_support_from_ci(metric, summary["ci_low"], summary["ci_high"]),
                    }
                )
    columns = [
        "dataset", "comparison_type", "comparison", "display_comparison", "metric", "mean_treatment",
        "mean_baseline", "delta", "delta_pct", "ci_low", "ci_high", "p_sign",
        "wilcoxon_p", "sign_test_p", "n_positive", "n_negative", "n_zero",
        "n_seeds", "direction", "claim_support", "bh_p", "bh_reject_05",
    ]
    out = add_bh_correction(pd.DataFrame(out_rows), p_col="wilcoxon_p")
    out = add_display_columns(out)
    out = out.reindex(columns=columns)
    write_display_csv(out, base_out / "multi_seed_paired_significance.csv")
    return out


def episode_bootstrap_table(base_out: Path, seeds: Sequence[int], reps: int = 5000) -> pd.DataFrame:
    out_rows = []
    for dataset, comparisons in REVIEWER_COMPARISONS.items():
        df = read_episode_level_results(base_out, seeds, dataset)
        if df.empty or "date" not in df.columns:
            continue
        for treatment, baseline, comparison_type in comparisons:
            comparison = f"{treatment} minus {baseline}"
            for metric in REVIEWER_EPISODE_METRICS:
                if metric not in df.columns:
                    continue
                pivot = df.pivot_table(index=["seed", "date"], columns="method", values=metric, aggfunc="first")
                if treatment not in pivot.columns or baseline not in pivot.columns:
                    continue
                pair = pivot[[treatment, baseline]].dropna()
                deltas = pair[treatment].to_numpy(dtype=float) - pair[baseline].to_numpy(dtype=float)
                summary = bootstrap_delta_summary(deltas, reps=reps, seed=20260521 + len(out_rows))
                mean_treatment = float(pair[treatment].mean()) if not pair.empty else np.nan
                mean_baseline = float(pair[baseline].mean()) if not pair.empty else np.nan
                out_rows.append(
                    {
                        "dataset": dataset,
                        "comparison_type": comparison_type,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_treatment": mean_treatment,
                        "mean_baseline": mean_baseline,
                        "delta": summary["delta"],
                        "delta_pct": float(summary["delta"] / (abs(mean_baseline) + REVIEWER_EPS)) if np.isfinite(mean_baseline) else np.nan,
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "p_two_sided": summary["p_sign"],
                        "n_pairs": summary["n"],
                        "direction": metric_direction(metric),
                        "claim_support": claim_support_from_ci(metric, summary["ci_low"], summary["ci_high"]),
                    }
                )
    columns = [
        "dataset", "comparison_type", "comparison", "display_comparison", "metric", "mean_treatment",
        "mean_baseline", "delta", "delta_pct", "ci_low", "ci_high", "p_two_sided",
        "n_pairs", "direction", "claim_support",
    ]
    out = pd.DataFrame(out_rows, columns=columns)
    out = add_display_columns(out)
    out = out.reindex(columns=columns)
    write_display_csv(out, base_out / "multi_seed_episode_bootstrap.csv")
    return out


def cluster_episode_bootstrap_table(base_out: Path, seeds: Sequence[int], reps: int = 5000) -> pd.DataFrame:
    out_rows = []
    for dataset, comparisons in REVIEWER_COMPARISONS.items():
        df = read_episode_level_results(base_out, seeds, dataset)
        if df.empty or "date" not in df.columns:
            continue
        for treatment, baseline, comparison_type in comparisons:
            comparison = f"{treatment} minus {baseline}"
            for metric in REVIEWER_EPISODE_METRICS:
                if metric not in df.columns:
                    continue
                pivot = df.pivot_table(index=["seed", "date"], columns="method", values=metric, aggfunc="first")
                if treatment not in pivot.columns or baseline not in pivot.columns:
                    continue
                pair = pivot[[treatment, baseline]].dropna().reset_index()
                if pair.empty:
                    continue
                pair["delta"] = pair[treatment].astype(float) - pair[baseline].astype(float)
                seed_delta = pair.groupby("seed", sort=False)["delta"].mean()
                deltas = seed_delta.to_numpy(dtype=float)
                summary = bootstrap_delta_summary(deltas, reps=reps, seed=20260522 + len(out_rows))
                signs = sign_consistency(deltas)
                mean_treatment = float(pair[treatment].mean())
                mean_baseline = float(pair[baseline].mean())
                out_rows.append(
                    {
                        "dataset": dataset,
                        "comparison_type": comparison_type,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_treatment": mean_treatment,
                        "mean_baseline": mean_baseline,
                        "delta": summary["delta"],
                        "delta_pct": float(summary["delta"] / (abs(mean_baseline) + REVIEWER_EPS)) if np.isfinite(mean_baseline) else np.nan,
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "p_two_sided": summary["p_sign"],
                        "wilcoxon_p": wilcoxon_pvalue(deltas),
                        "sign_test_p": sign_test_pvalue(deltas),
                        **signs,
                        "n_seed_clusters": int(len(seed_delta)),
                        "n_episode_pairs": int(len(pair)),
                        "cluster_unit": "seed",
                        "direction": metric_direction(metric),
                        "claim_support": claim_support_from_ci(metric, summary["ci_low"], summary["ci_high"]),
                    }
                )
    columns = [
        "dataset", "comparison_type", "comparison", "display_comparison", "metric", "mean_treatment",
        "mean_baseline", "delta", "delta_pct", "ci_low", "ci_high", "p_two_sided",
        "wilcoxon_p", "sign_test_p", "n_positive", "n_negative", "n_zero",
        "n_seed_clusters", "n_episode_pairs", "cluster_unit", "direction",
        "claim_support", "bh_p", "bh_reject_05",
    ]
    out = add_bh_correction(pd.DataFrame(out_rows), p_col="wilcoxon_p")
    out = add_display_columns(out)
    out = out.reindex(columns=columns)
    write_display_csv(out, base_out / "multi_seed_cluster_episode_bootstrap.csv")
    return out


def random_signal_boundary_table(paired: pd.DataFrame, base_out: Path) -> pd.DataFrame:
    columns = [
        "dataset", "metric", "placement_delta", "placement_ci_low", "placement_ci_high",
        "placement_supported", "learned_signal_delta", "learned_signal_ci_low",
        "learned_signal_ci_high", "learned_signal_supported", "random_control",
        "display_random_control", "interpretation",
    ]
    rows = []
    if paired.empty:
        out = pd.DataFrame(columns=columns)
        write_display_csv(out, base_out / "multi_seed_random_signal_boundary.csv")
        return out
    for dataset in ["industry", "etf"]:
        for metric in ["CVaR95_loss", "CVaR99_loss", "crowding", "co_crash_freq", "policy_objective", "principal_CE"]:
            placement = paired[
                (paired["dataset"] == dataset)
                & (paired["comparison_type"] == "placement_vs_feature_only")
                & (paired["metric"] == metric)
            ]
            learned = paired[
                (paired["dataset"] == dataset)
                & (paired["comparison_type"] == "learned_signal_vs_random_signal")
                & (paired["metric"] == metric)
            ]
            if placement.empty and learned.empty:
                continue
            placement_support = str(placement.iloc[0]["claim_support"]) if not placement.empty else "missing"
            learned_support = str(learned.iloc[0]["claim_support"]) if not learned.empty else "missing"
            placement_supported = placement_support == "supports_treatment"
            learned_signal_supported = learned_support == "supports_treatment"
            if placement_supported and learned_signal_supported:
                interpretation = "placement_and_learned_signal_supported"
            elif placement_supported and not learned_signal_supported:
                interpretation = "placement_supported_random_signal_boundary"
            elif not placement_supported and learned_signal_supported:
                interpretation = "learned_signal_only_mixed_placement"
            else:
                interpretation = "not_supported_or_mixed"
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "placement_delta": float(placement.iloc[0]["delta"]) if not placement.empty else np.nan,
                    "placement_ci_low": float(placement.iloc[0]["ci_low"]) if not placement.empty else np.nan,
                    "placement_ci_high": float(placement.iloc[0]["ci_high"]) if not placement.empty else np.nan,
                    "placement_supported": bool(placement_supported),
                    "learned_signal_delta": float(learned.iloc[0]["delta"]) if not learned.empty else np.nan,
                    "learned_signal_ci_low": float(learned.iloc[0]["ci_low"]) if not learned.empty else np.nan,
                    "learned_signal_ci_high": float(learned.iloc[0]["ci_high"]) if not learned.empty else np.nan,
                    "learned_signal_supported": bool(learned_signal_supported),
                    "random_control": "RandomSignal-ICL",
                    "interpretation": interpretation,
                }
            )
    out = pd.DataFrame(rows)
    out = add_display_columns(out).reindex(columns=columns)
    write_display_csv(out, base_out / "multi_seed_random_signal_boundary.csv")
    return out


def multiple_testing_table(paired: pd.DataFrame, clustered: pd.DataFrame, base_out: Path) -> pd.DataFrame:
    frames = []
    for name, df in [("seed_level", paired), ("cluster_episode", clustered)]:
        if df.empty:
            continue
        keep = [
            c
            for c in [
                "dataset",
                "comparison_type",
                "comparison",
                "metric",
                "wilcoxon_p",
                "sign_test_p",
                "bh_p",
                "bh_reject_05",
                "claim_support",
            ]
            if c in df.columns
        ]
        tmp = df[keep].copy()
        tmp.insert(0, "source_table", name)
        frames.append(tmp)
    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "source_table",
                "dataset",
                "comparison_type",
                "comparison",
                "metric",
                "wilcoxon_p",
                "sign_test_p",
                "bh_p",
                "bh_reject_05",
                "claim_support",
            ]
        )
    )
    out = add_display_columns(out)
    write_display_csv(out, base_out / "multi_seed_multiple_testing.csv")
    return out


def method_design_audit_table(base_out: Path, seeds: Sequence[int]) -> pd.DataFrame:
    metadata_cols = [
        "feature_mode",
        "message_graph_kind",
        "benchmark_kind",
        "penalty_kind",
        "constrained_during_training",
        "posthoc_projection",
    ]
    rows = []
    for dataset in ["industry", "etf"]:
        df = read_seed_level_results(base_out, seeds, dataset)
        if df.empty:
            continue
        keep = ["dataset", "method"] + [c for c in metadata_cols if c in df.columns]
        if len(keep) > 2:
            rows.append(df[keep].drop_duplicates())
    out = (
        pd.concat(rows, ignore_index=True).drop_duplicates()
        if rows
        else pd.DataFrame(columns=["dataset", "method"] + metadata_cols)
    )
    write_display_csv(out, base_out / "reviewer_method_design_audit.csv")
    return out


def deprecated_random_graph_boundary_note(base_out: Path) -> pd.DataFrame:
    out = pd.DataFrame(
        [
            {
                "status": "deprecated_not_clean_identification",
                "reason": "RandomGraphGNN changes message passing, feature mode, and penalty score; use multi_seed_random_signal_boundary.csv for same-architecture random-score evidence.",
                "replacement": "multi_seed_random_signal_boundary.csv",
            }
        ]
    )
    write_display_csv(out, base_out / "multi_seed_random_graph_boundary.csv")
    return out


def claim_diagnostics_table(paired: pd.DataFrame, boundary: pd.DataFrame, base_out: Path) -> pd.DataFrame:
    def supported(dataset: str, comparison_type: str, metrics: Sequence[str], min_count: int = 1) -> bool:
        if paired.empty or "dataset" not in paired.columns:
            return False
        sub = paired[
            (paired["dataset"] == dataset)
            & (paired["comparison_type"] == comparison_type)
            & (paired["metric"].isin(metrics))
        ]
        return int((sub["claim_support"] == "supports_treatment").sum()) >= min_count if not sub.empty else False

    risk_metrics = ["CVaR95_loss", "CVaR99_loss", "crowding", "co_crash_freq"]
    etf_boundary = boundary[(boundary["dataset"] == "etf") & (boundary["metric"].isin(risk_metrics))] if "dataset" in boundary.columns else pd.DataFrame()
    random_signal_available = bool(boundary.get("learned_signal_delta", pd.Series(dtype=float)).notna().any()) if not boundary.empty else False
    etf_learned_all = bool((etf_boundary["learned_signal_supported"] == True).all()) if not etf_boundary.empty and "learned_signal_supported" in etf_boundary.columns else False
    rows = [
        {
            "claim": "Explicit risk-penalty placement improves downside/systemic metrics versus graph-score feature-only input.",
            "required_evidence": "The graph-penalty contract beats the feature-only contract on CVaR/crowding/co-crash paired comparisons.",
            "status": "supported" if supported("industry", "placement_vs_feature_only", risk_metrics, min_count=2) else "not_supported_or_mixed",
            "recommended_wording": "Explicit risk-signal penalties yield small but statistically stable reductions in downside and crowding metrics.",
        },
        {
            "claim": "Learned graph-risk scores dominate randomized same-architecture risk scores in all universes.",
            "required_evidence": "The graph-penalty contract beats the randomized-score contract in both industry and ETF settings.",
            "status": "missing_random_signal_cell" if not random_signal_available else ("not_supported" if not etf_learned_all else "supported"),
            "recommended_wording": "Use the randomized-score control before claiming a learned-score boundary.",
        },
        {
            "claim": "The graph-penalty contract improves certainty-equivalent/principal CE.",
            "required_evidence": "Positive principal_CE delta with CI excluding zero.",
            "status": "supported" if supported("industry", "placement_vs_feature_only", ["principal_CE"], min_count=1) else "not_supported_or_mixed",
            "recommended_wording": "The main effect is downside/systemic-risk control, not CE improvement.",
        },
        {
            "claim": "The contribution is distinct from generic decision-focused learning.",
            "required_evidence": "Paper positions the design as signal placement in a partially specified constrained decision layer.",
            "status": "writing_required",
            "recommended_wording": "We study whether risk signals enter as ordinary neural features or explicit semantic penalty terms, not a new differentiable optimizer.",
        },
        {
            "claim": "The penalty is externally auditable in a real-world regulatory sense.",
            "required_evidence": "External audit protocol and verified contract settlement data.",
            "status": "avoid_claim",
            "recommended_wording": "Use explicit, interpretable, or contractible risk penalty instead of an auditability claim.",
        },
    ]
    out = pd.DataFrame(rows)
    write_display_csv(out, base_out / "claim_diagnostics.csv")
    return out


def run_reviewer_facing_diagnostics(base_out: Path, seeds: Sequence[int], seed_reps: int = 10000, episode_reps: int = 5000) -> Dict[str, pd.DataFrame]:
    base_out = Path(base_out)
    paired = paired_seed_significance_table(base_out, seeds, reps=seed_reps)
    episode = episode_bootstrap_table(base_out, seeds, reps=episode_reps)
    clustered = cluster_episode_bootstrap_table(base_out, seeds, reps=episode_reps)
    boundary = random_signal_boundary_table(paired, base_out)
    multiple = multiple_testing_table(paired, clustered, base_out)
    audit = method_design_audit_table(base_out, seeds)
    deprecated = deprecated_random_graph_boundary_note(base_out)
    claims = claim_diagnostics_table(paired, boundary, base_out)
    return {
        "paired_significance": paired,
        "episode_bootstrap": episode,
        "cluster_episode_bootstrap": clustered,
        "random_signal_boundary": boundary,
        "deprecated_random_graph_boundary": deprecated,
        "multiple_testing": multiple,
        "method_design_audit": audit,
        "claim_diagnostics": claims,
    }


def aggregate_multi_seed_outputs(base_out: Path, seeds: List[int]) -> None:
    aggregate_table_across_seeds(base_out, seeds, "industry_main_results_all.csv", ["method"])
    aggregate_table_across_seeds(base_out, seeds, "industry_neural_results.csv", ["method"])
    aggregate_table_across_seeds(base_out, seeds, "industry_ablation_results.csv", ["method"])
    aggregate_table_across_seeds(base_out, seeds, "industry_target_effort_frontier.csv", ["method", "effort_target"])
    aggregate_table_across_seeds(base_out, seeds, "industry_stress_period_results.csv", ["stress_period", "method"])
    aggregate_table_across_seeds(base_out, seeds, "etf_main_results_all.csv", ["method"])
    aggregate_table_across_seeds(base_out, seeds, "etf_neural_results.csv", ["method"])
    aggregate_table_across_seeds(base_out, seeds, "industry_zmin_sensitivity.csv", ["z_min", "method"])
    aggregate_table_across_seeds(base_out, seeds, "industry_projection_ablation_results.csv", ["method"])
    run_reviewer_facing_diagnostics(base_out, seeds)


def run_multi_seed(args: argparse.Namespace) -> None:
    seeds = parse_seed_list(args.seeds)
    if not seeds:
        run_pipeline(args)
        return
    base_out = ensure_dir(Path(args.out_dir).expanduser().resolve())
    print(f"Running multi-seed experiment with seeds={seeds}")
    for seed in seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed)
        seed_args.seeds = ""
        seed_args.out_dir = str(base_out / f"seed_{seed}")
        # Avoid spending time on figures for every seed unless the user explicitly
        # wants them. The aggregated CSVs are the important evidence.
        if not getattr(args, "no_plots", False):
            seed_args.no_plots = True
        print("\n" + "=" * 80)
        print(f"Starting seed {seed}: output -> {seed_args.out_dir}")
        print("=" * 80)
        run_pipeline(seed_args)
    aggregate_multi_seed_outputs(base_out, seeds)
    (base_out / "multi_seed_run_summary.txt").write_text(
        "seeds=" + ",".join(map(str, seeds)) + "\n" +
        "Aggregated summaries written as multi_seed_*_summary.csv\n",
        encoding="utf-8",
    )
    print(f"Multi-seed aggregation written under {base_out}")

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        if getattr(args, "seeds", ""):
            run_multi_seed(args)
        else:
            run_pipeline(args)
        if plt is not None:
            plt.close("all")
        sys.stdout.flush()
        sys.stderr.flush()
        # Some Windows/PyTorch CPU builds leave worker threads alive after printing Done.
        # All outputs are written before this point, so force a clean process exit.
        os._exit(0)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
