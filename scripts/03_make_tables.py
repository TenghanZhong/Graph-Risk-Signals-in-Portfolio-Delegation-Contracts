from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_VALUES = ROOT / "paper_values"

METRICS = [
    ("principal_CVaR95_loss_episode", "CVaR95 loss"),
    ("crowding", "Crowding"),
    ("co_crash", "Co-crash"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows: list[dict[str, str]], dataset: str, comparison_type: str, metric: str) -> dict[str, str]:
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["comparison_type"] == comparison_type
            and row["metric"] == metric
        ):
            return row
    raise KeyError(f"Missing row: {dataset}, {comparison_type}, {metric}")


def make_tables() -> None:
    PAPER_VALUES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    clustered = read_rows(RESULTS / "multi_seed_cluster_episode_bootstrap.csv")

    placement_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for dataset_label, dataset in [("Industry", "industry"), ("ETF", "etf")]:
        for metric, label in METRICS:
            placement = find_row(clustered, dataset, "placement_vs_feature_only", metric)
            quality = find_row(clustered, dataset, "learned_signal_vs_random_signal", metric)
            placement_rows.append(
                {
                    "dataset": dataset_label,
                    "metric": label,
                    "delta": float(placement["delta"]),
                    "relative_delta_pct": 100.0 * float(placement["delta_pct"]),
                    "ci_low": float(placement["ci_low"]),
                    "ci_high": float(placement["ci_high"]),
                }
            )
            diagnostic_rows.append(
                {
                    "dataset": dataset_label,
                    "metric": label,
                    "placement_delta": float(placement["delta"]),
                    "quality_delta": float(quality["delta"]),
                }
            )

    effort_rows: list[dict[str, object]] = []
    for dataset_label, dataset in [("Industry", "industry"), ("ETF", "etf")]:
        effort = find_row(clustered, dataset, "placement_vs_feature_only", "mean_effort")
        effort_rows.append(
            {
                "dataset": dataset_label,
                "mean_effort_delta": float(effort["delta"]),
                "ci_low": float(effort["ci_low"]),
                "ci_high": float(effort["ci_high"]),
            }
        )

    placement_fields = ["dataset", "metric", "delta", "relative_delta_pct", "ci_low", "ci_high"]
    diagnostic_fields = ["dataset", "metric", "placement_delta", "quality_delta"]
    effort_fields = ["dataset", "mean_effort_delta", "ci_low", "ci_high"]

    write_rows(RESULTS / "placement_table.csv", placement_rows, placement_fields)
    write_rows(RESULTS / "diagnostic_table.csv", diagnostic_rows, diagnostic_fields)
    write_rows(RESULTS / "effort_checks.csv", effort_rows, effort_fields)
    write_rows(PAPER_VALUES / "table1_values.csv", placement_rows, placement_fields)
    write_rows(PAPER_VALUES / "table2_values.csv", diagnostic_rows, diagnostic_fields)

    caption = {
        "industry_crowding_reduction_approx": 0.047,
        "industry_cvar95_loss_reduction_approx": 0.0075,
        "source_table": "table1_values.csv",
    }
    (PAPER_VALUES / "caption_numbers.json").write_text(json.dumps(caption, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [
            "results/multi_seed_cluster_episode_bootstrap.csv",
            "results/multi_seed_random_signal_boundary.csv",
            "results/multi_seed_paired_significance.csv",
            "results/multi_seed_multiple_testing.csv",
        ],
        "generated_files": [
            "results/placement_table.csv",
            "results/diagnostic_table.csv",
            "results/effort_checks.csv",
            "paper_values/table1_values.csv",
            "paper_values/table2_values.csv",
            "paper_values/caption_numbers.json",
        ],
        "note": "CVaR deltas use seed-clustered episode-level evaluation.",
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Wrote placement, diagnostic, effort, and paper-value tables.")


if __name__ == "__main__":
    make_tables()
