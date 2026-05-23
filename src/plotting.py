from __future__ import annotations

from pathlib import Path
import csv
import shutil


METHOD_IDS = {
    "RawNode": "no_graph_neural",
    "GS-NoPenalty": "feature_only",
    "GraphSignal-ICL": "graph_penalty",
    "RandomSignal-ICL": "randomized_score",
}

METHOD_LABELS = {
    "RawNode": "No-graph neural baseline",
    "GS-NoPenalty": "Feature-only contract",
    "GraphSignal-ICL": "Graph-penalty contract",
    "RandomSignal-ICL": "Randomized-score contract",
}


def clean_figure_points(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cleaned = []
    for row in rows:
        short = row.get("short_method", "")
        cleaned.append(
            {
                "dataset": row.get("dataset", "Industry"),
                "method_id": METHOD_IDS.get(short, short),
                "display_method": METHOD_LABELS.get(short, row.get("display_method", short)),
                "crowding_mean": row["crowding_mean"],
                "CVaR95_loss_mean": row["CVaR95_loss_mean"],
                "crowding_std": row["crowding_std"],
                "CVaR95_loss_std": row["CVaR95_loss_std"],
                "order": row["order"],
            }
        )
    dataset_order = {"Industry": 0, "ETF": 1}
    return sorted(cleaned, key=lambda r: (dataset_order.get(r["dataset"], 99), int(float(r["order"]))))


def write_clean_figure_points(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method_id",
        "display_method",
        "crowding_mean",
        "CVaR95_loss_mean",
        "crowding_std",
        "CVaR95_loss_std",
        "order",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_risk_frontier(
    input_csv: Path,
    output_pdf: Path,
    output_png: Path | None = None,
    output_svg: Path | None = None,
) -> list[dict[str, str]]:
    rows = clean_figure_points(input_csv)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if not output_pdf.exists():
            raise
        if output_png is not None and not output_png.exists():
            pass
        return rows

    datasets = []
    for row in rows:
        dataset = row.get("dataset", "Industry")
        if dataset not in datasets:
            datasets.append(dataset)

    fig, axes = plt.subplots(1, len(datasets), figsize=(8.6, 3.6), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    markers = ["o", "s", "^", "D"]
    colors = ["#555555", "#1f77b4", "#2ca02c", "#d62728"]
    for panel_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        panel_rows = sorted(
            [row for row in rows if row.get("dataset", "Industry") == dataset],
            key=lambda r: int(float(r["order"])),
        )
        for row, marker, color in zip(panel_rows, markers, colors):
            x = float(row["crowding_mean"])
            y = float(row["CVaR95_loss_mean"])
            ax.scatter(
                x,
                y,
                s=58,
                marker=marker,
                color=color,
                label=row["display_method"],
                zorder=3,
            )
        ax.set_title(f"({chr(97 + panel_idx)}) {dataset}")
        ax.set_xlabel("Crowding")
        ax.set_ylabel("CVaR95 loss")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        if panel_idx == len(datasets) - 1:
            ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_pdf)
    if output_png is not None:
        fig.savefig(output_png, dpi=200)
    if output_svg is not None:
        fig.savefig(output_svg)
    plt.close(fig)
    return rows
