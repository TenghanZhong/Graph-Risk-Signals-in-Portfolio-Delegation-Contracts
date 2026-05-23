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
                "method_id": METHOD_IDS.get(short, short),
                "display_method": METHOD_LABELS.get(short, row.get("display_method", short)),
                "crowding_mean": row["crowding_mean"],
                "CVaR95_loss_mean": row["CVaR95_loss_mean"],
                "crowding_std": row["crowding_std"],
                "CVaR95_loss_std": row["CVaR95_loss_std"],
                "order": row["order"],
            }
        )
    return sorted(cleaned, key=lambda r: int(float(r["order"])))


def write_clean_figure_points(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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


def make_risk_frontier(input_csv: Path, output_pdf: Path, output_png: Path | None = None) -> list[dict[str, str]]:
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

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    markers = ["o", "s", "^", "D"]
    colors = ["#555555", "#1f77b4", "#2ca02c", "#d62728"]
    for row, marker, color in zip(rows, markers, colors):
        x = float(row["crowding_mean"])
        y = float(row["CVaR95_loss_mean"])
        ax.scatter(
            x,
            y,
            s=70,
            marker=marker,
            color=color,
            label=row["display_method"],
            zorder=3,
        )
        ax.annotate(
            row["display_method"],
            (x, y),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Crowding")
    ax.set_ylabel("CVaR95 loss")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_pdf)
    if output_png is not None:
        fig.savefig(output_png, dpi=200)
    plt.close(fig)
    return rows
