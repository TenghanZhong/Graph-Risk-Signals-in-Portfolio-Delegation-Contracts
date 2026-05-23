from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def summarize_bins(df: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    seed_rows = []
    for (dataset, score_bin, seed), group in df.groupby(["dataset", "score_bin", "seed"], observed=True):
        seed_rows.append(
            {
                "dataset": dataset,
                "score_bin": str(score_bin),
                "seed": int(seed),
                "mean_delta": float(group[delta_col].mean()),
                "n": int(len(group)),
            }
        )
    seed_df = pd.DataFrame(seed_rows)

    rows = []
    for (dataset, score_bin), group in seed_df.groupby(["dataset", "score_bin"], observed=True):
        values = group["mean_delta"].to_numpy(float)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / (len(values) ** 0.5)) if len(values) > 1 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "score_bin": score_bin,
                "mean_delta": mean,
                "ci_low": mean - 1.96 * se,
                "ci_high": mean + 1.96 * se,
                "seed_count": int(len(values)),
                "node_episode_count": int(group["n"].sum()),
            }
        )

    out = pd.DataFrame(rows)
    dataset_order = {"industry": 0, "etf": 1}
    out["dataset_order"] = out["dataset"].map(dataset_order)
    out["bin_order"] = out["score_bin"].str.extract(r"Q(\d+)").astype(int)
    out = out.sort_values(["dataset_order", "bin_order"])
    return out.drop(columns=["dataset_order", "bin_order"])


def write_summary(summary: pd.DataFrame, name: str) -> None:
    result_path = ROOT / "results" / name
    paper_path = ROOT / "paper_values" / name
    summary.to_csv(result_path, index=False)
    summary.to_csv(paper_path, index=False)


def plot_summary(summary: pd.DataFrame, output_stem: str) -> None:
    import matplotlib.pyplot as plt

    datasets = [d for d in ["industry", "etf"] if d in set(summary["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(8.2, 3.1), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    colors = {"industry": "#2F6B9A", "etf": "#B45F3C"}
    for ax, dataset in zip(axes, datasets):
        sub = summary[summary["dataset"] == dataset].copy()
        sub["x"] = sub["score_bin"].str.extract(r"Q(\d+)").astype(int)
        sub = sub.sort_values("x")
        y = sub["mean_delta"].to_numpy(float)
        yerr = [
            y - sub["ci_low"].to_numpy(float),
            sub["ci_high"].to_numpy(float) - y,
        ]
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.errorbar(
            sub["x"],
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors.get(dataset, "#444444"),
        )
        ax.set_title("(a) Industry" if dataset == "industry" else "(b) ETF")
        ax.set_xlabel("Graph-risk score quintile")
        ax.set_xticks(sub["x"])
        ax.set_xticklabels(sub["score_bin"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)

    axes[0].set_ylabel("Delta exposure")
    fig.tight_layout()
    for suffix, kwargs in {
        ".pdf": {},
        ".png": {"dpi": 300},
    }.items():
        fig.savefig(ROOT / "figures" / f"{output_stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    node_path = ROOT / "results" / "mechanism_node_exposure_data.csv.gz"
    node_df = pd.read_csv(node_path)

    exposure = summarize_bins(node_df, "delta_graph_minus_feature")
    write_summary(exposure, "mechanism_exposure_bins.csv")
    plot_summary(exposure, "fig2_node_exposure_mechanism")

    channel = summarize_bins(node_df, "delta_penalty_channel")
    write_summary(channel, "mechanism_penalty_channel_bins.csv")
    plot_summary(channel, "figS_penalty_channel_mechanism")

    print("Wrote node-exposure mechanism figures and summaries.")


if __name__ == "__main__":
    main()
