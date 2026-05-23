from __future__ import annotations

from pathlib import Path

from src.plotting import make_risk_frontier, write_clean_figure_points


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    input_csv = ROOT / "results" / "figure1_points.csv"
    paper_csv = ROOT / "paper_values" / "figure1_points.csv"
    rows = make_risk_frontier(
        input_csv=input_csv,
        output_pdf=ROOT / "figures" / "fig1_risk_frontier_core.pdf",
        output_png=ROOT / "figures" / "fig1_risk_frontier_core.png",
    )
    write_clean_figure_points(rows, paper_csv)
    print("Wrote risk-frontier figure and clean figure-point CSV.")


if __name__ == "__main__":
    main()
