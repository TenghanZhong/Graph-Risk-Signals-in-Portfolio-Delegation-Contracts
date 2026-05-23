from __future__ import annotations

from pathlib import Path
import csv
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "requirements.txt",
    "src/contract_experiments.py",
    "results/placement_table.csv",
    "results/diagnostic_table.csv",
    "paper_values/table1_values.csv",
    "paper_values/table2_values.csv",
    "paper_values/caption_numbers.json",
    "figures/fig1_risk_frontier_core.pdf",
]

FORBIDDEN = [
    "ChatGPT",
    "GPT-5",
    "Claude",
    "OpenAI",
    "Codex",
    "BaiduSyncdisk",
    "C:\\Users",
    "D:\\",
    "26876",
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml", ".gitignore", ""}


def csv_len(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return len(list(csv.DictReader(handle)))


def scan_identity_terms() -> list[str]:
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_dir() or ".git" in path.parts:
            continue
        if path.name == "smoke_test.py":
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in FORBIDDEN:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)} contains {term}")
    return hits


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    table1_n = csv_len(ROOT / "paper_values" / "table1_values.csv")
    table2_n = csv_len(ROOT / "paper_values" / "table2_values.csv")
    if table1_n != 6 or table2_n != 6:
        print(f"Unexpected table rows: table1={table1_n}, table2={table2_n}")
        return 1

    hits = scan_identity_terms()
    if hits:
        print("Anonymization scan found potential identity/local-path strings:")
        for hit in hits:
            print(f"  - {hit}")
        return 1

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
