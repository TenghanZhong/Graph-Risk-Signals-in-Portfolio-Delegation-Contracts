from __future__ import annotations

from pathlib import Path


def main() -> None:
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    note = raw / "README.local.txt"
    if not note.exists():
        note.write_text(
            "Place public raw market data here. See data/README.md for expected file names.\n",
            encoding="utf-8",
        )
    print(f"Prepared raw-data directory: {raw.resolve()}")


if __name__ == "__main__":
    main()
