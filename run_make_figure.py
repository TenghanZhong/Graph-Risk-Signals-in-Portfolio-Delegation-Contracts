from __future__ import annotations

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "scripts" / "04_make_figures.py"), run_name="__main__")
