from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "run_reproduce.py",
        "--data_dir",
        "data/raw",
        "--out_dir",
        "outputs/full",
        "--seeds",
        "7,11,13,17,19,23,29,31,37,41",
        "--run_neural",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
