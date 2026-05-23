from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.contract_experiments import build_arg_parser, run_multi_seed, run_pipeline


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if getattr(args, "seeds", ""):
        run_multi_seed(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
