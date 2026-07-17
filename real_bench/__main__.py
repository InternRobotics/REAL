"""Command-line validation for the released REAL-Bench bundle."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from real_bench.loader import DEFAULT_BENCHMARK_ROOT, RealBenchValidationError, load_real_bench


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and validate the REAL-Bench task bundle.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="benchmark bundle directory (default: repository benchmark/)",
    )
    parser.add_argument(
        "--family",
        help="optionally return one family after validating the complete bundle",
    )
    args = parser.parse_args()

    try:
        all_tasks = load_real_bench(args.root)
        selected_tasks = (
            all_tasks if args.family is None else load_real_bench(args.root, family=args.family)
        )
    except RealBenchValidationError as error:
        parser.exit(1, f"REAL-Bench validation failed: {error}\n")

    counts = Counter(task["family"] for task in all_tasks)
    counts_text = ", ".join(f"{family}={count}" for family, count in counts.items())
    print(f"Loaded and validated {len(all_tasks)} REAL-Bench tasks ({counts_text}).")
    if args.family is not None:
        print(f"Selected {len(selected_tasks)} {args.family.upper()} tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
