"""Public loader for the released REAL-Bench task definitions."""

from real_bench.loader import (
    DEFAULT_BENCHMARK_ROOT,
    RealBenchValidationError,
    load_real_bench,
)

__all__ = [
    "DEFAULT_BENCHMARK_ROOT",
    "RealBenchValidationError",
    "load_real_bench",
]
