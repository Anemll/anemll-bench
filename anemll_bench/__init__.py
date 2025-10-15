"""
ANEMLL-Bench: A benchmarking tool for measuring Apple Neural Engine performance
"""

__version__ = "0.1.0"

# Avoid importing heavy/optional deps at package import time so that
# lightweight utilities (e.g., plotting) work without torch/psutil installed.
try:
    from anemll_bench.benchmark import Benchmark  # noqa: F401
    __all__ = ["Benchmark"]
except Exception:
    # Soft-fail: allow importing submodules like anemll_bench.utils without torch
    __all__ = []