"""gemm_validator — correctness and performance validation for CUTLASS GEMM kernels."""

from .reference import golden_gemm
from .correctness import compare_results, ToleranceViolation
from .cutlass_runner import CutlassRunner
from .benchmark import parse_profiler_csv, compute_tflops, roofline_position

__all__ = [
    "golden_gemm",
    "compare_results",
    "ToleranceViolation",
    "CutlassRunner",
    "parse_profiler_csv",
    "compute_tflops",
    "roofline_position",
]
