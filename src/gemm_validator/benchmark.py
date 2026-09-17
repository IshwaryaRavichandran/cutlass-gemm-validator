"""Benchmark + roofline utilities."""

from __future__ import annotations
import csv, io
from dataclasses import dataclass

GPU_SPECS = {
    "T4":   {"fp16_tflops": 65.0,  "tf32_tflops": 32.0,  "fp32_tflops": 8.1,   "bw_gbs": 320},
    "A100": {"fp16_tflops": 312.0, "tf32_tflops": 156.0, "fp32_tflops": 19.5,  "bw_gbs": 1935},
    "L4":   {"fp16_tflops": 121.0, "tf32_tflops": 60.0,  "fp32_tflops": 30.3,  "bw_gbs": 300},
}


def parse_profiler_csv(csv_text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(csv_text)))


def compute_tflops(M: int, N: int, K: int, runtime_ms: float) -> float:
    flops = 2 * M * N * K
    seconds = runtime_ms / 1000.0
    return (flops / seconds) / 1e12 if seconds > 0 else 0.0


@dataclass
class RooflineResult:
    achieved_tflops: float
    peak_tflops: float
    pct_of_peak: float
    arithmetic_intensity: float
    bound: str


def roofline_position(M: int, N: int, K: int, runtime_ms: float, dtype: str, gpu: str, bytes_per_elem: int = 2) -> RooflineResult:
    specs = GPU_SPECS[gpu]
    peak_key = f"{dtype}_tflops" if f"{dtype}_tflops" in specs else "fp16_tflops"
    peak_tflops = specs[peak_key]
    bw_gbs = specs["bw_gbs"]
    achieved = compute_tflops(M, N, K, runtime_ms)
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * bytes_per_elem
    arithmetic_intensity = flops / bytes_moved
    ridge_point = (peak_tflops * 1e12) / (bw_gbs * 1e9)
    bound = "compute" if arithmetic_intensity > ridge_point else "memory"
    return RooflineResult(
        achieved_tflops=achieved, peak_tflops=peak_tflops,
        pct_of_peak=100.0 * achieved / peak_tflops if peak_tflops > 0 else 0.0,
        arithmetic_intensity=arithmetic_intensity, bound=bound,
    )
