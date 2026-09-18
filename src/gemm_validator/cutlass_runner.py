"""CutlassRunner — drives the built cutlass_profiler binary as a subprocess per shape/dtype config."""

from __future__ import annotations
import csv, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProfilerResult:
    name: str
    runtime_ms: float
    gflops: float
    status: str
    raw_row: dict


class CutlassRunner:
    def __init__(self, cutlass_root: str = "/content/cutlass", build_dir: str = "/content/cutlass/build"):
        self.cutlass_root = Path(cutlass_root)
        self.build_dir = Path(build_dir)
        self.profiler_bin = self.build_dir / "tools" / "profiler" / "cutlass_profiler"

    def is_built(self) -> bool:
        return self.profiler_bin.exists()

    def run_gemm(self, M: int, N: int, K: int, dtype: str, layout=("row", "col"),
                 alpha: float = 1.0, beta: float = 0.0, extra_args=None) -> ProfilerResult:
        if not self.is_built():
            raise FileNotFoundError(
                f"cutlass_profiler not found at {self.profiler_bin}. Build it first."
            )
        cutlass_dtype = {"fp16": "f16", "bf16": "bf16", "tf32": "tf32", "fp32": "f32", "fp8": "f8"}[dtype]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = str(Path(tmpdir) / "profile_run")
            cmd = [
                str(self.profiler_bin), "--operation=Gemm",
                f"--m={M}", f"--n={N}", f"--k={K}",
                f"--A={cutlass_dtype}", f"--B={cutlass_dtype}", "--C=f32",
                f"--alpha={alpha}", f"--beta={beta}",
                "--verification-enabled=true",
                f"--output={output_prefix}",
            ]
            if extra_args:
                cmd.extend(extra_args)

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                raise RuntimeError(f"cutlass_profiler failed (exit {proc.returncode}):\n{proc.stderr}")

            csv_files = sorted(Path(tmpdir).glob("profile_run*.csv"))
            if not csv_files:
                raise RuntimeError(
                    f"No CSV output file found (looked in {tmpdir}).\nstdout:\n{proc.stdout[-1000:]}"
                )
            with open(csv_files[0], newline="") as f:
                rows = list(csv.DictReader(f))

            if not rows:
                # Not an error: this CUTLASS build's kernel catalog has no
                # kernel whose alignment/tile constraints this problem shape
                # satisfies (e.g., an align8 kernel set requires M % 8 == 0
                # for vectorized output writes). Report it explicitly rather
                # than crashing, so callers can distinguish "no kernel
                # available for this shape" from "kernel ran and failed."
                return ProfilerResult(
                    name=f"gemm_{M}x{N}x{K}_{dtype}",
                    runtime_ms=-1.0,
                    gflops=-1.0,
                    status="no_kernel_available",
                    raw_row={},
                )

            row = rows[0]
            return ProfilerResult(
                name=f"gemm_{M}x{N}x{K}_{dtype}",
                runtime_ms=float(row.get("Runtime", -1.0)),
                gflops=float(row.get("GFLOPs", -1.0)),
                status=row.get("Disposition", row.get("Verification", "unknown")),
                raw_row=row,
            )
