# cutlass-gemm-validator

A correctness + performance validation harness for NVIDIA CUTLASS GEMM kernels, built for Google Colab (T4/A100/L4 runtimes) and portable to any CUDA-capable machine.

**What this is, in validation-engineering terms:** a regression suite that treats a CUTLASS GEMM kernel the way you'd treat a DUT (device under test) — sweep the input space (shapes, dtypes, layouts, alignment edge cases), assert functional correctness against a golden reference, and log performance telemetry so silent regressions get caught the same way a broken API contract would in a traditional CI pipeline.

**What this is, in hardware terms:** CUTLASS exposes the GPU's tiling hierarchy — thread block tile → warp tile → instruction-level (MMA) tile — as C++ template parameters, mirroring systolic-array PE partitioning under fixed register-file and on-chip SRAM budgets.

## Project structure
See `docs/VALIDATION_PLAN.md` for the full test methodology.
