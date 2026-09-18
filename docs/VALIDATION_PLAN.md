# Validation Plan — CUTLASS GEMM

## Scope
1. Functional correctness — kernel output matches a high-precision reference within dtype-appropriate tolerance.
2. Performance — achieved TFLOPS relative to theoretical peak, and compute- vs memory-bound classification.

## Confirmed findings (T4, CUTLASS v3.5.1, this build)

### Finding 1: bf16/tf32 GEMM unsupported on T4 (Turing, sm_75)
T4's tensor cores do not support bf16 or tf32 matrix-multiply-accumulate — those
require Ampere (sm_80) or later. The profiler returns no matching kernel for
these dtypes on this hardware. This is correct hardware behavior, not a bug.

### Finding 2: M must be divisible by 8 for fp16 GEMM on this build
This CUTLASS build only compiled `align8` fp16 tensor-core kernels for sm75
(no align1/2/4 fallback variants). `align8` requires 16-byte-aligned
vectorized output writes (`LDG.128`), which for column-major output requires
M % 8 == 0. Confirmed empirically by bisection:

| M | Divisible by 8 | Kernels matched |
|---|---|---|
| 4095, 4097, 1, 513, 1023, 2047 | No | 0 |
| 8, 512, 2048, 4096 | Yes | 8 |

This is a real, documentable adversarial-shape finding: any inference workload
with a non-multiple-of-8 batch/sequence dimension feeding this kernel family
would need a different (non-align8) kernel build or explicit padding.

## Known limitations
- fp8 shapes need sm_89+ (Ada/Hopper); auto-skipped on T4/A100.
- Single-GPU only — no multi-GPU/NCCL validation yet.
