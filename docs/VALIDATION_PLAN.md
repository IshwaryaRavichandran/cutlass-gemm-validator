# Validation Plan — CUTLASS GEMM

## Scope
1. Functional correctness — kernel output matches a high-precision reference within dtype-appropriate tolerance.
2. Performance — achieved TFLOPS relative to theoretical peak, and compute- vs memory-bound classification.

## Known limitations
- fp8 shapes need sm_89+ (Ada/Hopper); auto-skipped on T4/A100.
- Single-GPU only — no multi-GPU/NCCL validation yet.
