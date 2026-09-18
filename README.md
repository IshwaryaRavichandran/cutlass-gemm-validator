# CUTLASS GEMM Validator

CUTLASS is NVIDIA's production GEMM kernel library. It ships hundreds of kernel variants and picks one per shape and dtype at compile time. When no kernel matches a shape, the profiler doesn't throw an error - it just returns nothing. This harness drives `cutlass_profiler` across a shape/dtype matrix, checks each result against CUTLASS's own reference GEMM, and reports what actually runs versus what silently returns empty.

## Architecture

```
+-----------------------------------------------------------------------+
|                          PyTest Framework                             |
|      (Boundary Analysis, Hardware Support, LLM Shape Coverage)        |
+-----------------------------------------------------------------------+
                                  |
                        (subprocess + CSV output)
                                  v
+-----------------------------------------------------------------------+
|                  NVIDIA CUTLASS cutlass_profiler (v3.5.1)             |
|  Kernels: s884gemm (Volta) + s1688gemm (Turing tensor-op) fp16        |
|  align8 vectorized loads, internal reference verification             |
+-----------------------------------------------------------------------+
                                  |
                          (Device Execution)
                                  v
+-----------------------------------------------------------------------+
|                      NVIDIA GPU (T4, sm_75)                           |
+-----------------------------------------------------------------------+
```

## Kernel Coverage

### M Alignment

M=4096, 4095, 4097 - only 4096 matches a kernel. This build only compiled align8 fp16 kernels for sm75, no fallback for unaligned output. align8 needs 16-byte-aligned writes, which means M has to be a multiple of 8.

### Batch-1 Decode

M=1. Zero kernels match. This isn't a contrived edge case - it's what a real decode step looks like, and it fails the same way 4095 does.

### bf16 / tf32

Both are in the shape config. Both return zero kernels on this GPU. T4 is Turing - bf16 and tf32 tensor cores don't exist until Ampere.

### LLM Shapes

Real transformer FFN dimensions (LLaMA-style up/down projections, N=14336) alongside the synthetic squares, so the matrix isn't just testing shapes nobody actually runs.

## M-Alignment Results

Bisected against the built binary, CSV row count per M:

| M | Kernels matched |
|---|---|
| 1, 513, 1023, 2047, 4095, 4097 | 0 |
| 8, 512, 2048, 4096 | 8 |

Every M divisible by 8 gets 8 kernel results back (4 transpose combinations × 2 SM generations). Everything else gets nothing.

## Test Results

All tests passing on NVIDIA T4:

```
tests/test_correctness.py::test_gemm_correctness SKIPPED (fp8_precis...) [ 16%]
tests/test_shapes.py::test_boundary_shapes_verify[on_boundary] PASSED    [ 33%]
tests/test_shapes.py::test_boundary_shapes_verify[one_below_boundary_M_not_div8] PASSED [ 50%]
tests/test_shapes.py::test_boundary_shapes_verify[one_above_boundary_M_not_div8] PASSED [ 66%]
tests/test_shapes.py::test_boundary_shapes_verify[degenerate_gemv_M_not_div8] PASSED [ 83%]
tests/test_shapes.py::test_boundary_shapes_verify[unaligned_k_M_N_div8] PASSED [100%]

5 passed, 1 skipped, 1 deselected in 124.96s
```

Each boundary case checks the shape-specific expected result - a passing kernel, or a confirmed no-kernel-available - not a single pass/fail check.

## Performance Benchmarks

fp16 tensor-core (`s1688gemm`), T4:

| Shape | Runtime | TFLOPS | % of Peak | Arithmetic Intensity | Bound |
| --- | --- | --- | --- | --- | --- |
| 4096×4096×4096 | 4.14 ms | 33.2 | 51.1% | 1365.3 FLOPs/byte | compute |
| 2048×4096×4096 | 2.08 ms | 33.1 | 50.9% | 1024.0 FLOPs/byte | compute |

### Performance Notes

T4 fp16 peak is about 65 TFLOPS. Both shapes land around 51% of that. `roofline_position()` confirms compute-bound at 4096³ - arithmetic intensity 1365 FLOPs/byte against T4's ridge point of ~200-217 FLOPs/byte. The 51%-of-peak gap isn't bandwidth; it's likely pipeline depth (`stages=2` is shallow for hiding global-memory latency on Turing), CTA tile occupancy, or register pressure - none of which this harness measures yet. Needs Nsight Compute occupancy data to pin down.

## Getting Started

### Requirements

- NVIDIA GPU, tested on T4 (sm_75)
- CUTLASS v3.5.1 - `main` doesn't build on Turing, an int4 atomic call broke against newer CUDA toolkits
- Python 3.10+, PyTorch, pytest

### Build & Test

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git
cd cutlass && mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=75 -DCUTLASS_ENABLE_PROFILER=ON -DCUTLASS_LIBRARY_OPERATIONS=gemm
make cutlass_profiler -j$(nproc)
```

Run the full test suite:

```bash
pip install -e .
pytest tests/ -v -m gpu --junitxml=results.xml
```

## Roadmap

- Roofline chart export (arithmetic intensity vs. TFLOPS)
- L2 cache control (`--llc-capacity`) so small-shape bandwidth numbers aren't inflated by cache residency
- A100/L4 run to fill out the bf16/tf32 support matrix
- Bank-conflict testing via explicit leading-dimension strides - needs the CUTLASS device API directly, the profiler CLI doesn't expose stride overrides

## Key Takeaways

This harness validates CUTLASS GEMM kernels across:

- Kernel-catalog alignment constraints, not just numerical correctness
- Hardware dtype support by GPU generation
- Real LLM workload shapes, not only synthetic squares
- Performance read through roofline analysis instead of raw TFLOPS alone

## Related Projects

- [`gpu-correctness-harness`](https://github.com/IshwaryaRavichandran/gpu-correctness-harness) - raw CUDA kernels, ctypes-bridged, validated against NumPy
- [`flash-attn-validator`](https://github.com/IshwaryaRavichandran/flash-attn-validator) - Triton Flash Attention
