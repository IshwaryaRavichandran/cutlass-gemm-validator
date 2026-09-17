import pytest
from gemm_validator import CutlassRunner
from gemm_validator.benchmark import compute_tflops


def _flatten_shapes(config: dict, categories: list[str]) -> list[dict]:
    out = []
    for cat in categories:
        for shape in config.get(cat, []):
            shape = dict(shape)
            shape["_category"] = cat
            out.append(shape)
    return out


@pytest.fixture
def all_shapes(gemm_config):
    return _flatten_shapes(gemm_config, ["nominal", "llm_shapes", "boundary", "adversarial"])


def test_shapes_load(all_shapes):
    assert len(all_shapes) > 0
    for shape in all_shapes:
        assert {"M", "N", "K", "dtype", "name"} <= shape.keys()


@pytest.mark.gpu
def test_gemm_correctness(gemm_config, cutlass_runner: CutlassRunner, all_shapes, request):
    failures = []
    for shape in all_shapes:
        if shape["dtype"] == "fp8":
            pytest.skip(f"{shape['name']}: fp8 requires sm_89+; skip on T4/A100.")
            continue
        try:
            result = cutlass_runner.run_gemm(M=shape["M"], N=shape["N"], K=shape["K"], dtype=shape["dtype"])
        except FileNotFoundError as e:
            pytest.skip(str(e))
            return
        tflops = compute_tflops(shape["M"], shape["N"], shape["K"], result.runtime_ms)
        print(f"[{shape['_category']}] {shape['name']}: status={result.status} "
              f"runtime={result.runtime_ms:.4f}ms tflops={tflops:.2f}")
        if result.status.lower() not in ("passed", "success"):
            failures.append((shape["name"], result.status))
    if failures:
        pytest.fail(f"{len(failures)} shape(s) failed verification: {failures}")
