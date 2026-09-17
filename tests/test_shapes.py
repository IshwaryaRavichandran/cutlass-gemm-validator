import pytest
from gemm_validator import CutlassRunner


@pytest.mark.gpu
@pytest.mark.parametrize("case", [
    pytest.param({"M": 4096, "N": 4096, "K": 4096}, id="on_boundary"),
    pytest.param({"M": 4095, "N": 4096, "K": 4096}, id="one_below_boundary"),
    pytest.param({"M": 4097, "N": 4096, "K": 4096}, id="one_above_boundary"),
    pytest.param({"M": 1, "N": 4096, "K": 4096}, id="degenerate_gemv"),
    pytest.param({"M": 512, "N": 512, "K": 513}, id="unaligned_k"),
])
def test_boundary_shapes_verify(cutlass_runner: CutlassRunner, case):
    try:
        result = cutlass_runner.run_gemm(M=case["M"], N=case["N"], K=case["K"], dtype="fp16")
    except FileNotFoundError as e:
        pytest.skip(str(e))
        return
    assert result.status.lower() in ("passed", "success"), f"Boundary shape {case} failed: status={result.status}"
