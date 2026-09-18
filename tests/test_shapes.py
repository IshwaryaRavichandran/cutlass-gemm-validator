import pytest
from gemm_validator import CutlassRunner


@pytest.mark.gpu
@pytest.mark.parametrize("case,expect_kernel", [
    pytest.param({"M": 4096, "N": 4096, "K": 4096}, True, id="on_boundary"),
    pytest.param({"M": 4095, "N": 4096, "K": 4096}, False, id="one_below_boundary_M_not_div8"),
    pytest.param({"M": 4097, "N": 4096, "K": 4096}, False, id="one_above_boundary_M_not_div8"),
    pytest.param({"M": 1, "N": 4096, "K": 4096}, False, id="degenerate_gemv_M_not_div8"),
    pytest.param({"M": 512, "N": 512, "K": 513}, True, id="unaligned_k_M_N_div8"),
])
def test_boundary_shapes_verify(cutlass_runner: CutlassRunner, case, expect_kernel):
    """
    This CUTLASS build only compiled align8 fp16 tensor-core kernels for
    sm75, which require M % 8 == 0 for vectorized output writes. Shapes
    with M not divisible by 8 correctly have zero matching kernels — that
    is expected, documented behavior, not a failure.
    """
    try:
        result = cutlass_runner.run_gemm(M=case["M"], N=case["N"], K=case["K"], dtype="fp16")
    except FileNotFoundError as e:
        pytest.skip(str(e))
        return

    if expect_kernel:
        assert result.status.lower() in ("passed", "success"), (
            f"Expected a matching kernel for {case}, got status={result.status}"
        )
    else:
        assert result.status == "no_kernel_available", (
            f"Expected no kernel to match {case} (M % 8 != 0), but got status={result.status}"
        )
