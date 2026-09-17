"""Golden-model GEMM reference — computed at fp32 accumulation regardless of storage dtype."""

from __future__ import annotations
import torch

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "tf32": torch.float32,
    "fp32": torch.float32,
    "fp8": torch.float16,
}


def golden_gemm(M: int, N: int, K: int, dtype: str, layout=("row", "col"),
                 alpha: float = 1.0, beta: float = 0.0, seed: int = 0, device: str = "cuda") -> dict:
    torch.manual_seed(seed)
    storage_dtype = DTYPE_MAP[dtype]
    a_layout, b_layout = layout
    A = torch.randn(M, K, dtype=storage_dtype, device=device)
    B = torch.randn(K, N, dtype=storage_dtype, device=device)
    C = torch.randn(M, N, dtype=storage_dtype, device=device) if beta != 0.0 else torch.zeros(
        M, N, dtype=storage_dtype, device=device
    )
    if a_layout == "col":
        A = A.t().contiguous().t()
    if b_layout == "col":
        B = B.t().contiguous().t()
    with torch.no_grad():
        D_ref = alpha * (A.float() @ B.float()) + beta * C.float()
    return {"A": A, "B": B, "C": C, "D_ref": D_ref, "alpha": alpha, "beta": beta}
