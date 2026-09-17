import pathlib
import pytest
import torch
import yaml
from gemm_validator import CutlassRunner

CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs" / "gemm_shapes.yaml"


@pytest.fixture(scope="session")
def gemm_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def gpu_name() -> str:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available.")
    name = torch.cuda.get_device_name(0)
    for key in ("T4", "A100", "L4"):
        if key in name:
            return key
    return name


@pytest.fixture(scope="session")
def cutlass_runner() -> CutlassRunner:
    return CutlassRunner()
