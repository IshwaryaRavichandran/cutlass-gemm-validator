"""Tolerance-aware correctness comparison."""

from __future__ import annotations
from dataclasses import dataclass
import torch


class ToleranceViolation(AssertionError):
    pass


@dataclass
class ComparisonReport:
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    rtol: float
    atol: float
    num_elements: int
    num_violations: int

    def summary(self) -> str:
        return (f"{'PASS' if self.passed else 'FAIL'} | max_abs_err={self.max_abs_error:.3e} "
                f"mean_abs_err={self.mean_abs_error:.3e} max_rel_err={self.max_rel_error:.3e} | "
                f"violations={self.num_violations}/{self.num_elements} (rtol={self.rtol}, atol={self.atol})")


def compare_results(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> ComparisonReport:
    actual = actual.float()
    expected = expected.float()
    abs_err = (actual - expected).abs()
    rel_err = abs_err / (expected.abs() + 1e-12)
    threshold = atol + rtol * expected.abs()
    violations = abs_err > threshold
    return ComparisonReport(
        passed=bool(violations.sum().item() == 0),
        max_abs_error=float(abs_err.max().item()),
        mean_abs_error=float(abs_err.mean().item()),
        max_rel_error=float(rel_err.max().item()),
        rtol=rtol, atol=atol,
        num_elements=int(actual.numel()),
        num_violations=int(violations.sum().item()),
    )


def assert_within_tolerance(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> ComparisonReport:
    report = compare_results(actual, expected, rtol, atol)
    if not report.passed:
        raise ToleranceViolation(report.summary())
    return report
