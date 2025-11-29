"""Nobel-level validation orchestrator for WillowLab theorems."""

from __future__ import annotations

import cmath
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typing import Dict, List

from willowlab.schema import (
    NobelValidationSuite,
    TheoremValidationResult,
    WillowDataset,
)
from willowlab.spg import run_cosmic_ratchet_test


class NobelValidationRunner:
    """Coordinate consolidated theorem validation runs."""

    def __init__(self) -> None:
        self.suite = NobelValidationSuite()

    def validate_theorem_b1(self, dataset, dataset_label: str) -> TheoremValidationResult:
        from willowlab.resolvent import validate_theorem_b1 as run_validation

        falsification_criteria: Dict[str, object] = {
            "slope_range": (0.9, 1.1),
            "min_r2": 0.9,
            "requirement": "T_spec ∝ T_ent with slope 1±0.1 and R²>0.9",
        }

        try:
            if (
                dataset.resolvent_trace is None
                or dataset.entropy is None
                or dataset.effective_energy is None
            ):
                raise ValueError("Dataset missing resolvent, entropy, or energy data")

            trace_abs = [float(abs(value)) for value in dataset.resolvent_trace]
            result = run_validation(
                trace_abs,
                dataset.JT_scan_points,
                dataset.entropy,
                dataset.effective_energy,
            )
            validated = bool(
                result.get("passed")
                and 0.9 <= result.get("slope", float("nan")) <= 1.1
                and result.get("r2", 0.0) > 0.9
            )
            failure_reason = None
            if not validated:
                failure_reason = (
                    f"slope={result.get('slope', float('nan')):.3f}, "
                    f"R²={result.get('r2', float('nan')):.3f}"
                )
            return TheoremValidationResult(
                theorem_id="B.1",
                dataset_used=dataset_label,
                falsification_criteria=falsification_criteria,
                actual_results=result,
                validated=validated,
                failure_reason=failure_reason,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return TheoremValidationResult(
                theorem_id="B.1",
                dataset_used=dataset_label,
                falsification_criteria=falsification_criteria,
                actual_results={"error": str(exc)},
                validated=False,
                failure_reason=f"Test execution failed: {exc}",
            )

    def validate_theorem_b2(self, dataset, dataset_label: str) -> TheoremValidationResult:
        falsification_criteria: Dict[str, object] = {
            "requirement": "All divergences coincide with exceptional points",
            "tolerance": "Zero tolerance for divergence without EP",
        }

        try:
            problematic_points = self._find_divergences_without_eps(dataset)
            validated = len(problematic_points) == 0
            failure_reason = None
            if not validated:
                failure_reason = (
                    f"{len(problematic_points)} divergences without EP signatures"
                )
            return TheoremValidationResult(
                theorem_id="B.2",
                dataset_used=dataset_label,
                falsification_criteria=falsification_criteria,
                actual_results={
                    "divergences_found": len(problematic_points),
                    "problematic_points": problematic_points,
                },
                validated=validated,
                failure_reason=failure_reason,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return TheoremValidationResult(
                theorem_id="B.2",
                dataset_used=dataset_label,
                falsification_criteria=falsification_criteria,
                actual_results={"error": str(exc)},
                validated=False,
                failure_reason=f"Test execution failed: {exc}",
            )

    def validate_theorem_spg1(self, dataset, dataset_label: str) -> TheoremValidationResult:
        falsification_criteria: Dict[str, object] = {
            "omega_op_limit": 0.0179,
            "ap_prime_detection": "Must detect discrete contractions",
        }

        try:
            res = run_cosmic_ratchet_test(dataset)

            # For the Nobel suite, we might allow breaches IF they are caught by triggers
            # But strictly, the raw data shouldn't violate the bounds if the chip worked.
            validated = res.crosstalk_breaches == 0 and res.critical_crossings >= 2

            failure_reason = None
            if not validated:
                reasons = []
                if res.crosstalk_breaches > 0:
                    reasons.append(f"Noise floor breached limit {res.crosstalk_breaches} times")
                if res.critical_crossings < 2:
                    reasons.append(f"Insufficient critical events ({res.critical_crossings} < 2)")
                failure_reason = "; ".join(reasons)

            return TheoremValidationResult(
                theorem_id="SPG.1",  # Stochastic Projective Gravity
                dataset_used=dataset_label,
                falsification_criteria=falsification_criteria,
                actual_results={
                    "max_omega": float(max(res.omega_op_series)) if isinstance(res.omega_op_series, list) else float(res.omega_op_series.max()),
                    "events": res.critical_crossings,
                },
                validated=validated,
                failure_reason=failure_reason,
            )
        except Exception as exc:
            return TheoremValidationResult(
                theorem_id="SPG.1",
                dataset_used=dataset_label,
                falsification_criteria=falsification_criteria,
                actual_results={"error": str(exc)},
                validated=False,
                failure_reason=str(exc),
            )

    def run_complete_validation(self, datasets: Dict[str, object]) -> NobelValidationSuite:
        print("🚀 STARTING NOBEL-LEVEL VALIDATION SUITE")
        print("=" * 60)

        results: List[TheoremValidationResult] = []

        print("🔬 Validating Theorem B.1 (Spectral-Entanglement Duality)...")
        b1_result = self.validate_theorem_b1(
            datasets["sept_dec_2025"], "Willow_Sept+Dec_2025"
        )
        print(f"   Result: {'PASS' if b1_result.validated else 'FAIL'}")
        results.append(b1_result)

        print("🔬 Validating Theorem B.2 (Exceptional Point Reduction)...")
        b2_result = self.validate_theorem_b2(datasets["sept_2025"], "Willow_Sept_2025")
        print(f"   Result: {'PASS' if b2_result.validated else 'FAIL'}")
        results.append(b2_result)

        print("🔬 Validating SPG.1 (Cosmic Ratchet)...")
        spg_result = self.validate_theorem_spg_ratchet(
            datasets["sept_dec_2025"], "Willow_Sept+Dec_2025"
        )
        print(f"   Result: {'PASS' if spg_result.validated else 'FAIL'}")
        results.append(spg_result)

        self.suite.results.extend(results)
        self.suite.theorems_tested = [result.theorem_id for result in results]
        dataset_order: List[str] = []
        for result in results:
            if result.dataset_used not in dataset_order:
                dataset_order.append(result.dataset_used)
        self.suite.datasets_used = dataset_order

        self.suite.overall_status = all(result.validated for result in results)

        print("=" * 60)
        if self.suite.overall_status:
            print("🎉 ALL THEOREMS VALIDATED - READY FOR NOBEL SUBMISSION")
        else:
            print("⚠️  FRAMEWORK REQUIRES REVISION - SOME THEOREMS FALSIFIED")

        return self.suite

    def _find_divergences_without_eps(self, dataset, threshold: float = 1e6) -> List[Dict[str, float]]:
        from willowlab.resolvent import trace_resolvent_from_evals
        from willowlab.tests.t_resolvent import validate_theorem_b2 as detect_eps

        if dataset.floquet_eigenvalues is None:
            return []

        evals_raw = dataset.floquet_eigenvalues or []
        if hasattr(evals_raw, "to_list"):
            evals_raw = evals_raw.to_list()
        elif hasattr(evals_raw, "tolist"):
            evals_raw = evals_raw.tolist()

        evals_list: List[List[complex]] = []
        for row in evals_raw:
            if hasattr(row, "to_list"):
                row = row.to_list()
            elif hasattr(row, "tolist"):
                row = row.tolist()
            evals_list.append([complex(value) for value in row])

        if dataset.resolvent_trace is not None:
            trace_abs = [float(abs(value)) for value in dataset.resolvent_trace]
        else:
            trace_abs = [float(value) for value in trace_resolvent_from_evals(evals_list)]

        ep_flags = detect_eps(evals_list, trace_abs)
        jt_raw = dataset.JT_scan_points
        if hasattr(jt_raw, "to_list"):
            jt_values = jt_raw.to_list()
        elif hasattr(jt_raw, "tolist"):
            jt_values = jt_raw.tolist()
        else:
            jt_values = list(jt_raw)

        problematic: List[Dict[str, float]] = []
        for idx, (flag, trace_value) in enumerate(zip(ep_flags, trace_abs)):
            if trace_value > threshold and not flag:
                jt_val = float(jt_values[idx]) if idx < len(jt_values) else float(idx)
                problematic.append(
                    {
                        "jt": jt_val,
                        "trace": float(trace_value),
                        "ep_detected": bool(flag),
                    }
                )
        return problematic
def _linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def _synthetic_dataset() -> WillowDataset:
    jt = _linspace(0.5, 1.5, 40)
    evals = []
    overlap_matrices = []
    for value in jt:
        angles = [2.0 * math.pi * k / 6 for k in range(6)]
        radii = 0.9 + 0.1 * math.exp(-20.0 * (value - 1.0) ** 2)
        evals.append([radii * cmath.exp(1j * angle) for angle in angles])
        overlap_matrices.append(
            [
                [1.0 if i == j else 0.0 for j in range(6)]
                for i in range(6)
            ]
        )

    trace_abs = [math.exp(0.02 * value ** 3 + 0.19 * value ** 2) for value in jt]
    entropy = [4 * 0.02 * value ** 3 + 2 * 0.19 * value ** 2 for value in jt]
    energy = [value ** 2 for value in jt]
    resolvent_trace = trace_abs

    return WillowDataset(
        JT_scan_points=jt,
        floquet_eigenvalues=evals,
        overlap_matrices=overlap_matrices,
        resolvent_trace=resolvent_trace,
        entropy=entropy,
        effective_energy=energy,
    )


def _load_default_datasets() -> Dict[str, WillowDataset]:
    from willowlab.io import load_willow

    try:
        sept = load_willow("willow_sept_2025.npz")
        pooled = load_willow("willow_pooled_2025.npz")
        return {"sept_2025": sept, "sept_dec_2025": pooled}
    except FileNotFoundError as exc:  # pragma: no cover - depends on filesystem
        raise FileNotFoundError(
            "Missing required Nobel validation dataset: "
            "willow_sept_2025.npz"
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on file integrity
        raise RuntimeError(
            "Failed to load default Nobel validation datasets. "
            "Ensure willow_sept_2025.npz and willow_pooled_2025.npz are valid."
        ) from exc


def execute_nobel_validation(
    report_path: Path | str = "nobel_validation_report.json",
    datasets: Dict[str, WillowDataset] | None = None,
) -> Dict[str, object]:
    dataset_map = datasets if datasets is not None else _load_default_datasets()
    runner = NobelValidationRunner()
    suite = runner.run_complete_validation(dataset_map)
    report = suite.generate_nobel_report()

    path = Path(report_path)
    path.write_text(json.dumps(report, indent=2))
    return report


def test_nobel_validation(tmp_path):
    """Exercise the Nobel validation runner on synthetic datasets."""

    report_path = tmp_path / "nobel_validation_report.json"
    synthetic = _synthetic_dataset()
    datasets = {"sept_2025": synthetic, "sept_dec_2025": synthetic}
    report = execute_nobel_validation(report_path, datasets=datasets)

    assert report["theorems_tested"] == 3
    assert report_path.exists()


def test_load_default_datasets_missing(monkeypatch):
    def fake_load(path):
        raise FileNotFoundError("not found")

    monkeypatch.setitem(sys.modules, "willowlab.io", SimpleNamespace(load_willow=fake_load))

    with pytest.raises(FileNotFoundError):
        _load_default_datasets()


def test_load_default_datasets_corrupted(monkeypatch):
    def fake_load(path):
        raise ValueError("bad data")

    monkeypatch.setitem(sys.modules, "willowlab.io", SimpleNamespace(load_willow=fake_load))

    with pytest.raises(RuntimeError):
        _load_default_datasets()
