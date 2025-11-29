"""
Stochastic Projective Gravity (SPG) / Cosmic Ratchet Validation Module.
Authoritative Implementation per Prediction Registry v2.0.
"""

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np
    HAVE_NUMPY = True
except ModuleNotFoundError:  # pragma: no cover - shim fallback for tests
    from . import _numpy_shim as np  # type: ignore
    HAVE_NUMPY = False


@dataclass
class RatchetResult:
    """Standardized output for SPG tests."""

    ap_prime_series: Any
    omega_op_series: Any
    xi_series: Any
    trigger_indices: Any
    critical_crossings: int
    crosstalk_breaches: int
    passed: bool
    mode: str  # 'calibration' or 'validation'
    meta: Dict[str, float]


def _std(values: Sequence[float]) -> float:
    seq = list(values)
    if not seq:
        return 0.0
    mean = sum(seq) / len(seq)
    return math.sqrt(sum((val - mean) ** 2 for val in seq) / len(seq))


def _off_diag_norm(matrix: Sequence[Sequence[complex]]) -> float:
    if HAVE_NUMPY:
        M = np.asarray(matrix)
        diag = np.diag(np.diag(M))
        off_diag = M - diag
        return float(np.linalg.norm(off_diag) / M.shape[0])

    n = len(matrix)
    if n == 0:
        return 0.0
    total = 0.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i != j:
                total += abs(value) ** 2
    return math.sqrt(total) / n


def _max_value(values: Iterable[float]) -> float:
    return float(max(values)) if values else float("nan")


def _min_value(values: Iterable[float]) -> float:
    return float(min(values)) if values else float("nan")


def _mean_value(values: Iterable[float]) -> float:
    seq = list(values)
    if not seq:
        return float("nan")
    return float(sum(seq) / len(seq))


class CosmicRatchetValidator:
    """Implements Tier 1 predictions CR-4 and CR-5 from the Registry."""

    # CR-5: FRW-Radar Acceleration Threshold
    # Derived from w_eff = -1/3 (Deceleration -> Acceleration transition)
    AP_THRESHOLD: float = -0.333333

    # CR-4: Pantheon+ Compliance
    # Cumulative geometric noise bound (95% C.L.)
    OMEGA_THRESHOLD: float = 0.0179

    def __init__(self, ds):
        self.ds = ds
        self.T = len(ds.JT_scan_points)

    def _compute_xi(self) -> Any:
        """
        Calculates Order Parameter ξ per Registry CR-5.
        ξ(JT) = 1 - min|1 - λ_k(JT)|

        Represents proximity to the decoherence threshold (Instability Tongue).
        """
        if self.ds.floquet_eigenvalues is None:
            raise ValueError("SPG requires Floquet Eigenvalues for CR-5 validation.")

        if HAVE_NUMPY:
            dist = np.min(np.abs(1.0 - np.asarray(self.ds.floquet_eigenvalues)), axis=1)
            return 1.0 - dist

        dist = [min(abs(1.0 - val) for val in row) for row in self.ds.floquet_eigenvalues]
        return [1.0 - value for value in dist]

    def _compute_ap_prime(self, xi: Any) -> Any:
        """
        Calculates Operational Acceleration AP' per Registry CR-5.
        AP' = d²ξ/dJT² (normalized)
        """
        ap_prime = np.gradient(np.gradient(xi, self.ds.JT_scan_points), self.ds.JT_scan_points)

        if HAVE_NUMPY:
            std_val = float(np.std(ap_prime))
            if std_val > 1e-12:
                return ap_prime / std_val
            return ap_prime

        series = list(ap_prime)
        std_val = _std(series)
        if std_val > 1e-12:
            return [val / std_val for val in series]
        return series

    def _compute_omega_op(self) -> Any:
        """
        Calculates Protractor Noise Ω_op per Registry CR-4.
        Primary: Off-diagonal Frobenius norm of overlap matrices.
        Fallback (CCC-4): Eigenvector Condition Number (EP signature).
        """
        if self.ds.overlap_matrices is not None:
            values = [_off_diag_norm(self.ds.overlap_matrices[t]) for t in range(self.T)]
            return np.asarray(values) if HAVE_NUMPY else values

        if self.ds.floquet_eigenvectors is not None:
            print(
                "WARNING: Overlap matrices missing. Using EP Condition Number fallback (CCC-4)."
            )
            omega_series: List[float] = []
            for t in range(self.T):
                if not HAVE_NUMPY:
                    raise ValueError("Eigenvector condition number requires NumPy support.")
                kappa = np.linalg.cond(self.ds.floquet_eigenvectors[t])
                omega_series.append(np.log10(kappa) * (0.02 / 6.0))
            return np.asarray(omega_series) if HAVE_NUMPY else omega_series

        raise ValueError("Cannot compute Ω_op: Missing overlap_matrices and eigenvectors.")

    def run_calibration(self) -> Dict[str, Any]:
        """
        Phase 2: Data Calibration.
        Characterizes the data distribution without failing the test.
        """
        xi = self._compute_xi()
        ap_prime = self._compute_ap_prime(xi)
        omega_op = self._compute_omega_op()

        xi_values = xi if HAVE_NUMPY else list(xi)
        ap_values = ap_prime if HAVE_NUMPY else list(ap_prime)
        omega_values = omega_op if HAVE_NUMPY else list(omega_op)

        return {
            "mode": "CALIBRATION",
            "stats": {
                "xi_mean": float(np.mean(xi_values)) if HAVE_NUMPY else _mean_value(xi_values),
                "xi_max": float(np.max(xi_values)) if HAVE_NUMPY else _max_value(xi_values),
                "ap_prime_min": float(np.min(ap_values)) if HAVE_NUMPY else _min_value(ap_values),
                "ap_prime_std": float(np.std(ap_values)) if HAVE_NUMPY else _std(ap_values),
                "omega_op_max": float(np.max(omega_values)) if HAVE_NUMPY else _max_value(omega_values),
            },
            "recommendation": (
                "Check if ap_prime_min is approx -0.33 scaling. If not, normalization factor needed."
            ),
        }

    def run_validation(self, strict: bool = True) -> RatchetResult:
        """
        Phase 3: Validation Run.
        Executes strict checks against Registry Tier 1 thresholds.
        """
        xi = self._compute_xi()
        ap_prime = self._compute_ap_prime(xi)
        omega_op = self._compute_omega_op()

        if HAVE_NUMPY:
            accel_triggers = np.where(ap_prime < self.AP_THRESHOLD)[0]
            xtalk_triggers = np.where(omega_op > self.OMEGA_THRESHOLD)[0]
            all_triggers = np.unique(np.concatenate((accel_triggers, xtalk_triggers)))
        else:
            accel_triggers = [idx for idx, val in enumerate(ap_prime) if val < self.AP_THRESHOLD]
            xtalk_triggers = [idx for idx, val in enumerate(omega_op) if val > self.OMEGA_THRESHOLD]
            all_triggers = list(sorted(set(accel_triggers + xtalk_triggers)))

        crosstalk_breaches = len(xtalk_triggers)
        critical_crossings = len(accel_triggers)

        has_critical_events = critical_crossings >= 2
        no_fatal_breaches = crosstalk_breaches == 0

        passed = has_critical_events and no_fatal_breaches

        return RatchetResult(
            ap_prime_series=ap_prime,
            omega_op_series=omega_op,
            xi_series=xi,
            trigger_indices=all_triggers,
            critical_crossings=critical_crossings,
            crosstalk_breaches=crosstalk_breaches,
            passed=passed,
            mode="VALIDATION",
            meta={
                "ap_threshold": self.AP_THRESHOLD,
                "omega_threshold": self.OMEGA_THRESHOLD,
            },
        )


def run_cosmic_ratchet_test(dataset) -> RatchetResult:
    validator = CosmicRatchetValidator(dataset)
    return validator.run_validation()


def validate_theorem_spg(dataset) -> Dict[str, Any]:
    result = run_cosmic_ratchet_test(dataset)
    omega_series = result.omega_op_series if HAVE_NUMPY else list(result.omega_op_series)
    return {
        "omega_max": float(np.max(omega_series)) if HAVE_NUMPY else _max_value(omega_series),
        "critical_instanton_events": int(result.critical_crossings),
        "crosstalk_breaches": int(result.crosstalk_breaches),
        "passed": bool(result.passed),
    }


# CLI Hook (UPDATED)
def _load_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required for this command") from exc

    return yaml.safe_load(open(config_path))


def run_spg(config_path, mode: str = "validate"):
    from .io import load_willow

    cfg = _load_config(config_path)
    ds = load_willow(cfg["dataset"])

    validator = CosmicRatchetValidator(ds)

    if mode == "calibrate":
        print("🔧 RUNNING SPG CALIBRATION (PHASE 2)")
        report = validator.run_calibration()
        print(json.dumps(report, indent=2))

    else:
        print("🪐 RUNNING SPG VALIDATION (PHASE 3)")
        res = validator.run_validation()

        print(f"   CR-5 (AP' < -1/3) Events: {res.critical_crossings} (Req: >= 2)")
        print(f"   CR-4 (Ω_op > 0.0179) Breaches: {res.crosstalk_breaches} (Req: 0)")

        art = pathlib.Path(cfg.get("artifacts_dir", "./artifacts"))
        art.mkdir(parents=True, exist_ok=True)
        (art / "spg_results.json").write_text(
            json.dumps(
                {
                    "critical_crossings": res.critical_crossings,
                    "crosstalk_breaches": res.crosstalk_breaches,
                    "passed": res.passed,
                    "trigger_indices": list(res.trigger_indices),
                    "ap_prime_series": list(res.ap_prime_series),
                    "omega_op_series": list(res.omega_op_series),
                },
                indent=2,
            )
        )

        if res.passed:
            print("✅ TIER 1 VALIDATED: Framework holds.")
        else:
            print("❌ TIER 1 FALSIFIED: Check registry falsification protocols.")
