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
    """
    Compute the population standard deviation of a sequence of floats.
    
    Parameters:
    	values (Sequence[float]): Iterable of numeric values to measure.
    
    Returns:
    	float: Population standard deviation (divides by N). Returns 0.0 for an empty input.
    """
    seq = list(values)
    if not seq:
        return 0.0
    mean = sum(seq) / len(seq)
    return math.sqrt(sum((val - mean) ** 2 for val in seq) / len(seq))


def _off_diag_norm(matrix: Sequence[Sequence[complex]]) -> float:
    """
    Compute the off-diagonal Frobenius norm of a square matrix normalized by its size.
    
    Parameters:
        matrix (Sequence[Sequence[complex]]): 2D sequence representing a square matrix.
    
    Returns:
        float: The Frobenius norm of the matrix with diagonal entries removed, divided by the matrix dimension.
               Returns 0.0 for an empty matrix.
    """
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
    """
    Return the maximum numeric value from an iterable, or NaN when the iterable is empty.
    
    Parameters:
        values (Iterable[float]): Sequence of numeric values to inspect.
    
    Returns:
        float: The largest value found in `values`, or NaN if `values` is empty.
    """
    return float(max(values)) if values else float("nan")


def _min_value(values: Iterable[float]) -> float:
    """
    Return the minimum value from an iterable.
    
    Parameters:
        values (Iterable[float]): Sequence of numeric values to examine.
    
    Returns:
        float: The smallest value in `values`, or `NaN` if `values` is empty.
    """
    return float(min(values)) if values else float("nan")


def _mean_value(values: Iterable[float]) -> float:
    """
    Compute the arithmetic mean of a sequence of floats.
    
    Parameters:
    	values (Iterable[float]): Sequence of numeric values to average.
    
    Returns:
    	mean (float): The arithmetic mean of the input values, or `NaN` if the input is empty.
    """
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
        """
        Initialize the validator with a dataset and precompute the number of JT scan points.
        
        Parameters:
            ds: Dataset-like object containing simulation/measurement fields required by the validator.
                Must provide a sequence attribute `JT_scan_points`; its length is stored as `self.T`.
        """
        self.ds = ds
        self.T = len(ds.JT_scan_points)

    def _compute_xi(self) -> Any:
        """
        Compute the CR-5 order parameter ξ for each JT scan point.
        
        For each JT, ξ = 1 - min_k |1 - λ_k(JT)| where λ_k are the Floquet eigenvalues;
        ξ quantifies proximity to the decoherence/instability threshold.
        
        Returns:
            xi_series (sequence): Sequence of ξ values, one per JT scan point.
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
        Compute the normalized second derivative of the order parameter ξ with respect to JT_scan_points.
        
        Parameters:
            xi (array-like): Sequence of ξ values sampled at the dataset's JT_scan_points.
        
        Returns:
            ap_prime_series: Sequence of AP' values equal to d²ξ/d(JT)². If the standard deviation of the raw second-derivative series is greater than 1e-12, the series is normalized by that standard deviation; otherwise the unnormalized series is returned.
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
        Compute the Protractor Noise (Ω_op) series used for CR-4 validation.
        
        Primary method: compute the off-diagonal Frobenius norm of each overlap matrix.
        Fallback method: compute a scaled log10 of the condition number of each Floquet eigenvector matrix when overlap matrices are missing.
        
        Returns:
            omega_series (numpy.ndarray or list[float]): Sequence of Ω_op values for each JT scan point. If NumPy is available, a NumPy array is returned; otherwise a plain Python list of floats is returned.
        
        Raises:
            ValueError: If both overlap_matrices and floquet_eigenvectors are missing.
            ValueError: If the fallback (eigenvector condition number) is required but NumPy is not available.
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
        Produce calibration statistics summarizing the order parameter (ξ), AP' and Ω_op without applying validation thresholds.
        
        Computes ξ, the AP' series (second derivative of ξ normalized), and the Ω_op series (protractor noise) and returns summary statistics and a brief recommendation for scaling AP'. Uses NumPy when available; falls back to internal helpers otherwise.
        
        Returns:
            dict: A calibration report with keys:
                - mode (str): Always "CALIBRATION".
                - stats (dict): Numeric summaries:
                    - xi_mean (float): Mean of the ξ series.
                    - xi_max (float): Maximum of the ξ series.
                    - ap_prime_min (float): Minimum value of the AP' series.
                    - ap_prime_std (float): Standard deviation of the AP' series.
                    - omega_op_max (float): Maximum of the Ω_op series.
                - recommendation (str): Textual suggestion about AP' scaling.
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
        Run Tier 1 validation checks (CR-4 and CR-5) on the dataset and produce a RatchetResult.
        
        This computes the order parameter (ξ), its second-derivative series (AP'), and the protractor noise series (Ω_op), then evaluates them against the class thresholds:
        - Acceleration trigger when AP' < AP_THRESHOLD.
        - Crosstalk trigger when Ω_op > OMEGA_THRESHOLD.
        
        The final verdict marks the run as passed when there are at least two acceleration triggers and zero crosstalk breaches.
        
        Parameters:
            strict (bool): Kept for compatibility with the public API; currently not used by the implementation.
        
        Returns:
            RatchetResult: Contains:
              - ap_prime_series: series of AP' values.
              - omega_op_series: series of Ω_op values.
              - xi_series: series of ξ values.
              - trigger_indices: sorted indices where either threshold is triggered.
              - critical_crossings: count of acceleration triggers.
              - crosstalk_breaches: count of crosstalk triggers.
              - passed: `true` if the run meets the pass criteria described above, `false` otherwise.
              - mode: the string "VALIDATION".
              - meta: dictionary with applied thresholds (`ap_threshold`, `omega_threshold`).
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
    """
    Run the Cosmic Ratchet Tier 1 validation using the provided dataset.
    
    Parameters:
        dataset: Dataset object containing the required fields for SPG validation (e.g., JT_scan_points, floquet_eigenvalues and either overlap_matrices or floquet_eigenvectors).
    
    Returns:
        RatchetResult: Structured validation outcome containing AP' series, Ω_op series, ξ series, trigger indices, counts of critical crossings and crosstalk breaches, overall pass flag, mode, and meta information.
    """
    validator = CosmicRatchetValidator(dataset)
    return validator.run_validation()


def validate_theorem_spg(dataset) -> Dict[str, Any]:
    """
    Run the SPG cosmic-ratchet validation on a dataset and return a compact summary of Tier 1 results.
    
    Parameters:
        dataset: The dataset object to validate; must be compatible with the CosmicRatchetValidator input (contains JT scan points and Floquet data).
    
    Returns:
        A dictionary with:
            omega_max (float): Maximum observed protractor noise (Ω_op) across the scan.
            critical_instanton_events (int): Number of acceleration-trigger events (AP' threshold crossings).
            crosstalk_breaches (int): Number of protractor-noise breaches (Ω_op threshold crossings).
            passed (bool): Whether the dataset passed Tier 1 (CR-4/CR-5) validation.
    """
    result = run_cosmic_ratchet_test(dataset)
    omega_series = result.omega_op_series if HAVE_NUMPY else list(result.omega_op_series)
    return {
        "omega_max": float(np.max(omega_series)) if HAVE_NUMPY else _max_value(omega_series),
        "critical_instanton_events": int(result.critical_crossings),
        "crosstalk_breaches": int(result.crosstalk_breaches),
        "passed": bool(result.passed),
    }


# CLI Hook (UPDATED)
def run_spg_with_mode(config_path: str, mode: str = "validate"):
    """
    Run the SPG workflow in either calibration or validation mode using a configuration file.
    
    Loads the configuration at config_path, builds the dataset, instantiates a CosmicRatchetValidator, and either
    runs calibration (prints a JSON report) or runs validation (prints a summary, writes spg_results.json to an
    artifacts directory, and prints a pass/fail message).
    
    Parameters:
        config_path (str): Path to the configuration file used to locate the dataset and settings.
        mode (str): Operation mode, either "calibrate" to run the calibration workflow or any other value
            (default "validate") to run the validation workflow.
    
    Side effects:
        - Prints progress and results to stdout.
        - Creates the artifacts directory (cfg["artifacts_dir"] or ./artifacts) if missing.
        - Writes artifacts/spg_results.json containing the validation results when running validation.
    """
    from .cli import _load_config
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