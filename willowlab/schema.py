from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

try:  # pragma: no cover - exercised indirectly in environments with NumPy
    import numpy as _np
    ArrayLike = _np.ndarray  # type: ignore[attr-defined]
except (ModuleNotFoundError, AttributeError):  # pragma: no cover - testing shim fallback
    from willowlab import _numpy_shim as _np  # type: ignore

    ArrayLike = Any

@dataclass(frozen=True)
class WillowDataset:
    JT_scan_points: ArrayLike             # [T]
    floquet_eigenvalues: Optional[ArrayLike] = None # [T,N]
    floquet_eigenvectors: Optional[ArrayLike] = None # [T,N,N]
    floquet_operators: Optional[ArrayLike] = None # [T,N,N]
    resolvent_trace: Optional[ArrayLike] = None   # [T] complex
    entropy: Optional[ArrayLike] = None           # [T]
    effective_energy: Optional[ArrayLike] = None  # [T]
    eta_oscillations: Optional[ArrayLike] = None  # [T]
    chern_mod2: Optional[ArrayLike] = None        # [T] in {0,1}
    spectral_flow_crossings: Optional[ArrayLike] = None # [T] ints
    overlap_matrices: Optional[ArrayLike] = None # [T,N,N] (for geometry)
    meta: Dict[str, Any] = field(default_factory=dict)

    def check_basic(self) -> None:
        assert _np.asarray(self.JT_scan_points).ndim == 1
        T = _np.asarray(self.JT_scan_points).shape[0]
        for name in ["floquet_eigenvalues","resolvent_trace","entropy","effective_energy",
                     "eta_oscillations","chern_mod2","spectral_flow_crossings"]:
            arr = getattr(self, name)
            if arr is not None:
                assert len(arr) == T, f"{name} length != T"


@dataclass
class TheoremValidationResult:
    """Standardized record of a single theorem validation."""

    theorem_id: str
    dataset_used: str
    falsification_criteria: Dict[str, Any]
    actual_results: Dict[str, Any]
    validated: bool
    failure_reason: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class NobelValidationSuite:
    """Container for Nobel committee-facing validation summaries."""

    suite_id: str = "ccc_nobel_validation_2025"
    theorems_tested: List[str] = field(default_factory=list)
    datasets_used: List[str] = field(default_factory=list)
    results: List[TheoremValidationResult] = field(default_factory=list)
    overall_status: Optional[bool] = None

    def generate_nobel_report(self) -> Dict[str, Any]:
        """Create a structured report for external reviewers."""

        self.theorems_tested = [result.theorem_id for result in self.results]
        dataset_order = []
        for result in self.results:
            if result.dataset_used not in dataset_order:
                dataset_order.append(result.dataset_used)
        self.datasets_used = dataset_order

        return {
            "framework": "CCC/Mecha Operational Gravity",
            "validation_date": datetime.now(timezone.utc).isoformat(),
            "theorems_tested": self.theorems_tested,
            "theorems_validated": sum(1 for r in self.results if r.validated),
            "critical_falsification_tests": [
                {
                    "theorem": r.theorem_id,
                    "dataset": r.dataset_used,
                    "falsification_criteria": r.falsification_criteria,
                    "result": "PASS" if r.validated else f"FAIL: {r.failure_reason}",
                    "scientific_implication": self._get_implication(r.theorem_id),
                }
                for r in self.results
            ],
            "overall_conclusion": (
                "ALL CRITICAL THEOREMS VALIDATED"
                if self.overall_status
                else "FRAMEWORK FALSIFIED - REQUIRES REVISION"
            ),
        }

    def _get_implication(self, theorem_id: str) -> str:
        implications = {
            "B.1": "Spectral signatures map uniquely onto entanglement thermodynamics.",
            "B.2": "Divergent resolvent behavior necessitates observable exceptional points.",
            "B.3": "Nested 14D Wilson loops remain essential for higher-form protection.",
            "B.4": "η-lock phenomena statistically enforce mod-2 Chern protection.",
            "B.5": "Residue landscapes faithfully identify exceptional-point saddles.",
        }
        return implications.get(theorem_id, "Implication not catalogued.")
