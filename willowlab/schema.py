from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class WillowDataset:
    JT_scan_points: np.ndarray             # [T]
    floquet_eigenvalues: Optional[np.ndarray] = None # [T,N]
    floquet_eigenvectors: Optional[np.ndarray] = None # [T,N,N]
    floquet_operators: Optional[np.ndarray] = None # [T,N,N]
    resolvent_trace: Optional[np.ndarray] = None   # [T] complex
    entropy: Optional[np.ndarray] = None           # [T]
    effective_energy: Optional[np.ndarray] = None  # [T]
    eta_oscillations: Optional[np.ndarray] = None  # [T]
    chern_mod2: Optional[np.ndarray] = None        # [T] in {0,1}
    spectral_flow_crossings: Optional[np.ndarray] = None # [T] ints
    overlap_matrices: Optional[np.ndarray] = None # [T,N,N] (for geometry)
    meta: Dict[str, Any] = field(default_factory=dict)

    def check_basic(self) -> None:
        assert self.JT_scan_points.ndim == 1
        T = self.JT_scan_points.shape[0]
        for name in ["floquet_eigenvalues","resolvent_trace","entropy","effective_energy",
                     "eta_oscillations","chern_mod2","spectral_flow_crossings"]:
            arr = getattr(self, name)
            if arr is not None:
                assert len(arr) == T, f"{name} length != T"
