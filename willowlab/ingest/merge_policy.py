#willowlab/ingest/merge_policy.py
import numpy as np
from typing import Dict, Tuple

def derive_from_operators(U):
    """Compute eigenpairs from Floquet operators U."""
    U = np.asarray(U)
    if U.ndim == 3:
        T, N, _ = U.shape
        evals = np.empty((T, N), dtype=np.complex128)
        evecs = np.empty_like(U)
        for t in range(T):
            w, V = np.linalg.eig(U[t])
            evals[t] = w; evecs[t] = V
        return evals, evecs
    elif U.ndim == 2:
        w, V = np.linalg.eig(U)
        return w[None, :], V[None, :, :]
    raise ValueError("U must be 2D or 3D array.")

def reconcile_eigenpairs(mapping: Dict, policy: str = "auto") -> Dict:
    """
    Resolve conflicts between supplied eigenpairs and operator-derived ones.
    policy in {'auto', 'prefer_operator', 'prefer_supplied'}
    """
    out = dict(mapping)
    has_U = 'floquet_operators' in out and out['floquet_operators'] is not None and out['floquet_operators'] != []
    has_ev = 'floquet_eigenvalues' in out and out['floquet_eigenvalues'] is not None and out['floquet_eigenvalues'] != []
    if not has_U and not has_ev:
        return out #nothing to do

    derived_evals = derived_evecs = None
    if has_U:
        try:
            derived_evals, derived_evecs = derive_from_operators(out['floquet_operators'])
        except Exception:
            derived_evals = derived_evecs = None

    if not has_ev and derived_evals is not None:
        out['floquet_eigenvalues'] = derived_evals
        out.setdefault('floquet_eigenvectors', derived_evecs)
        return out

    if has_ev and derived_evals is None:
        return out

    # both present, shapes may disagree
    ev_sup = np.asarray(out['floquet_eigenvalues'])
    JT = np.asarray(out.get('JT_scan_points')) if 'JT_scan_points' in out else None
    sup_ok = ev_sup.ndim == 2
    drv_ok = derived_evals is not None and derived_evals.ndim == 2

    # shape sanity
    def _score(ev):
        if JT is None: return 0
        return int(ev.shape[0] == JT.shape[0])

    if policy == "prefer_operator" and drv_ok:
        out['floquet_eigenvalues'] = derived_evals
        out.setdefault('floquet_eigenvectors', derived_evecs)
    elif policy == "prefer_supplied" and sup_ok:
        # keep supplied; only fill evecs if missing
        if 'floquet_eigenvectors' not in out and derived_evecs is not None and ev_sup.shape == derived_evals.shape:
            out['floquet_eigenvectors'] = derived_evecs
    else: # auto
        # prefer the one that matches JT length; tie-breaker: larger N (more bands)
        if drv_ok and (_score(derived_evals) > _score(ev_sup)):
            out['floquet_eigenvalues'] = derived_evals
            out.setdefault('floquet_eigenvectors', derived_evecs)
        elif drv_ok and sup_ok and _score(derived_evals) == _score(ev_sup):
            choose_derived = (derived_evals.shape[1] >= ev_sup.shape[1])
            if choose_derived:
                out['floquet_eigenvalues'] = derived_evals
                out.setdefault('floquet_eigenvectors', derived_evecs)
        # else keep supplied
    # else keep supplied
    return out
