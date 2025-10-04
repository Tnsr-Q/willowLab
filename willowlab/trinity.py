import numpy as np
from schema import WillowDataset

#--- tiny epsilon + helpers (per your spec) ---
_EPS = 1e-18

def _phase_align_evals(evals):
    T, N = evals.shape
    aligned = np.empty_like(evals, dtype=np.complex128)
    for t in range(T):
        phi = np.angle(np.linalg.det(np.diag(evals[t])) + 0j) / N
        aligned[t] = evals[t] * np.exp(-1j* phi)
    return aligned

# surrogate |Tr(I-U)⁻¹| robust to cancellations
def _trace_resolvent_abs_from_phase(evals):
    angles = np.angle(evals)
    on_circle = np.isclose(np.abs(evals), 1.0, atol=1e-6)
    mag = np.empty_like(evals.real)
    sin_half = np.maximum(np.abs(np.sin(angles/2.0)), _EPS)
    mag[on_circle] = 1.0/(2.0*sin_half[on_circle])
    mag[~on_circle] = 1.0/np.abs(1.0 - evals[~on_circle])
    return np.sum(mag, axis=1)

def _min_dist_to_one(evals): return np.min(np.abs(1.0 - evals), axis=1)

def _solve_trace(I_minus_U):
    n = I_minus_U.shape[0]
    try:
        X = np.linalg.solve(I_minus_U, np.eye(n, dtype=np.complex128))
        return np.trace(X)
    except np.linalg.LinAlgError:
        return np.nan + 1j*np.nan

def _pinv_trace(I_minus_U, rcond=1e-12):
    U, s, Vh = np.linalg.svd(I_minus_U)
    s_inv = np.where(s > rcond * s.max(), 1.0/s, 0.0)
    X = (Vh.conj().T*s_inv) @ U.conj().T
    return np.trace(X)

def _det_winding(I_minus_U_series):
    dets = np.array([np.linalg.det(M) for M in I_minus_U_series], dtype=np.complex128)
    if dets.size < 4: return np.nan
    ang = np.unwrap(np.angle(dets))
    return (ang[-1] - ang[0]) / (2.0*np.pi)

def _band_track_by_overlap(evecs_list):
    T = len(evecs_list); N = evecs_list[0].shape[1]
    perms = np.zeros((T, N), dtype=int); perms[0] = np.arange(N)
    prev = evecs_list[0]
    for t in range(1,T):
        V = evecs_list[t]
        M = np.abs(prev.conj().T @ V)
        chosen = set(); order = np.zeros(N, dtype=int)
        for r in range(N):
            c = int(np.argmax(M[r]))
            while c in chosen:
                M[r,c] = -1.0
                c = int(np.argmax(M[r]))
            order[r] = c; chosen.add(c)
        perms[t] = order; prev = V[:,order]
    return perms

class WillowTrinityStep1:
    def __init__(self, ds: WillowDataset, align_phase=True):
        JT = np.asarray(ds.JT_scan_points); self.JT = JT; self.T = len(JT)
        self.U = ds.floquet_operators
        self.evals = ds.floquet_eigenvalues
        self.evecs = ds.floquet_eigenvectors

        if self.evals is None and self.U is None:
            raise ValueError("Provide either eigenvalues or operators.")

        if self.evals is None:
            evals = []; evecs = []
            for t in range(self.T):
                w, V = np.linalg.eig(self.U[t])
                evals.append(w); evecs.append(V)
            self.evals = np.stack(evals, axis=0)
            self.evecs = np.stack(evecs, axis=0)

        if self.evecs is not None:
            perms = _band_track_by_overlap([self.evecs[t] for t in range(self.T)])
            for t in range(self.T):
                self.evals[t] = self.evals[t][perms[t]]
                self.evecs[t] = self.evecs[t][:, perms[t]]

        if align_phase:
            self.evals = _phase_align_evals(self.evals)

    def compute_all(self, jt_star=1.0, window=0.05, big_abs=1e6, dyn_factor=100.0):
        JT = self.JT; T = self.T; evals = self.evals
        trace_abs = _trace_resolvent_abs_from_phase(evals)
        min_d1 = _min_dist_to_one(evals)
        trace_solve = trace_pinv = None; det_wind = np.nan

        if self.U is not None:
            I_minus_U_series = np.eye(self.U.shape[1], dtype=np.complex128)[None,:,:] - self.U
            trace_solve = np.array([_solve_trace(M) for M in I_minus_U_series])
            trace_pinv = np.array([_pinv_trace(M) for M in I_minus_U_series])
            mask = (JT >= jt_star - window) & (JT <= jt_star + window)
            idx = np.where(mask)[0]
            if idx.size >= 4:
                det_wind = _det_winding(I_minus_U_series[idx])

        ep_kappa = None
        if self.evecs is not None:
            ep_kappa = np.array([
                (np.linalg.svd(self.evecs[t], compute_uv=False)[0] /
                 max(np.linalg.svd(self.evecs[t], compute_uv=False)[-1], _EPS))
                for t in range(T)
            ], float)

        i_star = int(np.argmin(np.abs(JT - jt_star)))
        local = (JT >= jt_star - window) & (JT <= jt_star + window)
        local_else = local & (np.arange(T) != i_star)
        baseline = np.median(trace_abs[local_else]) if np.any(local_else) else \
                   np.median(trace_abs)
        explodes = (trace_abs[i_star] > big_abs) or (trace_abs[i_star] > dyn_factor * \
                                                     max(baseline, 1.0))
        return {
            "JT": JT, "JT_star_value": float(JT[i_star]), "JT_star_index": i_star,
            "traces": {"solve_trace": trace_solve, "pinv_trace": trace_pinv},
            "surrogate_abs": trace_abs, "min_dist_to_one": min_d1,
            "ep_condition_number": ep_kappa,
            "det_winding_near_star": None if np.isnan(det_wind) else float(det_wind),
            "decision": {
                "computational_trace_explodes": bool(explodes),
                "peak_abs_value": float(trace_abs[i_star]),
                "min_dist_at_star": float(min_d1[i_star]),
                "baseline_abs": float(baseline)
            }
        }
