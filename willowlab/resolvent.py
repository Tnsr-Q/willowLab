“”“Resolvent trace witness for CCC validation.”””
import numpy as np
from typing import Dict, Tuple, Optional

_EPS = 1e-12

def _phase_align(evals):
“”“Remove global U(1) phase per time-step.”””
T, N = evals.shape
aligned = np.empty_like(evals, dtype=np.complex128)
for t in range(T):
phi = np.angle(np.prod(evals[t])) / N
aligned[t] = evals[t] * np.exp(-1j*phi)
return aligned

def trace_resolvent_from_evals(evals, safe=True):
“””
Compute |Tr(I-U)^{-1}| from eigenvalues, avoiding cancellation.

```
If safe=True, use angle-based surrogate on |λ|≈1:
    on circle: 1/(2|sin(θ/2)|)
    off circle: |1/(1-λ)|
"""
if evals.ndim == 1: evals = evals[None, :]
T, N = evals.shape
if not safe:
    return np.abs(np.sum(1.0/(1.0 - evals + _EPS), axis=1))

# Safe path
out = np.zeros(T, dtype=float)
for t in range(T):
    angles = np.angle(evals[t])
    on_circle = np.isclose(np.abs(evals[t]), 1.0, atol=1e-6)
    mag = np.empty(N, dtype=float)
    sin_half = np.maximum(np.abs(np.sin(angles/2.0)), _EPS)
    mag[on_circle] = 1.0/(2.0*sin_half[on_circle])
    mag[~on_circle] = 1.0/np.abs(1.0 - evals[t][~on_circle])
    out[t] = np.sum(mag)
return out
```

def r_op_from_trace(trace_abs, N, kappa=np.sqrt(3)):
“”“R_op proxy: κ·Tr/N (dimensionless).”””
return kappa * trace_abs / N

def resolvent_scan(evals, JT, align_phase=True, kappa=np.sqrt(3)):
“””
Scan resolvent trace and R_op across JT parameter space.

```
Returns dict with:
    JT, trace_abs, r_op, min_dist_to_one, peak_jt, peak_r_op
"""
if align_phase:
    evals = _phase_align(evals)

trace_abs = trace_resolvent_from_evals(evals, safe=True)
N = evals.shape[1]
r_op = r_op_from_trace(trace_abs, N, kappa)
min_d1 = np.min(np.abs(1.0 - evals), axis=1)

peak_idx = np.argmax(r_op)
return {
    'JT': JT,
    'trace_abs': trace_abs,
    'r_op': r_op,
    'min_dist_to_one': min_d1,
    'peak_jt': float(JT[peak_idx]),
    'peak_r_op': float(r_op[peak_idx]),
    'peak_idx': peak_idx
}
```

def spectral_temperature(trace_abs, JT):
“””
T_spec = (|d²(log Tr)/dJT²| + ε)^{-1}

```
Used in Theorem B.1 (Spectral-Entanglement Duality).
"""
log_tr = np.log(trace_abs + _EPS)
d2 = np.gradient(np.gradient(log_tr, JT), JT)
return 1.0 / (np.abs(d2) + _EPS)
```

def validate_theorem_b1(trace_abs, JT, entropy, effective_energy, window=0.05):
“””
Test Theorem B.1: T_spec ~ T_ent with slope 1±0.1, R²>0.9.

```
Returns dict: slope, r2, passed (bool).
"""
from .tests.t_spec_ent import entanglement_temperature

T_spec = spectral_temperature(trace_abs, JT)
T_ent = entanglement_temperature(entropy, effective_energy)

mask = (JT > 1.0 - window) & (JT < 1.0 + window)
if np.sum(mask) < 4:
    return {'slope': np.nan, 'r2': np.nan, 'passed': False}

a = np.log(T_spec[mask] + _EPS)
b = np.log(T_ent[mask] + _EPS)
slope = np.polyfit(a, b, 1)[0]
r2 = np.corrcoef(a, b)[0,1]**2

passed = (abs(slope - 1.0) < 0.1) and (r2 > 0.9)
return {'slope': float(slope), 'r2': float(r2), 'passed': bool(passed)}
```

def validate_theorem_b2(evals, trace_abs, tol=1e-8):
“””
Test Theorem B.2: Divergences ⟺ Exceptional Points.

```
Returns array of bools (per time-step): divergent but no EP flag → False.
"""
T = len(evals)
passed = np.ones(T, dtype=bool)

for t in range(T):
    divergent = trace_abs[t] > 1e6
    if not divergent: continue
    
    # EP signature: det(I-E)≈0 or eigenvalue degeneracy
    I_minus_E = np.eye(len(evals[t])) - np.diag(evals[t])
    det_small = np.abs(np.linalg.det(I_minus_E)) < tol
    
    gaps = np.diff(np.sort(np.abs(evals[t])))
    degenerate = np.min(gaps) < tol if len(gaps) > 0 else False
    
    has_ep = det_small or degenerate
    if divergent and not has_ep:
        passed[t] = False

return passed
```

def validate_lemma_5(evals, JT):
“””
Test Lemma 5: d(Tr)/dθ = Σ λ̇_k/(1-λ_k)².

```
Returns correlation between numerical and analytical derivatives.
"""
trace_abs = trace_resolvent_from_evals(evals, safe=False)
d_trace_num = np.gradient(trace_abs, JT)

d_trace_ana = np.zeros(len(JT))
for t in range(1, len(JT)):
    d_lambda = (evals[t] - evals[t-1]) / (JT[t] - JT[t-1])
    d_trace_ana[t] = np.sum(d_lambda / (1.0 - evals[t] + _EPS)**2).real

# Correlation near JT=1
mask = (JT > 0.95) & (JT < 1.05)
if np.sum(mask) < 4:
    return {'correlation': np.nan, 'passed': False}

corr = np.corrcoef(d_trace_num[mask], d_trace_ana[mask])[0,1]
return {'correlation': float(corr), 'passed': bool(abs(corr) > 0.8)}
```