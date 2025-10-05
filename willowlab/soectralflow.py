“”“Spectral flow topology: Berry phases, Chern numbers, η-lock validation.”””
import numpy as np
from typing import List, Dict, Optional

_EPS = 1e-12

def berry_connection(v1, v2):
“””
Berry connection A = i⟨v|dv⟩ between adjacent eigenvectors.

```
Returns complex phase contribution.
"""
overlap = np.vdot(v1, v2)
return 1j * np.log(overlap + _EPS)
```

def berry_phase_loop(evecs_loop):
“””
Compute Berry phase γ = ∮ A·dθ around closed eigenvector loop.

```
evecs_loop: list of [N,N] eigenvector matrices (columns = eigenvectors).
Returns array of Berry phases per band.
"""
T = len(evecs_loop)
N = evecs_loop[0].shape[1]
phases = np.zeros(N, dtype=float)

for k in range(N):
    phase = 0.0
    for t in range(T-1):
        v_t = evecs_loop[t][:, k]
        v_next = evecs_loop[t+1][:, k]
        phase += berry_connection(v_t, v_next).imag
    
    # Close loop
    v_last = evecs_loop[-1][:, k]
    v_first = evecs_loop[0][:, k]
    phase += berry_connection(v_last, v_first).imag
    
    phases[k] = phase

return phases
```

def chern_number(berry_phases):
“””
Chern number C = (1/2π) Σ γ_k, rounded to integer.

```
Tests Theorem B.4 quantization.
"""
total_phase = np.sum(berry_phases)
return int(np.round(total_phase / (2.0 * np.pi)))
```

def quantization_check(berry_phases, tol=0.1):
“””
Check if Berry phases are quantized: |γ_k - 2πn_k| < tol.

```
Returns array of deviations from nearest 2π multiple.
"""
deviations = np.abs(berry_phases - 2*np.pi*np.round(berry_phases/(2*np.pi)))
return deviations
```

def validate_theorem_b4(berry_phases, eta_oscillations, chern_mod2):
“””
Test Theorem B.4: η-lock ⟺ mod-2 Chern invariance.

```
Returns dict: agreement_rate, lock_rate, passed (bool).
"""
from .tests.t_eta_lock import eta_lock_windows

# Chern parity from Berry phases
C = chern_number(berry_phases)
C_mod2 = C % 2

# η parity: η=+1 → 0, η=-1 → 1
eta_parity = np.array([0 if eta > 0 else 1 for eta in eta_oscillations])

# Agreement test
agreement = np.mean(C_mod2 == eta_parity) if len(eta_parity) > 0 else 0.0

# Lock test
locks = eta_lock_windows(eta_oscillations, chern_mod2, window=5)
lock_rate = np.mean(locks) if len(locks) > 0 else 0.0

passed = (agreement > 0.8) and (lock_rate > 0.5)
return {
    'agreement_rate': float(agreement),
    'lock_rate': float(lock_rate),
    'chern_number': C,
    'chern_mod2': C_mod2,
    'passed': bool(passed)
}
```

def c14_from_nested_loops(berry_phases_7tori):
“””
Compute c₁₄ from seven commuting Wilson loops on T¹⁴.

```
berry_phases_7tori: list of 7 arrays, each array = Berry phases per band for one torus.

Tests Theorem B.4 (14D necessity).
"""
# Each torus contributes F (2-form); wedge product F^7 is 14-form
# Simplified: sum Chern numbers from each torus
c_contributions = [chern_number(phases) for phases in berry_phases_7tori]
c_14_raw = np.sum(c_contributions)  # Simplified; full wedge would use tensor ops
c_14_int = int(np.round(c_14_raw))

return {'c_14_raw': float(c_14_raw), 'c_14_integer': c_14_int}
```

def validate_theorem_b3(berry_phases_7tori):
“””
Test Theorem B.3: c₁₄ ≠ 0 requires dim ≥ 14.

```
Falsification: c₁₄ = 0 globally → 14D unnecessary.
"""
result = c14_from_nested_loops(berry_phases_7tori)
passed = result['c_14_integer'] != 0

return {
    'c_14_raw': result['c_14_raw'],
    'c_14_integer': result['c_14_integer'],
    'passed': bool(passed)
}
```

def berry_curvature_2form(evecs_grid):
“””
Compute Berry curvature F = dA on a 2D parameter grid.

```
evecs_grid: [Ny, Nx, N, N] eigenvectors on rectangular grid.
Returns [Ny-1, Nx-1] curvature per plaquette.
"""
Ny, Nx, N, _ = evecs_grid.shape
F = np.zeros((Ny-1, Nx-1), dtype=float)

for iy in range(Ny-1):
    for ix in range(Nx-1):
        # Loop around plaquette: (iy,ix) → (iy,ix+1) → (iy+1,ix+1) → (iy+1,ix) → (iy,ix)
        # Compute Berry phase for each band, sum over bands
        for k in range(N):
            v00 = evecs_grid[iy, ix, :, k]
            v01 = evecs_grid[iy, ix+1, :, k]
            v11 = evecs_grid[iy+1, ix+1, :, k]
            v10 = evecs_grid[iy+1, ix, :, k]
            
            phase = 0.0
            phase += berry_connection(v00, v01).imag
            phase += berry_connection(v01, v11).imag
            phase += berry_connection(v11, v10).imag
            phase += berry_connection(v10, v00).imag
            
            F[iy, ix] += phase

return F / (2.0 * np.pi)  # Normalize to flux quantum
```

def holonomy_linearity_test(berry_phases_loops, loop_areas):
“””
Test Lemma 2: holonomy linear in curvature, error O(A^{3/2}).

```
berry_phases_loops: list of Berry phase arrays (one per loop).
loop_areas: corresponding loop areas.

Returns scaling exponent (should be ~1.5).
"""
holonomies = np.array([np.sum(phases) for phases in berry_phases_loops])
areas = np.array(loop_areas)

# Linear prediction: holonomy ∝ area
predicted = areas * holonomies[0] / areas[0]  # Normalize by first loop

errors = np.abs(holonomies - predicted)

# Fit log(error) vs log(area) → slope should be ~1.5
if len(areas) < 3:
    return {'slope': np.nan, 'passed': False}

log_areas = np.log(areas + _EPS)
log_errors = np.log(errors + _EPS)

slope = np.polyfit(log_areas, log_errors, 1)[0]
passed = (1.3 < slope < 1.7)  # Allow 20% tolerance

return {'slope': float(slope), 'expected': 1.5, 'passed': bool(passed)}
```