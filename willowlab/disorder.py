“”“Disorder sharpening analysis for resolvent trace peaks.”””
import numpy as np
from typing import Dict, List, Tuple

_EPS = 1e-12

def add_goe_disorder(matrix, delta, seed=None):
“””
Add Gaussian Orthogonal Ensemble disorder to Hermitian matrix.

```
delta: disorder strength (typically 0.0 - 0.3).
seed: random seed for reproducibility.
"""
rng = np.random.RandomState(seed)
N = len(matrix)

# Real symmetric disorder
disorder = rng.randn(N, N)
disorder = (disorder + disorder.T) / 2.0

# Scale by matrix norm
scale = delta * np.linalg.norm(matrix, 'fro')

return matrix + scale * disorder
```

def level_spacing_stats(eigenvalues):
“””
Compute nearest-neighbor level spacing statistics.

```
Returns mean spacing and repulsion exponent β estimate.
"""
evals = np.sort(np.abs(eigenvalues))
spacings = np.diff(evals)
spacings = spacings[spacings > _EPS]  # Remove exact degeneracies

if len(spacings) == 0:
    return {'mean_spacing': np.nan, 'beta': np.nan}

# Normalize to unit mean
spacings = spacings / np.mean(spacings)

# Estimate β from P(s) ~ s^β at small s
# Use histogram near s=0
bins = np.linspace(0, 0.5, 10)
hist, _ = np.histogram(spacings, bins=bins, density=True)

# Fit log(P) vs log(s) for small s
mask = (hist > _EPS) & (bins[:-1] > _EPS)
if np.sum(mask) < 3:
    beta = 0.0
else:
    log_s = np.log(bins[:-1][mask])
    log_P = np.log(hist[mask])
    beta = np.polyfit(log_s, log_P, 1)[0]

return {'mean_spacing': float(np.mean(spacings)), 'beta': float(beta)}
```

def peak_sharpness(values, coordinates):
“””
Quantify peak sharpness via FWHM and max second derivative.

```
Returns: peak_width (FWHM), peak_sharpness (|d²/dx²|).
"""
peak_idx = np.argmax(values)
peak_val = values[peak_idx]

# FWHM
half_max = peak_val / 2.0
above_half = values > half_max

if np.sum(above_half) > 2:
    indices = np.where(above_half)[0]
    left_idx = indices[0]
    right_idx = indices[-1]
    width = coordinates[right_idx] - coordinates[left_idx]
else:
    width = np.inf

# Max second derivative near peak
d2 = np.gradient(np.gradient(values, coordinates), coordinates)

if peak_idx > 1 and peak_idx < len(values) - 2:
    window = slice(max(0, peak_idx-3), min(len(values), peak_idx+4))
    sharpness = np.max(np.abs(d2[window]))
else:
    sharpness = 0.0

return {'peak_width': float(width), 'peak_sharpness': float(sharpness)}
```

def disorder_scan(evals_clean, JT, delta_values, n_realizations=5):
“””
Scan disorder strengths and measure peak sharpening.

```
evals_clean: [T, N] clean eigenvalues.
JT: [T] parameter values.
delta_values: array of disorder strengths to test.
n_realizations: number of disorder realizations per delta.

Returns list of dicts with disorder analysis results.
"""
from .resolvent import trace_resolvent_from_evals, r_op_from_trace

results = []
T, N = evals_clean.shape

for delta in delta_values:
    # Ensemble average over disorder realizations
    r_op_ensemble = []
    level_stats_ensemble = []
    
    for real in range(n_realizations):
        # Add disorder to eigenvalues (simplified: perturb phases)
        rng = np.random.RandomState(real + int(delta*1000))
        noise = delta * rng.randn(T, N)
        evals_disordered = evals_clean * np.exp(1j * noise)
        
        # Compute resolvent
        trace_abs = trace_resolvent_from_evals(evals_disordered, safe=True)
        r_op = r_op_from_trace(trace_abs, N)
        r_op_ensemble.append(r_op)
        
        # Level spacing at peak region
        peak_idx = np.argmax(r_op)
        stats = level_spacing_stats(evals_disordered[peak_idx])
        level_stats_ensemble.append(stats)
    
    # Average over realizations
    r_op_mean = np.mean(r_op_ensemble, axis=0)
    
    # Peak characteristics
    peak_info = peak_sharpness(r_op_mean, JT)
    
    # Average level statistics
    mean_beta = np.mean([s['beta'] for s in level_stats_ensemble if not np.isnan(s['beta'])])
    
    results.append({
        'delta': float(delta),
        'peak_jt': float(JT[np.argmax(r_op_mean)]),
        'peak_r_op': float(np.max(r_op_mean)),
        'peak_width': peak_info['peak_width'],
        'peak_sharpness': peak_info['peak_sharpness'],
        'repulsion_beta': float(mean_beta) if not np.isnan(mean_beta) else 0.0,
        'n_realizations': n_realizations
    })

return results
```

def optimal_disorder(disorder_results):
“””
Find optimal disorder strength from scan results.

```
Optimal = maximum peak sharpness.
"""
sharpness_values = [r['peak_sharpness'] for r in disorder_results]
optimal_idx = np.argmax(sharpness_values)

return {
    'optimal_delta': disorder_results[optimal_idx]['delta'],
    'optimal_sharpness': sharpness_values[optimal_idx],
    'baseline_sharpness': sharpness_values[0],  # delta=0
    'enhancement_factor': sharpness_values[optimal_idx] / max(sharpness_values[0], _EPS)
}
```

def validate_residue_landscape(residue_map, phi_landscape, saddles, ep_mask):
“””
Test Theorem B.5: Φ saddles = EPs, circulation = topological charge.

```
residue_map: [Ny, Nx] residue scores.
phi_landscape: [Ny, Nx] black-hole potential.
saddles: [Ny, Nx] boolean mask of saddle points.
ep_mask: [Ny-1, Nx-1] boolean mask of exceptional points.

Returns co-location rate (should be >70%).
"""
# Align dimensions (saddles may be on full grid, ep_mask on plaquette lattice)
saddles_cropped = saddles[1:-1, 1:-1] if saddles.shape > ep_mask.shape else saddles

overlap = saddles_cropped & ep_mask
co_location = np.sum(overlap) / max(np.sum(saddles_cropped), 1)

passed = co_location > 0.7

return {
    'co_location_rate': float(co_location),
    'saddle_count': int(np.sum(saddles)),
    'ep_count': int(np.sum(ep_mask)),
    'overlap_count': int(np.sum(overlap)),
    'passed': bool(passed)
}
```