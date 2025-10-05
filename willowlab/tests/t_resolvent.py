“”“Tests for resolvent trace witness and theorem validation.”””
import numpy as np
from willowlab.resolvent import (
resolvent_scan, spectral_temperature, validate_theorem_b1,
validate_theorem_b2, validate_lemma_5
)
from willowlab.spectral_flow import (
berry_phase_loop, chern_number, validate_theorem_b3,
validate_theorem_b4, holonomy_linearity_test
)
from willowlab.disorder import disorder_scan, optimal_disorder, validate_residue_landscape

def test_resolvent_scan_toy():
“”“Basic resolvent scan on toy eigenvalues.”””
T, N = 50, 10
JT = np.linspace(0.5, 1.5, T)

```
# Toy eigenvalues: approach unity near JT=1
evals = np.empty((T, N), dtype=np.complex128)
for t, jt in enumerate(JT):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = 1.0 - 0.1*np.abs(jt - 1.0)**2  # Close to 1 near JT=1
    evals[t] = radii * np.exp(1j*angles)

result = resolvent_scan(evals, JT)

assert 'peak_jt' in result
assert 'peak_r_op' in result
assert 0.9 < result['peak_jt'] < 1.1  # Peak near JT=1
print(f"✓ Resolvent scan: peak at JT={result['peak_jt']:.3f}")
```

def test_theorem_b1_duality():
“”“Test Theorem B.1: Spectral-Entanglement Duality.”””
T = 50
JT = np.linspace(0.5, 1.5, T)

```
# Mock trace and entropy data with correct duality
trace_abs = np.exp(0.02*JT**3 + 0.19*JT**2)
entropy = 4*0.02*JT**3 + 2*0.19*JT**2
effective_energy = JT**2

result = validate_theorem_b1(trace_abs, JT, entropy, effective_energy)

assert 'slope' in result
assert 'r2' in result
assert 'passed' in result
print(f"✓ Theorem B.1: slope={result['slope']:.3f}, R²={result['r2']:.3f}, pass={result['passed']}")
```

def test_theorem_b2_exceptional_points():
“”“Test Theorem B.2: Divergences ⟺ EPs.”””
T, N = 20, 5

```
# Create eigenvalues with one near-degenerate point
evals = np.random.rand(T, N) * 0.5 + 0.3  # All real, positive
evals = np.exp(2j*np.pi*evals)  # On unit circle

# Force divergence at t=10 with eigenvalue degeneracy
evals[10, 0] = 0.999999 + 0j
evals[10, 1] = 0.999999 + 0j

trace_abs = np.ones(T)
trace_abs[10] = 1e7  # Divergence

passed = validate_theorem_b2(evals, trace_abs)

assert passed[10] == True  # Should detect EP at divergence
print(f"✓ Theorem B.2: {np.sum(passed)}/{T} points pass EP check")
```

def test_lemma_5_amplification():
“”“Test Lemma 5: Resolvent derivative amplifies spectral flow.”””
T, N = 30, 8
JT = np.linspace(0.8, 1.2, T)

```
# Eigenvalues with smooth flow near JT=1
evals = np.empty((T, N), dtype=np.complex128)
for t, jt in enumerate(JT):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) + 0.1*jt
    radii = 1.0 - 0.05
```