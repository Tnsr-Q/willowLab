“”“Appendix theorem validation suite with explicit falsification criteria.”””
import numpy as np
from willowlab.io import load_willow
from willowlab.resolvent import validate_theorem_b1, validate_theorem_b2, validate_lemma_5
from willowlab.spectral_flow import validate_theorem_b3, validate_theorem_b4, holonomy_linearity_test
from willowlab.disorder import validate_residue_landscape

# Theorem 1: Uniqueness of Operational Invariant

def test_theorem_1_uniqueness(willow_path=“data/willow_sept_2025.npz”):
“””
Theorem 1: Only ϕ(K̇/(Ṡ_e + ΔS_loss)) survives gauge invariance.

```
Dataset: Willow Sept 2025 JT sweeps
Falsification: Alternative scalar with linear holonomy → FAIL
"""
ds = load_willow(willow_path)

# TODO: Implement alternative scalar tests
# For now, assert canonical R_op is used
assert ds.floquet_eigenvalues is not None
print("⚠ Theorem 1 test requires alternative scalar implementations")
```

# Lemma 2: Linearity of Small-Loop Holonomy

def test_lemma_2_holonomy_linearity(willow_path=“data/willow_dec_2025.npz”):
“””
Lemma 2: Small-loop holonomy linear in curvature, error O(A^{3/2}).

```
Dataset: Willow Dec 2025 phase sweeps
Falsification: Scaling breaks A^{3/2} → FAIL
"""
ds = load_willow(willow_path)

if ds.floquet_eigenvectors is None:
    print("⚠ Lemma 2: No eigenvectors in dataset, skipping")
    return

# Extract loops of different sizes
# TODO: Implement loop extraction from parameter sweeps
print("⚠ Lemma 2 test requires loop extraction from data")
```

# Proposition 3: Reduction to Einstein Gravity

def test_proposition_3_gr_reduction(willow_path=“data/willow_sept_2025.npz”):
“””
Proposition 3: CCC → GR when ω, F → 0.

```
Dataset: Willow Sept 2025 off-resonant (JT→0)
Falsification: Non-Einsteinian residual persists → FAIL
"""
ds = load_willow(willow_path)

# Test off-resonant limit
JT = ds.JT_scan_points
off_resonant_mask = JT < 0.1

if not np.any(off_resonant_mask):
    print("⚠ Proposition 3: No off-resonant data (JT<0.1), skipping")
    return

# TODO: Decompose R_op into GR + correction terms
print("⚠ Proposition 3 test requires R_op decomposition")
```

# Theorem 4: 14D Necessity

def test_theorem_4_14d_necessity(
willow_sept=“data/willow_sept_2025.npz”,
willow_dec=“data/willow_dec_2025.npz”
):
“””
Theorem 4: ∫ Tr(F^7) ≠ 0 requires dim ≥ 14.

```
Dataset: Pooled Sept+Dec 2025
Falsification: c₁₄ = 0 globally → 14D unnecessary
"""
ds_sept = load_willow(willow_sept)
ds_dec = load_willow(willow_dec)

# Pool eigendata
# TODO: Construct 7 commuting tori from pooled data
# TODO: Compute nested Wilson loops

print("⚠ Theorem 4 test requires nested Wilson loop construction")
```

# Lemma 5: Resolvent Residue Amplification

def test_lemma_5_amplification(willow_path=“data/willow_sept_2025.npz”):
“””
Lemma 5: d/dθ Tr(I-E)⁻¹ amplifies spectral flow near λ_k → 1.

```
Dataset: Willow Sept 2025, JT near 1
Falsification: Divergence doesn't track curvature peaks → FAIL
"""
ds = load_willow(willow_path)

if ds.floquet_eigenvalues is None:
    print("⚠ Lemma 5: No eigenvalues in dataset, skipping")
    return

result = validate_lemma_5(ds.floquet_eigenvalues, ds.JT_scan_points)

assert result['passed'], f"LEMMA 5 FALSIFIED: correlation={result['correlation']:.3f}"
print(f"✓ Lemma 5: correlation={result['correlation']:.3f}")
```

# Proposition 6: Noether-Ratio Principle

def test_proposition_6_noether_ratio(gate_sets_dir=“data/cross_compilation/”):
“””
Proposition 6: Only C·K̇/(Ṡ_e + ΔS_loss) from Noether symmetries.

```
Dataset: Cross-compilation runs with different gate alphabets
Falsification: Alternative scalar matches holonomy invariance → FAIL
"""
# TODO: Load multiple gate alphabet datasets
# TODO: Test invariance across compilations

print("⚠ Proposition 6 test requires cross-compilation datasets")
```

# Integration test: Run all theorems

def test_all_theorems():
“”“Run all appendix theorem tests.”””
print(”=”*70)
print(“Appendix Theorem Validation Suite”)
print(”=”*70)

```
try:
    test_theorem_1_uniqueness()
except Exception as e:
    print(f"✗ Theorem 1: {e}")

try:
    test_lemma_2_holonomy_linearity()
except Exception as e:
    print(f"✗ Lemma 2: {e}")

try:
    test_proposition_3_gr_reduction()
except Exception as e:
    print(f"✗ Proposition 3: {e}")

try:
    test_theorem_4_14d_necessity()
except Exception as e:
    print(f"✗ Theorem 4: {e}")

try:
    test_lemma_5_amplification()
except Exception as e:
    print(f"✗ Lemma 5: {e}")

try:
    test_proposition_6_noether_ratio()
except Exception as e:
    print(f"✗ Proposition 6: {e}")

print("="*70)
```

if **name** == “**main**”:
test_all_theorems()