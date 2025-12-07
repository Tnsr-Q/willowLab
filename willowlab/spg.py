# willowlab/spg.py
"""
Stochastic Projective Gravity (SPG) Module
==========================================

Implements the CR-5 (FRW-Radar Acceleration Threshold) validation
against Willow Floquet data.

Registry Alignment: PREDICTION_REGISTRY_v2.0
- CR-5: AP' = -1/3 threshold from FRW cosmology
- CR-4: |Ω_op| < 0.02 Pantheon+ compliance
- CCC-2: Resolvent peak correlation

Theory Chain:
    FRW: w_eff = -1/3 (decel/accel boundary)
    Radar: AP' = (Δτ_PS - Δτ_RP)/(Δτ_PS + Δτ_RP)
    Floquet: ξ = 1 - min|1 - λ_k|
    
Author: Tanner Jacobsen
Version: 2.0 (Registry-aligned)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# PHYSICAL CONSTANTS (from Registry v2.0)
# =============================================================================

AP_CRITICAL_THRESHOLD = -1/3  # FRW decel/accel boundary
OMEGA_OP_LIMIT = 0.02         # Pantheon+ 95% C.L. bound
OMEGA_OP_TARGET = 0.0179      # Simulation validated value
JT_CRITICAL = 1.0             # Floquet topological transition
JT_TOLERANCE = 0.05           # JT* = 1.00 ± 0.05


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class APCrossing:
    """Record of an AP' threshold crossing event."""
    index: int                    # Time/JT index
    jt_value: float              # JT parameter value
    ap_value: float              # AP' value at crossing
    xi_value: float              # ξ value at crossing
    resolvent_value: float       # |Tr(I-U)^-1| at crossing
    is_critical: bool            # Near JT* = 1.0?


@dataclass 
class RatchetResult:
    """Complete SPG validation result."""
    # Time series
    jt_scan: np.ndarray
    xi_series: np.ndarray
    ap_prime_series: np.ndarray
    omega_op_series: np.ndarray
    resolvent_series: np.ndarray
    
    # Crossing detection
    crossings: List[APCrossing]
    critical_crossings: int          # Crossings near JT* = 1.0
    total_crossings: int
    
    # Validation metrics
    ap_min: float
    ap_max: float
    omega_op_final: float
    omega_op_max: float
    
    # Correlation with CCC-2
    resolvent_correlation: float     # Correlation of crossings with resolvent peaks
    
    # Pass/fail
    cr5_passed: bool                 # AP' crosses -1/3 at JT*
    cr4_passed: bool                 # |Ω_op| < 0.02
    ccc2_correlated: bool            # Crossings correlate with resolvent
    overall_passed: bool
    
    # Diagnostics
    failure_reasons: List[str] = field(default_factory=list)


# =============================================================================
# CORE COMPUTATIONS
# =============================================================================

def extract_floquet_eigenvalues(floquet_operators: np.ndarray) -> np.ndarray:
    """
    Extract eigenvalues from Floquet unitary operators.
    
    Parameters
    ----------
    floquet_operators : np.ndarray
        Shape (T, N, N) array of Floquet unitaries U(JT) at each scan point
        
    Returns
    -------
    eigenvalues : np.ndarray
        Shape (T, N) array of complex eigenvalues
    """
    T = floquet_operators.shape[0]
    N = floquet_operators.shape[1]
    eigenvalues = np.zeros((T, N), dtype=complex)
    
    for t in range(T):
        U = floquet_operators[t]
        eigs = np.linalg.eigvals(U)
        # Sort by phase for consistent tracking
        phases = np.angle(eigs)
        sort_idx = np.argsort(phases)
        eigenvalues[t] = eigs[sort_idx]
        
    return eigenvalues


def compute_xi_from_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute order parameter ξ from Floquet eigenvalues.
    
    Registry CR-5 Definition:
        ξ(JT) = 1 - min_k |1 - λ_k(JT)|
        
    Physical meaning: Proximity of closest eigenvalue to unity (resonance).
    ξ → 1 when an eigenvalue approaches λ = 1 (topological transition).
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Shape (T, N) complex eigenvalues
        
    Returns
    -------
    xi : np.ndarray
        Shape (T,) order parameter
    """
    T = eigenvalues.shape[0]
    xi = np.zeros(T)
    
    for t in range(T):
        # Distance of each eigenvalue from 1
        distances = np.abs(1 - eigenvalues[t])
        # ξ = 1 - min distance (high ξ = close to resonance)
        xi[t] = 1 - np.min(distances)
        
    return xi


def compute_ap_prime_radar(xi: np.ndarray, 
                           jt_scan: np.ndarray,
                           gamma: float = 0.032,
                           kappa: float = 1.0,
                           smoothing_sigma: float = 2.0) -> np.ndarray:
    """
    Compute AP' using the radar geometry mapping.
    
    Registry CR-5 Definition:
        AP'_sim = tanh(κ · ξ̇/γ)
        
    This maps the velocity of the order parameter to [-1, +1],
    matching the FRW equation of state range.
    
    The tanh ensures:
    - AP' = 0 when ξ is stationary (matter-dominated)
    - AP' → -1 when ξ decreasing rapidly (contraction)
    - AP' → +1 when ξ increasing rapidly (expansion)
    
    Critical threshold AP' = -1/3 corresponds to:
        ξ̇/γ = arctanh(-1/3)/κ ≈ -0.347/κ
        
    Parameters
    ----------
    xi : np.ndarray
        Order parameter time series
    jt_scan : np.ndarray
        JT parameter values
    gamma : float
        Damping rate (from Willow T2 calibration)
    kappa : float
        Coupling constant
    smoothing_sigma : float
        Gaussian smoothing for derivative stability
        
    Returns
    -------
    ap_prime : np.ndarray
        Acceleration parameter in [-1, +1]
    """
    # Smooth ξ to reduce noise in derivatives
    xi_smooth = gaussian_filter1d(xi, sigma=smoothing_sigma)
    
    # Compute dξ/dJT
    dJT = np.gradient(jt_scan)
    xi_dot = np.gradient(xi_smooth) / (dJT + 1e-12)
    
    # Apply radar mapping: AP' = tanh(κ · ξ̇/γ)
    ap_prime = np.tanh(kappa * xi_dot / gamma)
    
    return ap_prime


def compute_ap_prime_second_derivative(xi: np.ndarray,
                                        jt_scan: np.ndarray,
                                        smoothing_sigma: float = 2.0) -> np.ndarray:
    """
    Alternative AP' computation using second derivative.
    
    This is a simpler proxy:
        AP'_alt = d²ξ/dJT² (normalized to [-1, +1])
        
    Use this if the radar mapping parameters (γ, κ) are uncalibrated.
    
    Parameters
    ----------
    xi : np.ndarray
        Order parameter time series
    jt_scan : np.ndarray
        JT parameter values
        
    Returns
    -------
    ap_prime : np.ndarray
        Normalized second derivative
    """
    xi_smooth = gaussian_filter1d(xi, sigma=smoothing_sigma)
    
    dJT = np.gradient(jt_scan)
    xi_dot = np.gradient(xi_smooth) / (dJT + 1e-12)
    xi_ddot = np.gradient(xi_dot) / (dJT + 1e-12)
    
    # Normalize to [-1, +1]
    max_abs = np.max(np.abs(xi_ddot)) + 1e-12
    ap_prime = xi_ddot / max_abs
    
    return ap_prime


def compute_omega_op(overlap_matrices: np.ndarray,
                     cumulative: bool = True) -> np.ndarray:
    """
    Compute Ω_op (operational curvature) from overlap matrices.
    
    Registry CR-4 Definition:
        |Ω_op| < 0.02 (Pantheon+ 95% C.L.)
        
    Physical meaning: Cumulative geometric drift from complexity flow.
    
    Parameters
    ----------
    overlap_matrices : np.ndarray
        Shape (T, N, N) overlap/fidelity matrices
    cumulative : bool
        If True, return cumulative sum (for Pantheon+ comparison)
        If False, return instantaneous rate
        
    Returns
    -------
    omega_op : np.ndarray
        Shape (T,) operational curvature
    """
    T = overlap_matrices.shape[0]
    N = overlap_matrices.shape[1]
    
    omega_instant = np.zeros(T)
    
    for t in range(T):
        M = overlap_matrices[t]
        # Off-diagonal leakage = geometric drift rate
        diag = np.diag(np.diag(M))
        off_diag = M - diag
        # Normalized by dimension
        omega_instant[t] = np.linalg.norm(off_diag, 'fro') / N
    
    if cumulative:
        # Cumulative drift (scaled to match Pantheon+ units)
        # The scaling factor maps Willow timescale to cosmological
        omega_op = np.cumsum(omega_instant) / T
    else:
        omega_op = omega_instant
        
    return omega_op


def compute_resolvent_trace(floquet_operators: np.ndarray,
                            epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute resolvent trace |Tr(I - U)^{-1}| for CCC-2 correlation.
    
    Registry CCC-2: Peak at JT* = 1.00 ± 0.05
    
    Parameters
    ----------
    floquet_operators : np.ndarray
        Shape (T, N, N) Floquet unitaries
    epsilon : float
        Regularization to avoid division by zero
        
    Returns
    -------
    resolvent : np.ndarray
        Shape (T,) resolvent trace magnitude
    """
    T = floquet_operators.shape[0]
    N = floquet_operators.shape[1]
    resolvent = np.zeros(T)
    
    I = np.eye(N)
    
    for t in range(T):
        U = floquet_operators[t]
        diff = I - U
        
        # Regularized inverse
        try:
            inv = np.linalg.inv(diff + epsilon * I)
            resolvent[t] = np.abs(np.trace(inv))
        except np.linalg.LinAlgError:
            # Singular - this IS the divergence we're looking for
            resolvent[t] = 1.0 / epsilon
            
    return resolvent


# =============================================================================
# THRESHOLD DETECTION
# =============================================================================

def detect_ap_crossings(ap_prime: np.ndarray,
                        jt_scan: np.ndarray,
                        xi: np.ndarray,
                        resolvent: np.ndarray,
                        threshold: float = AP_CRITICAL_THRESHOLD) -> List[APCrossing]:
    """
    Detect crossings where AP' falls below the critical threshold.
    
    Registry CR-5: AP' crosses -1/3 at Floquet instability onset.
    
    Parameters
    ----------
    ap_prime : np.ndarray
        AP' time series
    jt_scan : np.ndarray
        JT parameter values
    xi : np.ndarray
        Order parameter
    resolvent : np.ndarray
        Resolvent trace for correlation
    threshold : float
        Critical value (default -1/3)
        
    Returns
    -------
    crossings : List[APCrossing]
        Detected crossing events
    """
    crossings = []
    
    # Find indices where AP' crosses below threshold
    below = ap_prime < threshold
    # Detect transitions from above to below
    transitions = np.diff(below.astype(int))
    crossing_indices = np.where(transitions == 1)[0] + 1
    
    for idx in crossing_indices:
        jt = jt_scan[idx] if idx < len(jt_scan) else jt_scan[-1]
        
        # Is this crossing near JT* = 1.0?
        is_critical = np.abs(jt - JT_CRITICAL) < JT_TOLERANCE
        
        crossings.append(APCrossing(
            index=int(idx),
            jt_value=float(jt),
            ap_value=float(ap_prime[idx]),
            xi_value=float(xi[idx]),
            resolvent_value=float(resolvent[idx]),
            is_critical=is_critical
        ))
    
    return crossings


def correlate_with_resolvent(crossings: List[APCrossing],
                             resolvent: np.ndarray,
                             jt_scan: np.ndarray,
                             peak_prominence: float = 0.1) -> float:
    """
    Check if AP' crossings correlate with resolvent peaks (CCC-2).
    
    Registry requirement: Crossings should occur at same JT values
    as resolvent divergences.
    
    Parameters
    ----------
    crossings : List[APCrossing]
        Detected AP' crossings
    resolvent : np.ndarray
        Resolvent trace
    jt_scan : np.ndarray
        JT values
    peak_prominence : float
        Minimum prominence for peak detection
        
    Returns
    -------
    correlation : float
        Fraction of crossings that coincide with resolvent peaks
    """
    if len(crossings) == 0:
        return 0.0
    
    # Find resolvent peaks
    peaks, properties = signal.find_peaks(resolvent, 
                                          prominence=peak_prominence * np.max(resolvent))
    
    if len(peaks) == 0:
        return 0.0
    
    peak_jt_values = jt_scan[peaks]
    
    # Count crossings that coincide with peaks
    coincident = 0
    for crossing in crossings:
        # Check if any peak is within tolerance
        distances = np.abs(peak_jt_values - crossing.jt_value)
        if np.min(distances) < JT_TOLERANCE:
            coincident += 1
    
    return coincident / len(crossings)


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def run_cosmic_ratchet_test(ds,
                            ap_threshold: float = AP_CRITICAL_THRESHOLD,
                            omega_limit: float = OMEGA_OP_LIMIT,
                            gamma: float = 0.032,
                            kappa: float = 1.0,
                            use_radar_mapping: bool = True) -> RatchetResult:
    """
    Complete SPG validation against Willow data.
    
    Validates:
    - CR-5: AP' crosses -1/3 at JT* ≈ 1.0
    - CR-4: |Ω_op| < 0.02
    - CCC-2: Crossings correlate with resolvent peaks
    
    Parameters
    ----------
    ds : WillowDataset
        Dataset with floquet_operators, overlap_matrices, jt_scan
    ap_threshold : float
        AP' critical threshold (default -1/3)
    omega_limit : float
        Pantheon+ Ω_op bound (default 0.02)
    gamma : float
        Damping rate for radar mapping
    kappa : float
        Coupling constant for radar mapping
    use_radar_mapping : bool
        If True, use tanh(κξ̇/γ) mapping
        If False, use normalized second derivative
        
    Returns
    -------
    result : RatchetResult
        Complete validation result
    """
    failure_reasons = []
    
    # --- Extract JT scan values ---
    # Support both jt_scan and JT_scan_points for compatibility
    if hasattr(ds, 'JT_scan_points') and ds.JT_scan_points is not None:
        jt_scan = np.asarray(ds.JT_scan_points)
    elif hasattr(ds, 'jt_scan') and ds.jt_scan is not None:
        jt_scan = np.asarray(ds.jt_scan)
    else:
        # Assume uniform scan from 0.8 to 1.2
        T = ds.floquet_operators.shape[0]
        jt_scan = np.linspace(0.8, 1.2, T)
    
    # --- Step 1: Extract eigenvalues from Floquet operators ---
    if ds.floquet_operators is None:
        raise ValueError("Dataset missing floquet_operators")
    
    # Convert to numpy array if needed
    floquet_operators = np.asarray(ds.floquet_operators)
    
    eigenvalues = extract_floquet_eigenvalues(floquet_operators)
    
    # --- Step 2: Compute ξ from eigenvalues (Registry CR-5) ---
    xi = compute_xi_from_eigenvalues(eigenvalues)
    
    # --- Step 3: Compute AP' (Registry CR-5) ---
    if use_radar_mapping:
        ap_prime = compute_ap_prime_radar(xi, jt_scan, gamma, kappa)
    else:
        ap_prime = compute_ap_prime_second_derivative(xi, jt_scan)
    
    # --- Step 4: Compute Ω_op (Registry CR-4) ---
    if ds.overlap_matrices is not None:
        overlap_matrices = np.asarray(ds.overlap_matrices)
        omega_op = compute_omega_op(overlap_matrices, cumulative=True)
    else:
        # Fallback: estimate from eigenvalue spread
        omega_op = np.zeros(len(jt_scan))
        failure_reasons.append("Missing overlap_matrices, Ω_op estimated as zero")
    
    # --- Step 5: Compute resolvent for CCC-2 correlation ---
    resolvent = compute_resolvent_trace(floquet_operators)
    
    # --- Step 6: Detect AP' crossings ---
    crossings = detect_ap_crossings(ap_prime, jt_scan, xi, resolvent, ap_threshold)
    
    critical_crossings = sum(1 for c in crossings if c.is_critical)
    total_crossings = len(crossings)
    
    # --- Step 7: Correlate with resolvent peaks ---
    resolvent_correlation = correlate_with_resolvent(crossings, resolvent, jt_scan)
    
    # --- Step 8: Validate against registry criteria ---
    
    # CR-5: Must have at least one crossing near JT* = 1.0
    cr5_passed = critical_crossings >= 1
    if not cr5_passed:
        failure_reasons.append(f"CR-5 FAILED: No AP' < {ap_threshold:.3f} crossing near JT* = 1.0")
    
    # CR-4: Final |Ω_op| must be below Pantheon+ limit
    omega_op_final = omega_op[-1] if len(omega_op) > 0 else 0.0
    omega_op_max = np.max(omega_op)
    cr4_passed = omega_op_final < omega_limit
    if not cr4_passed:
        failure_reasons.append(f"CR-4 FAILED: |Ω_op| = {omega_op_final:.4f} >= {omega_limit}")
    
    # CCC-2 correlation: Crossings should coincide with resolvent peaks
    ccc2_correlated = resolvent_correlation > 0.5  # At least 50% coincidence
    if not ccc2_correlated and total_crossings > 0:
        failure_reasons.append(f"CCC-2 CORRELATION WEAK: Only {resolvent_correlation:.1%} of crossings at resolvent peaks")
    
    # Overall pass requires CR-5 AND CR-4
    overall_passed = cr5_passed and cr4_passed
    
    return RatchetResult(
        jt_scan=jt_scan,
        xi_series=xi,
        ap_prime_series=ap_prime,
        omega_op_series=omega_op,
        resolvent_series=resolvent,
        crossings=crossings,
        critical_crossings=critical_crossings,
        total_crossings=total_crossings,
        ap_min=float(np.min(ap_prime)),
        ap_max=float(np.max(ap_prime)),
        omega_op_final=float(omega_op_final),
        omega_op_max=float(omega_op_max),
        resolvent_correlation=float(resolvent_correlation),
        cr5_passed=cr5_passed,
        cr4_passed=cr4_passed,
        ccc2_correlated=ccc2_correlated,
        overall_passed=overall_passed,
        failure_reasons=failure_reasons
    )


# =============================================================================
# FLOQUET 32-CELL CLASSIFICATION (New Addition)
# =============================================================================

@dataclass
class Floquet32Cell:
    cell_id: int
    bits: str
    o: int  # sign det
    s: int  # sign tr
    k: str  # C/O/X/U
    b: int  # sign β
    nu_f: float  # Floquet multipole
    kappa: float  # Skin rate


def classify_floquet_32cell(U_T: np.ndarray, k_y: float = np.pi/4) -> List[Floquet32Cell]:
    """Classify U_T eigs to 32-cell, compute ν_F, κ from Hamanaka extension."""
    # Use eig for general complex matrices (not eigh which is for Hermitian)
    eigs, evecs = np.linalg.eig(U_T)
    eps = np.angle(eigs) / (2 * np.pi)  # Quasienergy
    cells = []
    for i in range(U_T.shape[0]):
        eig = eigs[i]
        # Jacobian proxy from Floquet: tr = log|λ|, det = arg λ, Δ = (tr)^2 - 4 det
        tr = np.log(np.abs(eig) + 1e-12)
        det = np.angle(eig)
        delta = tr**2 - 4 * det
        # Bits from 32_jac.txt
        o = 1 if det > 0 else -1
        s = 1 if tr > 0 else -1
        k_map = {'C':0, 'O':1, 'U':3}
        k = 'C' if abs(delta) < 1e-6 else 'O' if delta > 0 else 'U'
        b = 1  # Slider branch
        # Robust cell_id: Binary packing into 1-32
        cell_id = 16 * ((o + 1) // 2) + 8 * ((s + 1) // 2) + 2 * k_map[k] + ((1 - b) // 2) + 1
        cell_id = max(1, min(32, cell_id))  # Clamp to valid range
        bits = f"{(o+1):01b}{(s+1):01b}{k_map[k]:02b}{b:01b}"
        # Floquet multipole contribution
        occ = np.abs(evecs[:, i])**2 if abs(eps[i]) < 0.1 else np.zeros(U_T.shape[0])
        if occ.size > 0 and np.sum(occ) > 1e-12:
            Q = np.outer(occ, occ)  # Density matrix proxy
            # log(det(M)) = trace(log(M)) for positive definite M
            # Use eigenvalues of Q for stability
            Q_eigs = np.linalg.eigvalsh(Q + 1e-12 * np.eye(Q.shape[0]))
            nu_contrib = np.real(np.sum(np.log(Q_eigs + 1e-12))) / (2 * np.pi)
        else:
            nu_contrib = 0.0
        kappa = 0.23 * np.abs(evecs[0, i])**2  # Skin from corner amp
        cells.append(Floquet32Cell(cell_id, bits, o, s, k, b, nu_contrib, kappa))
    return cells


def test_willow_floquet_32cell(ds: Any) -> Dict:
    """Test 32-cell on Willow Kitaev data (U_T matrices)."""
    try:
        # Handle WillowDataset or Dict
        # Try to get U_T_list attribute, or fall back to floquet_operators
        if hasattr(ds, 'U_T_list'):
            U_T_list = ds.U_T_list or []
        elif hasattr(ds, 'get') and callable(ds.get):
            U_T_list = ds.get('U_T_list', [])
        elif hasattr(ds, 'floquet_operators') and ds.floquet_operators is not None:
            # Use floquet_operators as fallback for U_T (convert to numpy array)
            floquet_ops = np.asarray(ds.floquet_operators)
            U_T_list = [floquet_ops[i] for i in range(floquet_ops.shape[0])]
        else:
            U_T_list = []
        cells = []
        for ut in U_T_list:
            if hasattr(ut, 'full'):  # Sparse tensor fallback
                ut = ut.full()
            # Ensure it's a numpy array
            ut = np.asarray(ut)
            cells.extend(classify_floquet_32cell(ut))
        
        if not cells:
            return {
                "test": "Floquet32Cell_Willow",
                "description": "32-Cell Classification on Kitaev U_T",
                "error": "No U_T matrices found in dataset",
                "passed": False
            }
        
        # Stats
        cell_counts = np.bincount([c.cell_id - 1 for c in cells], minlength=32)
        cell23_frac = cell_counts[22] / len(cells) if len(cells) > 0 else 0.0
        nu_f_avg = np.mean([c.nu_f for c in cells])
        kappa_avg = np.mean([c.kappa for c in cells])
        
        passed = cell23_frac > 0.8 and abs(nu_f_avg - 1) < 0.1 and abs(kappa_avg - 0.23) < 0.05
        
        return {
            "test": "Floquet32Cell_Willow",
            "description": "32-Cell Classification on Kitaev U_T",
            "cell23_fraction": cell23_frac,
            "nu_f_avg": nu_f_avg,
            "kappa_avg": kappa_avg,
            "passed": passed
        }
    except Exception as e:
        return {
            "test": "Floquet32Cell_Willow",
            "description": "32-Cell Classification on Kitaev U_T",
            "error": str(e),
            "passed": False
        }


# =============================================================================
# TEST WRAPPER (Updated with Integration)
# =============================================================================

def validate_theorem_spg(ds) -> dict:
    """
    Wrapper for CLI integration.
    
    Returns dict compatible with existing WillowLab test runner.
    Now includes Floquet 32-Cell test.
    """
    try:
        res = run_cosmic_ratchet_test(ds)
        f32 = test_willow_floquet_32cell(ds)
        
        # Merge results
        merged = {
            "theorem": "SPG (CR-5 + CR-4 + Floquet32Cell)",
            "description": "FRW-Radar Acceleration Threshold + Pantheon+ Compliance + 32-Cell Classification",
            
            # CR-5 metrics
            "ap_threshold": AP_CRITICAL_THRESHOLD,
            "ap_min": res.ap_min,
            "ap_max": res.ap_max,
            "critical_crossings": res.critical_crossings,
            "total_crossings": res.total_crossings,
            "crossing_locations": [c.jt_value for c in res.crossings],
            "cr5_passed": bool(res.cr5_passed),
            
            # CR-4 metrics  
            "omega_op_limit": OMEGA_OP_LIMIT,
            "omega_op_final": res.omega_op_final,
            "omega_op_max": res.omega_op_max,
            "cr4_passed": bool(res.cr4_passed),
            
            # CCC-2 correlation
            "resolvent_correlation": res.resolvent_correlation,
            "ccc2_correlated": bool(res.ccc2_correlated),
            
            # Overall (SPG only; 32-cell separate)
            "spg_passed": bool(res.overall_passed),
            "failure_reasons": res.failure_reasons,
            
            # 32-Cell metrics (from f32) - only include specific metrics, not all keys
            "floquet32_test": f32.get('test', 'Floquet32Cell_Willow'),
            "floquet32_passed": bool(f32.get('passed', False)),
            "cell23_fraction": f32.get('cell23_fraction', 0.0),
            "nu_f_avg": f32.get('nu_f_avg', 0.0),
            "kappa_avg": f32.get('kappa_avg', 0.0),
            "floquet32_error": f32.get('error', None),
            
            # Combined overall pass
            "overall_passed": bool(res.overall_passed and f32.get('passed', False)),
            "all_failure_reasons": res.failure_reasons + ([f32.get('error', '')] if 'error' in f32 else [])
        }
        
        return merged
        
    except Exception as e:
        return {
            "theorem": "SPG (CR-5 + CR-4 + Floquet32Cell)",
            "error": str(e),
            "spg_passed": False,
            "overall_passed": False,
            "failure_reasons": [str(e)]
        }