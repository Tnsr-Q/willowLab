import numpy as np
from typing import Tuple, Dict, Any, Optional

_EPS = 1e-12

def _adj_trace(I_minus_U: np.ndarray) -> complex:
    """Trace of adjugate(I-U) = det(I-U) * trace((I-U)^-1)."""
    det = np.linalg.det(I_minus_U)
    X = np.linalg.pinv(I_minus_U) # stable even near singularity
    return det * np.trace(X)

def poles_and_residues_on_grid(Ugrid: np.ndarray, tol: float = 1e-8) -> Dict[str, Any]:
    """
    Inputs:
        Ugrid: [Ny, Nx, N, N] complex Floquet operators on a rectangular grid.
    Returns:
        dict with per-gridpoint:
        - near_zero_eigs: boolean mask if min eig of (I-U) is < tol
        - residue_score: scalar trace(adjugate) magnitude
        - det_IminusU: determinant values
    """
    Ny, Nx, N, _ = Ugrid.shape
    near_zero = np.zeros((Ny, Nx), dtype=bool)
    residue = np.zeros((Ny, Nx), dtype=float)
    dets = np.zeros((Ny, Nx), dtype=np.complex128)
    I = np.eye(N, dtype=np.complex128)
    for iy in range(Ny):
        for ix in range(Nx):
            M = I - Ugrid[iy, ix]
            ev = np.linalg.eigvals(M)
            near_zero[iy, ix] = np.min(np.abs(ev)) < tol
            dets[iy, ix] = np.linalg.det(M)
            residue[iy, ix] = np.abs(_adj_trace(M))
    return {"near_zero_eigs": near_zero, "residue_score": residue, "det_IminusU": dets}

def black_hole_potential(residue_map: np.ndarray, mask: Optional[np.ndarray] = None, p: float = 2.0) -> np.ndarray:
    """
    Compute at each gridpoint using discrete superposition of 'masses' at high-residue sites.
    Simplified discrete version of sum|R_alpha|/d^p. Complexity O(N^2); use FFT or truncation for big grids.
    """
    Ny, Nx = residue_map.shape
    Y, X = np.indices((Ny, Nx))
    pts = np.column_stack([Y.ravel(), X.ravel()])
    R = residue_map.ravel()
    if mask is None:
        mask = R > np.percentile(R, 95.0)
    src_idx = np.where(mask.ravel())[0]
    phi = np.zeros_like(R, dtype=float)
    for j in src_idx:
        dy = pts[:, 0] - pts[j, 0]
        dx = pts[:, 1] - pts[j, 1]
        d2 = (dx*dx + dy*dy).astype(float)
        phi += R[j] / (np.power(d2 + 1e-9, p/2.0))
    return phi.reshape(Ny, Nx)

def phase_wind_field(G: np.ndarray) -> np.ndarray:
    """W = grad(arg G) over the grid."""
    phase = np.angle(G)
    Wy, Wx = np.gradient(phase)
    return np.stack([Wy, Wx], axis=-1)

def plaquette_det_winding(det_grid: np.ndarray) -> np.ndarray:
    """
    Compute winding of arg det(I-U) around each 1x1 plaquette.
    Returns array with shape [Ny-1, Nx-1] containing winding numbers / (2π).
    """
    ang = np.unwrap(np.angle(det_grid), axis=0)
    ang = np.unwrap(ang, axis=1)
    Ny, Nx = det_grid.shape
    w = np.zeros((Ny-1, Nx-1), dtype=float)
    for iy in range(Ny-1):
        for ix in range(Nx-1):
            loop = [ang[iy, ix], ang[iy, ix+1], ang[iy+1, ix+1], ang[iy+1, ix], ang[iy, ix]]
            w[iy,ix] = (loop[-1] - loop[0]) / (2.0*np.pi)
    return w

def hessian_saddles(phi: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """
    Detect saddles via sign of Hessian eigenvalues.
    Returns boolean mask [Ny, Nx].
    """
    d_dy, d_dx = np.gradient(phi)
    d2_dy2, _ = np.gradient(d_dy)
    _, d2_dx2 = np.gradient(d_dx)
    d2_dydx = np.gradient(d_dy, axis=1) # d^2(phi)/dydx
    Ny, Nx = phi.shape
    saddles = np.zeros((Ny, Nx), dtype=bool)
    for iy in range(1, Ny-1):
        for ix in range(1, Nx-1):
            H = np.array([[d2_dy2[iy, ix], d2_dydx[iy, ix]],
                          [d2_dydx[iy, ix], d2_dx2[iy, ix]]], float)
            ev = np.linalg.eigvals(H)
            saddles[iy, ix] = (ev[0] * ev[1] < -thresh)
    return saddles

def cancellation_safe_resolvent_abs(evals_grid: np.ndarray) -> np.ndarray:
    """
    Reuse Step-1 surrogate per gridpoint: on |λ|≈1 use angle -> 1/(2|sin(θ/2)|), else |1/(1-λ)|.
    evals_grid: [Ny, Nx, N]
    """
    Ny, Nx, N = evals_grid.shape
    out = np.zeros((Ny, Nx), dtype=float)
    for iy in range(Ny):
        angles = np.angle(evals_grid[iy])
        on_circle = np.isclose(np.abs(evals_grid[iy]), 1.0, atol=1e-6)
        mag = np.empty((Nx, N), float)
        sin_half = np.maximum(np.abs(np.sin(angles/2.0)), 1e-18)
        mag[on_circle] = 1.0/(2.0* sin_half[on_circle])
        off = ~on_circle
        mag[off] = 1.0/ np.abs(1.0 - evals_grid[iy][off])
        out[iy] = np.sum(mag, axis=1)
    return out

def ep_candidates(det_wind: np.ndarray, kappa_grid: Optional[np.ndarray]=None,
                  wind_thresh: float = 0.25, kappa_thresh: float=1e8) -> np.ndarray:
    """
    Fuse det-winding and condition number flags into an EP candidate mask on the plaquette lattice.
    """
    mask = np.abs(det_wind) > wind_thresh
    if kappa_grid is not None:
        kappa_mask = kappa_grid[1:-1, 1:-1] > kappa_thresh
        mask = mask | kappa_mask
    return mask
