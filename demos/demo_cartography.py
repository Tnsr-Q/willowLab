import numpy as np
import matplotlib.pyplot as plt
from willowlab.cartography import (
    poles_and_residues_on_grid, black_hole_potential, plaquette_det_winding,
    hessian_saddles, cancellation_safe_resolvent_abs, phase_wind_field, ep_candidates
)

# 1) Load/construct your U-grid: Ugrid[Ny, Nx, N, N] (complex).
# Here it's assumed available as np.load(...) or produced by your simulator.
Ugrid = np.load("Ugrid_willow.npz")["Ugrid"] # user-provided

# 2) Residues & det(I-U)
atlas = poles_and_residues_on_grid(Ugrid, tol=1e-8) # adjugate-based residues and dets
residue = atlas["residue_score"]                   # |trace adj(I-U)|
detM = atlas["det_IminusU"]                        # det(I-U)

# 3) Gravitational potential Phi(lambda)
Phi = black_hole_potential(residue, p=2.0)

# 4) Resolvent wind field from cancellation-safe |Tr(I-U)^-1|
# If you already stored eigenvalues per gridpoint, pass them; otherwise compute from U.
Ny, Nx, N, _ = Ugrid.shape
evals = np.zeros((Ny, Nx, N), dtype=np.complex128)
for iy in range(Ny):
    for ix in range(Nx):
        evals[iy, ix] = np.linalg.eigvals(Ugrid[iy, ix])

Gabs = cancellation_safe_resolvent_abs(evals)
# Construct a complex G surrogate with Gabs as magnitude and det-phase as angle:
G = Gabs * np.exp(1j* np.angle(detM))
W = phase_wind_field(G) # [Ny, Nx, 2]

# 5) Plaquette det-winding-> EP candidates; optionally combine with k(V)
w_det = plaquette_det_winding(detM) # [Ny-1, Nx-1]
ep_mask = ep_candidates(w_det)      # boolean [Ny-1, Nx-1]

# 6) Saddles of Phi (mountain passes)
saddles = hessian_saddles(Phi, thresh=0.0)

#---- Plots (each in its own figure; default styles) ----
plt.figure()
plt.title("Residue score |tr adj(I-U)|")
plt.imshow(residue, origin="lower")
plt.colorbar(); plt.tight_layout(); plt.savefig("residue_score.png", dpi=180)

plt.figure()
plt.title("Gravitational potential Phi from residues")
plt.imshow(Phi, origin="lower")
plt.colorbar(); plt.tight_layout(); plt.savefig("phi_potential.png", dpi=180)

plt.figure()
plt.title("Determinant winding per plaquette (EP contours where |v|>0.25)")
plt.imshow(w_det, origin="lower")
plt.contour(ep_mask.astype(float), levels=[0.5]) # contour of EP mask
plt.colorbar(); plt.tight_layout(); plt.savefig("det_winding.png", dpi=180)

plt.figure()
plt.title("Saddles of Phi (mountain passes)")
plt.imshow(Phi, origin="lower")
# mark saddles
ys, xs = np.where(saddles)
plt.scatter(xs, ys, s=10)
plt.tight_layout(); plt.savefig("phi_saddles.png", dpi=180)

print("Wrote residue_score.png, phi_potential.png, det_winding.png, phi_saddles.png")
