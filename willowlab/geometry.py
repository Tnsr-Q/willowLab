import numpy as np

def non_abelian_wilson_loop(evecs, path_idx):
    overlaps = []
    for i in range(len(path_idx)-1):
        V1 = evecs[path_idx[i]]
        V2 = evecs[path_idx[i+1]]
        M = V1.conj().T @ V2
        U,_,Vh = np.linalg.svd(M)
        overlaps.append(U @ Vh)
    W = np.eye(overlaps[0].shape[0])
    for U in overlaps: W = W @ U
    return {
        "abelian_phase": np.angle(np.linalg.det(W)),
        "non_abelian_spectrum": np.linalg.eigvals(W),
        "curvature": np.log(W)
    }

def residue_atlas(floquet_ops, eps=1e-8):
    poles = []; residues=[]
    for E in floquet_ops:
        I_minus_E = np.eye(E.shape[0]) - E
        evals = np.linalg.eigvals(I_minus_E)
        idx = np.where(np.abs(evals) < eps)[0]
        adj = np.linalg.det(I_minus_E) * np.linalg.inv(I_minus_E)
        residues.append(np.trace(adj))
        poles.append(idx)
    return {"poles": poles, "residues": residues}

def black_hole_potential_from_residues(poles, grid_points):
    phi = np.zeros(len(grid_points))
    for pole in poles:
        for i,pt in enumerate(grid_points):
            d = np.linalg.norm(pt - pole["position"])
            phi[i] += pole["residue_magnitude"] / (d**2 + 1e-12)
    return phi
