import numpy as np
from .geometry import non_abelian_wilson_loop

def nested_wilson_14(evecs_by_torus: list, plaquette_paths: list):
    W_stack = []
    for Vtorus, path in zip(evecs_by_torus, plaquette_paths):
        W = non_abelian_wilson_loop(Vtorus, path)["curvature"] # Lie algebra element
        W_stack.append(W)
    F = np.eye(W_stack[0].shape[0])
    for W in W_stack: F = np.kron(F, W)
    c14 = (1/(2*np.pi)**7) * np.trace(F)
    return {"c_14_integer": np.round(c14).astype(int), "c_14_raw": c14}
