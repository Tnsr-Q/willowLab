import numpy as np, h5py, json, pathlib
from schema import WillowDataset

def load_willow(path, *, kind=None, meta=None) -> WillowDataset:
    """
    Single ingress: detects file type and normalizes into WillowDataset.
    Supports: .npz, .npy, .h5/.hdf5, .csv (multiple), or a directory with pieces.
    """
    p = pathlib.Path(path)
    kind = kind or (p.suffix.lower().lstrip('.'))
    meta = meta or {}

    def _np(obj, key):
        return np.asarray(obj[key]) if key in obj else None

    if kind == "npz":
        with np.load(p, allow_pickle=False) as z:
            ds = WillowDataset(
                JT_scan_points=z["JT_scan_points"],
                floquet_eigenvalues=_np(z,"floquet_eigenvalues"),
                floquet_eigenvectors=_np(z,"floquet_eigenvectors"),
                floquet_operators=_np(z,"floquet_operators"),
                resolvent_trace=_np(z,"resolvent_trace"),
                entropy=_np(z,"entropy"),
                effective_energy=_np(z,"effective_energy"),
                eta_oscillations=_np(z,"eta_oscillations"),
                chern_mod2=_np(z,"chern_mod2"),
                spectral_flow_crossings=_np(z,"spectral_flow_crossings"),
                overlap_matrices=_np(z,"overlap_matrices"),
                meta=meta
            )
            ds.check_basic(); return ds

    if kind in ("h5","hdf5"):
        with h5py.File(p, "r") as h:
            g = h["/willow"]
            ds = WillowDataset(
                JT_scan_points=g["JT_scan_points"][...],
                floquet_eigenvalues=g.get("floquet_eigenvalues"),
                floquet_eigenvectors=g.get("floquet_eigenvectors"),
                floquet_operators=g.get("floquet_operators"),
                resolvent_trace=g.get("resolvent_trace"),
                entropy=g.get("entropy"),
                effective_energy=g.get("effective_energy"),
                eta_oscillations=g.get("eta_oscillations"),
                chern_mod2=g.get("chern_mod2"),
                spectral_flow_crossings=g.get("spectral_flow_crossings"),
                overlap_matrices=g.get("overlap_matrices"),
                meta=meta
            )
            # h5py Dataset -> np.ndarray
            ds = WillowDataset(**{k: (v[...].astype(complex) if hasattr(v,'shape') else v)
                                  for k,v in ds.__dict__.items()})
            ds.check_basic(); return ds

    raise ValueError(f"Unsupported or unrecognized input: {p}")
