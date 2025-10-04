# willowlab/ingest/normalize.py
import os, json, numpy as np
from dataclasses import asdict
from ..schema import WillowDataset
from .sniffers import content_hash

REQUIRED = ("JT_scan_points","floquet_eigenvalues")

def to_willow_dataset(mapping, meta=None):
    meta = dict(meta or {})
    JT = np.asarray(mapping.get("JT_scan_points")) if "JT_scan_points" in mapping else None
    lam = np.asarray(mapping.get("floquet_eigenvalues")) if "floquet_eigenvalues" in mapping else None
    V = mapping.get("floquet_eigenvectors")
    U = mapping.get("floquet_operators")
    S = mapping.get("entropy")
    E = mapping.get("effective_energy")
    eta = mapping.get("eta_oscillations")
    ch2 = mapping.get("chern_mod2")
    xings = mapping.get("spectral_flow_crossings")
    Ovl = mapping.get("overlap_matrices")

    ds = WillowDataset(
        JT_scan_points=JT if JT is not None else np.arange(lam.shape[0]) if lam is not None else np.arange(1),
        floquet_eigenvalues=lam,
        floquet_eigenvectors=V,
        floquet_operators=U,
        resolvent_trace=None,
        entropy=S,
        effective_energy=E,
        eta_oscillations=eta,
        chern_mod2=ch2,
        spectral_flow_crossings=xings,
        overlap_matrices=Ovl,
        meta=meta
    )
    ds.check_basic()
    return ds

def persist_dataset(ds: WillowDataset, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    mapping = {k: v for k,v in ds.__dict__.items() if k != "meta"}
    h = content_hash(mapping)
    base = os.path.join(out_dir, f"willow_{h}")
    np.savez_compressed(
        base + ".npz",
        JT_scan_points=ds.JT_scan_points,
        floquet_eigenvalues=ds.floquet_eigenvalues if ds.floquet_eigenvalues is not None else np.array([]),
        floquet_eigenvectors=ds.floquet_eigenvectors if ds.floquet_eigenvectors is not None else np.array([]),
        floquet_operators=ds.floquet_operators if ds.floquet_operators is not None else np.array([]),
        entropy=ds.entropy if ds.entropy is not None else np.array([]),
        effective_energy=ds.effective_energy if ds.effective_energy is not None else np.array([]),
        eta_oscillations=ds.eta_oscillations if ds.eta_oscillations is not None else np.array([]),
        chern_mod2=ds.chern_mod2 if ds.chern_mod2 is not None else np.array([]),
        spectral_flow_crossings=ds.spectral_flow_crossings if ds.spectral_flow_crossings is not None else np.array([]),
        overlap_matrices=ds.overlap_matrices if ds.overlap_matrices is not None else np.array([]),
    )
    with open(base + ".meta.json", "w") as f:
        json.dump(ds.meta, f, indent=2, default=str)
    return base + ".npz", base + ".meta.json"
