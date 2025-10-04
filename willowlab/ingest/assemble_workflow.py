# willowlab/ingest/assemble_workflow.py
import os, json, pathlib, glob
from typing import List, Optional
from .sniffers import triage_all_zips
from .file_index import index_willow_dir
from .normalize import to_willow_dataset, persist_dataset
from .merge_policy import reconcile_eigenpairs
from .cache_dir import DirCache
from ..io import load_willow
from ..trinity import WillowTrinityStep1
from ..tests.t_spec_ent import test_duality

PRIORITY_HINTS = ("floquet_braiding", "braiding_2_plaquettes")

def _pick_best(candidates):
    for src, name, w in candidates:
        if ('floquet_eigenvalues' in w) and ('JT_scan_points' in w):
            return (src, name, w)
    for src, name, w in candidates:
        if 'floquet_eigenvalues' in w: return (src, name, w)
    for src, name, w in candidates:
        if 'JT_scan_points' in w: return (src, name, w)
    return (None, None, None)

def assemble_from_zips(zip_paths: List[str], out_dir: str, expanded_dirs: Optional[List[str]] = None,
                         merge_policy: str = "auto", cache_root: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    cache_root = cache_root or os.path.commonpath(zip_paths + (expanded_dirs or [])) if zip_paths else (expanded_dirs[0] if expanded_dirs else out_dir)
    dcache = DirCache(cache_root)

    # Stage A: zip triage (skip cached zips that haven't changed)
    zip_paths = [p for p in zip_paths if dcache.should_process(p)]
    report, assembled = triage_all_zips(zip_paths) if zip_paths else ([], [])
    # bump priority
    assembled.sort(key=lambda x: (any(h in x[0].lower() for h in PRIORITY_HINTS) is False))
    src, inner, mapping = _pick_best(assembled)

    # Stage B: deep dive expanded dirs (skip cached files internally)
    if (mapping is None or 'floquet_eigenvalues' not in mapping or 'JT_scan_points' not in mapping) and expanded_dirs:
        merged = dict(mapping or {})
        for d in expanded_dirs:
            # skip whole dir if none of its files changed
            inv = index_willow_dir(d)
            for path, ftype, kind, summary, w in inv:
                if os.path.isfile(path) and not dcache.should_process(path):
                    continue
                if not w: continue
                for k in ('JT_scan_points', 'floquet_eigenvalues', 'floquet_eigenvectors', 'floquet_operators'):
                    if k in w and k not in merged:
                        merged[k] = w[k]
        if merged:
            mapping = merged

    if mapping is None:
        raise RuntimeError("No usable Willow mapping found in zips or expanded dirs.")

    # Merge-policy resolution
    mapping = reconcile_eigenpairs(mapping, policy=merge_policy)
    meta = {"source_zip": src, "source_entry": inner, "note": f"assembled by WillowLab (merge_policy={merge_policy})"}
    ds = to_willow_dataset(mapping, meta=meta)
    npz_path, meta_path = persist_dataset(ds, out_dir)

    # Update cache records for processed inputs
    for zp in zip_paths:
        dcache.update(zp, bundle=npz_path)
    if expanded_dirs:
        for d in expanded_dirs:
            for leaf in glob.glob(os.path.join(d, "*")):
                if os.path.isfile(leaf):
                    dcache.update(leaf, bundle=npz_path)
    dcache.persist()

    # Arm the test suite
    loaded = load_willow(npz_path)
    tri = WillowTrinityStep1(loaded)
    inv = tri.compute_all(jt_star=1.0, window=0.05)
    summary = {"bundle": npz_path, "meta": meta_path, "invariants": inv}
    if (loaded.resolvent_trace is not None) and (loaded.entropy is not None) and (loaded.effective_energy is not None):
        summary["t_spec_ent"] = test_duality(loaded)

    out_json = pathlib.Path(out_dir, "assembly_summary.json")
    out_json.write_text(json.dumps(summary, indent=2, default=lambda x: str(x)))
    return str(out_json)
