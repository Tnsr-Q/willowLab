#willowlab/ingest/file_index.py
import os, io, json, gzip, numpy as np
from .sniffers import (LIKELY_EVAL_KEYS, LIKELY_EVEC_KEYS, LIKELY_U_KEYS, LIKELY_JT_KEYS,
                       _ensure_complex_evals, _extract_willow_from_mapping)
try:
    import h5py
except Exception:
    h5py = None
try:
    import scipy.io as sio
except Exception:
    sio = None

def sniff_file_type(path, read_bytes=16):
    with open(path, 'rb') as f:
        head = f.read(read_bytes)
    if head.startswith(b'\x1f\x8b'): return 'gzip'
    if head.startswith(b'PK\x03\x04'): return 'zip'
    if head.startswith(b'\x89HDF\r\n\x1a\n'): return 'hdf5'
    if head.startswith(b'PAR1'): return 'parquet'
    if head.startswith(b'ARROW1'): return 'feather'
    if head.startswith(b'\x93NUMPY'): return 'npy'
    if b'MATLAB' in head or b'MATLAB 5.0 MAT-file' in head: return 'mat'
    try:
        txt = head.decode('utf-8', errors='ignore').strip()
        if txt.startswith('{') or txt.startswith('['): return 'json'
        if (',' in txt) or ('\t' in txt): return 'csv'
    except Exception:
        pass
    return 'unknown'

def load_any(path):
    ftype = sniff_file_type(path)
    try:
        if ftype == 'npy':
            arr = np.load(path, allow_pickle=False); return 'array', arr
        if ftype == 'zip':
            # treat as npz if possible
            try:
                with np.load(path, allow_pickle=False) as npz:
                    return 'mapping', {k: npz[k] for k in npz.files}
            except Exception:
                return 'unknown', None
        if ftype == 'hdf5' and h5py is not None:
            f = h5py.File(path, 'r'); return 'hdf5', f
        if ftype == 'mat' and sio is not None:
            d = sio.loadmat(path, squeeze_me=True, struct_as_record=False); return 'mapping', d
        if ftype == 'json':
            with open(path, 'r', encoding='utf-8', errors='ignore') as f: return 'mapping', json.load(f)
        if ftype == 'csv':
            arr = np.genfromtxt(path, delimiter=None, dtype=float); return 'array', arr
        # fallback attempts
        try:
            with np.load(path, allow_pickle=False) as npz:
                return 'mapping', {k: npz[k] for k in npz.files}
        except Exception:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return 'text', f.read()
            except Exception:
                return 'unknown', None
    except Exception:
        return 'unknown', None

def assemble_willow_from_mapping(mapping):
    # identical logic as our zip path-kept here for local use
    from .sniffers import _extract_willow_from_mapping
    return _extract_willow_from_mapping(mapping)

def _sample_hdf5_obj(obj, budget_items: int = 24, budget_elems: int = 200_000):
    """
    Build a small dict of datasets from HDF5 'obj' scanning root and one level deep.
    We cap items and elements to avoid RAM blowups.
    """
    out = {}
    def _take(name, dset):
        try:
            if hasattr(dset, "shape"):
                # respect budget
                numel = 1
                for s in getattr(dset, "shape", ()): numel *= max(int(s), 1)

                if numel > budget_elems:
                    # take a small head slice along first axis (if any)
                    sl = (slice(0, min(8, dset.shape[0])),) + tuple(slice(None) for _ in range(len(dset.shape)-1))
                    out[name] = dset[sl]
                else:
                    out[name] = dset[()]
                return True
        except Exception:
            return False
        return False

    # root datasets
    count = 0
    for k, v in obj.items():
        if count >= budget_items: break
        if hasattr(v, "shape"):
            if _take(k, v): count += 1

    # one level deep: groups under root
    for k, v in obj.items():
        if count >= budget_items: break
        if hasattr(v, "items"): # group
            for k2, v2 in v.items():
                if count >= budget_items: break
                full = f"{k}/{k2}"
                if hasattr(v2, "shape"):
                    if _take(full, v2): count += 1
    return out

def index_willow_dir(data_dir, name_filter=None):
    results = []
    for fn in os.listdir(data_dir):
        if fn.startswith('.'): continue
        if name_filter and name_filter not in fn: continue
        path = os.path.join(data_dir, fn)
        if os.path.isdir(path):
            results.append((path, 'dir', 'unknown', 'directory', None)); continue

        ftype = sniff_file_type(path)
        kind, obj = load_any(path)
        summary, willow = "", None
        if kind == 'array':
            summary = f'array shape={obj.shape} dtype={obj.dtype}'
        elif kind == 'mapping':
            keys = list(obj.keys())[:12]
            summary = f'mapping keys={keys}'
            willow = assemble_willow_from_mapping(obj)
        elif kind == 'hdf5':
            try:
                sample = _sample_hdf5_obj(obj)
                summary = f'hdf5 sampled keys={list(sample.keys())[:12]}'
                willow = assemble_willow_from_mapping(sample)
            finally:
                try: obj.close()
                except Exception: pass
        elif kind == 'text':
            summary = 'text'
        else:
            summary = 'unknown/unreadable'
        results.append((path, ftype, kind, summary, willow))
    return results
