# willowlab/ingest/sniffers.py
import os, io, json, gzip, zipfile, hashlib, pathlib
import numpy as np

# --- heuristic key sets (from your drafts)
LIKELY_EVAL_KEYS = ('eigenvalues', 'floquet_eigenvalues', 'evals', 'lambdas', 'lambda', 'eigvals', 'quasienergies', 'theta')
LIKELY_EVEC_KEYS = ('eigenvectors', 'floquet_eigenvectors', 'evecs', 'eigvecs')
LIKELY_U_KEYS = ('U_floquet', 'floquet_operator','floquet_operators', 'U', 'UF','U_Floquet')
LIKELY_JT_KEYS = ('JT_values','JT', 'jt', 'drive_strength','JT_scan_points', 'drive','t_values')

def _ensure_complex_evals(arr):
    arr = np.asarray(arr)
    return np.exp(1j*arr) if np.isrealobj(arr) else arr.astype(np.complex128, copy=False)

def _maybe_np_load(b):
    # npz or npy in bytes
    try:
        with np.load(io.BytesIO(b), allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}
    except Exception:
        try:
            return np.load(io.BytesIO(b), allow_pickle=False)
        except Exception:
            return None

def _sniff_inner_bytes(b):
    h = b[:16]
    if h.startswith(b'\x93NUMPY'): return 'npy'
    if h.startswith(b'PK\x03\x04'): return 'npz/zip'
    if h.startswith(b'\x89HDF\r\n\x1a\n'): return 'hdf5'
    if h.startswith(b'PAR1'): return 'parquet'
    if b'MATLAB' in b[:256]: return 'mat'
    try:
        t = b[:256].decode('utf-8', errors='ignore').strip()
        if t.startswith('{') or t.startswith('['): return 'json'
        if (',' in t) or ('\t' in t): return 'csv'
    except Exception:
        pass
    return 'unknown'

def _extract_willow_from_mapping(mapping):
    willow = {}
    def find_key(cands):
        for k in mapping.keys():
            kl = k.lower()
            for c in cands:
                if c.lower() in kl: return k
        return None

    kJT = find_key(LIKELY_JT_KEYS)
    if kJT is not None: willow['JT_scan_points'] = np.asarray(mapping[kJT]).ravel()

    ke = find_key(LIKELY_EVAL_KEYS)
    if ke is not None: willow['floquet_eigenvalues'] = _ensure_complex_evals(mapping[ke])

    kev = find_key(LIKELY_EVEC_KEYS)
    if kev is not None: willow['floquet_eigenvectors'] = np.asarray(mapping[kev])

    kU = find_key(LIKELY_U_KEYS)
    if kU is not None: willow['floquet_operators'] = np.asarray(mapping[kU])

    # derive evals/evecs if only U is present
    if 'floquet_eigenvalues' not in willow and 'floquet_operators' in willow:
        U = willow['floquet_operators']
        if U.ndim == 3:
            T, N, _ = U.shape
            evals = np.empty((T, N), dtype=np.complex128)
            evecs = np.empty_like(U)
            for t in range(T):
                w, V = np.linalg.eig(U[t])
                evals[t] = w; evecs[t] = V
            willow['floquet_eigenvalues'] = evals
            willow.setdefault('floquet_eigenvectors', evecs)
        elif U.ndim == 2:
            w, V = np.linalg.eig(U)
            willow['floquet_eigenvalues'] = w[None, :]
            willow['floquet_eigenvectors'] = V[None, :, :]
            willow.setdefault('JT_scan_points', np.array([np.nan]))
    return willow if willow else None

def scan_zip_for_willow(zip_path, max_bytes=2_000_000):
    """
    Stage A: shallow scan inside a .zip (or .npz) without full extraction.
    Returns (manifest, candidates)
    """
    cands, manifest = [], []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for zi in zf.infolist():
            if zi.is_dir(): continue
            name = zi.filename
            lower = name.lower()
            if not any(k in lower for k in ('floquet', 'eval','eig','jt','u_','operator', 'braid', 'czuniform', 'data')):
                continue
            if zi.file_size > max_bytes and not any(k in lower for k in ('floquet', 'eigen', 'u_floquet','operator','jt')):
                manifest.append((name, 'skipped (huge)')); continue

            b = zf.read(zi)
            ftype = _sniff_inner_bytes(b)
            manifest.append((name, ftype))
            if ftype in ('npy', 'npz/zip'):
                payload = _maybe_np_load(b)
                if isinstance(payload, dict):
                    w = _extract_willow_from_mapping(payload)
                    if w: cands.append((name, w))
                elif isinstance(payload, np.ndarray):
                    arr = payload
                    if 'jt' in lower: cands.append((name, {'JT_scan_points': arr}))
                    elif any(s in lower for s in ('eig', 'lambda', 'quasi')):
                        cands.append((name, {'floquet_eigenvalues': _ensure_complex_evals(arr)}))
            elif ftype == 'json':
                try:
                    d = json.loads(b.decode('utf-8', errors='ignore'))
                    if isinstance(d, dict):
                        w = _extract_willow_from_mapping(d)
                        if w: cands.append((name, w))
                except Exception:
                    pass
    return manifest, cands

def triage_all_zips(zip_paths):
    report, assembled = [], []
    for zp in zip_paths:
        manifest, cands = scan_zip_for_willow(zp)
        report.append((os.path.basename(zp), manifest))
        assembled.extend([(os.path.basename(zp), n, w) for (n, w) in cands])
    return report, assembled

def content_hash(willow_mapping):
    h = hashlib.sha256()
    for k in sorted(willow_mapping.keys()):
        v = np.asarray(willow_mapping[k])
        h.update(k.encode()); h.update(v.shape.__repr__().encode());
        h.update(v.dtype.__repr__().encode())
        h.update(np.ascontiguousarray(v).data)
    return h.hexdigest()[:16]
