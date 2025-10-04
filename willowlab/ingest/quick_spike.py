# willowlab/ingest/quick_spike.py
import numpy as np

def quick_smoking_gun(willow_data, jt_star=1.0, window=0.05, hard_thresh=1e6, dyn_factor=100.0):
    lam = np.asarray(willow_data['floquet_eigenvalues'])
    JT = np.asarray(willow_data['JT_scan_points'])
    T, N = lam.shape
    phi = np.angle(np.prod(lam, axis=1)) / N
    lam_aligned = lam * np.exp(-1j* phi[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        trace_res = np.sum(1.0 / (1.0 - lam_aligned), axis=1)

    i_star = int(np.argmin(np.abs(JT - jt_star)))
    local = (JT >= jt_star - window) & (JT <= jt_star + window)
    local_else = local & (np.arange(T) != i_star)
    baseline = np.median(np.abs(trace_res[local_else])) if np.any(local_else) else np.median(np.abs(trace_res))
    peak = np.abs(trace_res[i_star])
    min_d1 = np.min(np.abs(1.0 - lam_aligned[i_star]))
    explodes = (peak > hard_thresh) or (peak > dyn_factor * max(baseline, 1.0))
    return {"explodes": bool(explodes), "JT_star": float(JT[i_star]), "peak": float(peak),
            "baseline": float(baseline), "ratio": float(peak/max(baseline, 1e-12)),
            "min_dist_to_one": float(min_d1), "trace_series": trace_res}
