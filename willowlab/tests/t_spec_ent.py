import numpy as np
from typing import Dict

def spectral_temperature(resolvent_trace, JT):
    log_tr = np.log(resolvent_trace + 1e-12)
    d2 = np.gradient(np.gradient(log_tr, JT), JT)
    return 1.0 / (np.abs(d2) + 1e-12)

def entanglement_temperature(S, E):
    dS_dE = np.gradient(S, E)
    return 1.0 / (np.abs(dS_dE) + 1e-12)

def test_duality(ds) -> Dict[str,float]:
    T_spec = spectral_temperature(ds.resolvent_trace, ds.JT_scan_points)
    T_ent = entanglement_temperature(ds.entropy, ds.effective_energy)
    mask = (ds.JT_scan_points > 0.98) & (ds.JT_scan_points < 1.02)
    a = np.log(T_spec[mask]); b = np.log(T_ent[mask])
    slope = np.polyfit(a, b, 1)[0]
    r2 = np.corrcoef(a, b)[0,1]**2
    metrics = {"slope": float(slope), "r2": float(r2),
               "duality_holds": bool(abs(slope-1.0) < 0.1 and r2 > 0.9)}
    assert metrics["duality_holds"]
