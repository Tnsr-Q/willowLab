"""Helpers for the spectral/entanglement duality checks without NumPy."""

from __future__ import annotations

from math import fabs, log
from typing import Dict, Sequence

from willowlab import _numpy_shim as nps

_EPS = 1e-12


def _to_list(values: Sequence) -> list:
    if hasattr(values, "to_list"):
        return list(values.to_list())  # type: ignore[attr-defined]
    return list(values)


def spectral_temperature(resolvent_trace: Sequence[float], jt: Sequence[float]):
    log_tr = [log(value + _EPS) for value in resolvent_trace]
    d1 = nps.gradient(log_tr, jt)
    d1_list = d1.to_list() if hasattr(d1, "to_list") else list(d1)
    d2 = nps.gradient(d1_list, jt)
    d2_list = d2.to_list() if hasattr(d2, "to_list") else list(d2)
    return [1.0 / (fabs(value) + _EPS) for value in d2_list]


def entanglement_temperature(entropy: Sequence[float], energy: Sequence[float]):
    dS_dE = nps.gradient(entropy, energy)
    gradient_list = dS_dE.to_list() if hasattr(dS_dE, "to_list") else list(dS_dE)
    return [1.0 / (fabs(value) + _EPS) for value in gradient_list]


def test_duality(ds) -> Dict[str, float]:
    t_spec = spectral_temperature(ds.resolvent_trace, ds.JT_scan_points)
    t_ent = entanglement_temperature(ds.entropy, ds.effective_energy)
    mask = [0.98 < value < 1.02 for value in ds.JT_scan_points]
    a = [log(t_spec[idx] + _EPS) for idx, flag in enumerate(mask) if flag]
    b = [log(t_ent[idx] + _EPS) for idx, flag in enumerate(mask) if flag]
    slope = nps.polyfit(a, b, 1)[0]
    slope_val = float(slope)
    corr = float(nps.corrcoef(a, b)[0][1])
    metrics = {
        "slope": slope_val,
        "r2": corr * corr,
        "duality_holds": bool(abs(slope_val - 1.0) < 0.1 and corr * corr > 0.9),
    }
    assert metrics["duality_holds"]
