"""Simple utilities for adding disorder and analysing its effect without NumPy."""

from __future__ import annotations

import random
from cmath import exp
from math import fabs, sqrt
from typing import Dict, Iterable, List, Sequence

from willowlab import _numpy_shim as nps
from willowlab.resolvent import r_op_from_trace, trace_resolvent_from_evals

_EPS = 1e-12


def _to_list(values: Sequence) -> List:
    if hasattr(values, "to_list"):
        return list(values.to_list())  # type: ignore[attr-defined]
    return list(values)


def _gradient(values: Sequence[float], coordinates: Sequence[float]) -> List[float]:
    grad = nps.gradient(values, coordinates)
    return grad.to_list() if hasattr(grad, "to_list") else list(grad)


def add_goe_disorder(matrix: Sequence[Sequence[float]], delta: float, seed: int | None = None) -> List[List[float]]:
    rng = random.Random(seed)
    base = [list(row) for row in matrix]
    n = len(base)
    disorder = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = rng.gauss(0.0, 1.0)
            disorder[i][j] = val
            disorder[j][i] = val
    frob_norm = sqrt(sum(cell * cell for row in base for cell in row))
    scale = delta * frob_norm
    return [[base[i][j] + scale * disorder[i][j] for j in range(n)] for i in range(n)]


def level_spacing_stats(eigenvalues: Sequence[complex]) -> Dict[str, float]:
    values = sorted(abs(val) for val in eigenvalues)
    spacings = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    spacings = [s for s in spacings if s > _EPS]
    if not spacings:
        return {"mean_spacing": float("nan"), "beta": float("nan")}
    mean_spacing = sum(spacings) / len(spacings)
    return {"mean_spacing": float(mean_spacing), "beta": 0.0}


def peak_sharpness(values: Sequence[float], coordinates: Sequence[float]) -> Dict[str, float]:
    arr = _to_list(values)
    coords = _to_list(coordinates)
    peak_idx = max(range(len(arr)), key=lambda idx: arr[idx])
    peak_val = arr[peak_idx]
    half_max = peak_val / 2.0
    above = [idx for idx, val in enumerate(arr) if val > half_max]
    if len(above) > 1:
        width = coords[above[-1]] - coords[above[0]]
    else:
        width = float("inf")
    second = _gradient(_gradient(arr, coords), coords)
    if 1 <= peak_idx < len(arr) - 1:
        window = slice(max(0, peak_idx - 3), min(len(arr), peak_idx + 4))
        sharpness = max(fabs(val) for val in second[window])
    else:
        sharpness = 0.0
    return {"peak_width": float(width), "peak_sharpness": float(sharpness)}


def disorder_scan(
    evals_clean: Sequence[Sequence[complex]],
    jt: Sequence[float],
    delta_values: Iterable[float],
    n_realizations: int = 5,
) -> List[Dict[str, float]]:
    eigenvalues = [list(map(complex, row)) for row in evals_clean]
    jt_list = _to_list(jt)
    results: List[Dict[str, float]] = []
    n = len(eigenvalues[0]) if eigenvalues else 1

    for delta in delta_values:
        r_op_ensemble: List[List[float]] = []
        level_stats: List[Dict[str, float]] = []
        for realization in range(n_realizations):
            rng = random.Random(realization + int(delta * 1000))
            evals_disordered = []
            for row in eigenvalues:
                noisy_row = []
                for value in row:
                    noise = delta * rng.gauss(0.0, 1.0)
                    noisy_row.append(value * exp(1j * noise))
                evals_disordered.append(noisy_row)
            trace_abs = trace_resolvent_from_evals(evals_disordered, safe=True)
            r_op = r_op_from_trace(trace_abs, n)
            r_op_ensemble.append(r_op)
            peak_idx = max(range(len(r_op)), key=lambda idx: r_op[idx])
            level_stats.append(level_spacing_stats(evals_disordered[peak_idx]))

        r_op_mean = [sum(values[i] for values in r_op_ensemble) / len(r_op_ensemble) for i in range(len(jt_list))]
        peak_info = peak_sharpness(r_op_mean, jt_list)
        beta_values = [entry["beta"] for entry in level_stats if entry["beta"] == entry["beta"]]
        mean_beta = sum(beta_values) / len(beta_values) if beta_values else 0.0

        results.append(
            {
                "delta": float(delta),
                "peak_jt": float(jt_list[max(range(len(r_op_mean)), key=lambda idx: r_op_mean[idx])]),
                "peak_r_op": float(max(r_op_mean)),
                "peak_width": peak_info["peak_width"],
                "peak_sharpness": peak_info["peak_sharpness"],
                "repulsion_beta": float(mean_beta),
                "n_realizations": int(n_realizations),
            }
        )
    return results


def optimal_disorder(disorder_results: List[Dict[str, float]]) -> Dict[str, float]:
    if not disorder_results:
        raise ValueError("disorder_results must not be empty")
    sharpness_values = [entry["peak_sharpness"] for entry in disorder_results]
    optimal_idx = max(range(len(sharpness_values)), key=lambda idx: sharpness_values[idx])
    baseline = max(sharpness_values[0], _EPS)
    optimal_value = sharpness_values[optimal_idx]
    return {
        "optimal_delta": float(disorder_results[optimal_idx]["delta"]),
        "optimal_sharpness": float(optimal_value),
        "baseline_sharpness": float(sharpness_values[0]),
        "enhancement_factor": float(optimal_value / baseline),
    }


def validate_residue_landscape(
    residue_map: Sequence[Sequence[float]],
    phi_landscape: Sequence[Sequence[float]],
    saddles: Sequence[Sequence[bool]],
    ep_mask: Sequence[Sequence[bool]],
) -> Dict[str, float | int | bool]:
    _ = phi_landscape
    saddles_arr = [list(row) for row in saddles]
    ep_arr = [list(row) for row in ep_mask]
    if len(saddles_arr) > len(ep_arr) + 1:
        saddles_arr = [row[1:-1] for row in saddles_arr[1:-1]]

    overlap_count = 0
    saddle_count = sum(1 for row in saddles_arr for flag in row if flag)
    ep_count = sum(1 for row in ep_arr for flag in row if flag)
    for y in range(min(len(saddles_arr), len(ep_arr))):
        for x in range(min(len(saddles_arr[0]), len(ep_arr[0]))):
            if saddles_arr[y][x] and ep_arr[y][x]:
                overlap_count += 1
    co_location = overlap_count / max(saddle_count, 1)
    return {
        "co_location_rate": float(co_location),
        "saddle_count": int(saddle_count),
        "ep_count": int(ep_count),
        "overlap_count": int(overlap_count),
        "passed": bool(co_location > 0.7),
    }
