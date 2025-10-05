"""Utilities for working with resolvent-based witnesses without NumPy."""

from __future__ import annotations

from cmath import phase
from math import cos, fabs, log, sin, sqrt
from typing import Dict, Iterable, List, Sequence

from willowlab import _numpy_shim as nps

_EPS = 1e-12


def _to_list(values: Sequence) -> List:
    if hasattr(values, "to_list"):
        return list(values.to_list())  # type: ignore[attr-defined]
    return list(values)


def _as_matrix(evals: Sequence[Sequence[complex]]) -> List[List[complex]]:
    if hasattr(evals, "to_list"):
        evals = evals.to_list()  # type: ignore[assignment, attr-defined]
    if not isinstance(evals, Sequence):
        raise ValueError("eigenvalues must be a sequence of sequences")
    if evals and not isinstance(evals[0], Sequence):
        return [list(map(complex, evals))]  # type: ignore[arg-type]
    return [list(map(complex, row)) for row in evals]  # type: ignore[arg-type]


def _argmax(values: Sequence[float]) -> int:
    max_idx = 0
    max_val = float("-inf")
    for idx, value in enumerate(values):
        if value > max_val:
            max_val = value
            max_idx = idx
    return max_idx


def _product(values: Sequence[complex]) -> complex:
    result = 1 + 0j
    for value in values:
        result *= value
    return result


def _gradient(values: Sequence[float], coordinates: Sequence[float]) -> List[float]:
    grad = nps.gradient(values, coordinates)
    return grad.to_list() if hasattr(grad, "to_list") else list(grad)


def _polyfit_slope(x_vals: Sequence[float], y_vals: Sequence[float]) -> float:
    coeffs = nps.polyfit(x_vals, y_vals, 1)
    seq = coeffs.to_list() if hasattr(coeffs, "to_list") else list(coeffs)
    return float(seq[0])


def _corrcoef(x_vals: Sequence[float], y_vals: Sequence[float]) -> float:
    matrix = nps.corrcoef(x_vals, y_vals)
    rows = matrix.to_list() if hasattr(matrix, "to_list") else list(matrix)
    row = rows[0].to_list() if hasattr(rows[0], "to_list") else list(rows[0])
    return float(row[1])


def _phase_align(evals: Sequence[Sequence[complex]]) -> List[List[complex]]:
    arr = _as_matrix(evals)
    aligned: List[List[complex]] = []
    for row in arr:
        if not row:
            aligned.append([])
            continue
        avg_phase = phase(_product(row)) / len(row)
        factor = complex(cos(avg_phase), -sin(avg_phase))
        aligned.append([value * factor for value in row])
    return aligned


def trace_resolvent_from_evals(evals: Sequence[Sequence[complex]], safe: bool = True) -> List[float]:
    matrix = _as_matrix(evals)
    results: List[float] = []
    for row in matrix:
        total = 0.0
        for value in row:
            radius = abs(value)
            if safe and fabs(radius - 1.0) < 1e-6:
                angle = phase(value)
                sin_half = max(fabs(sin(angle / 2.0)), _EPS)
                total += 1.0 / (2.0 * sin_half)
            else:
                total += 1.0 / max(abs(1.0 - value), _EPS)
        results.append(total)
    return results


def r_op_from_trace(trace_abs: Sequence[float], n: int, kappa: float = sqrt(3.0)) -> List[float]:
    return [kappa * value / max(n, 1) for value in trace_abs]


def resolvent_scan(
    evals: Sequence[Sequence[complex]],
    jt_values: Iterable[float],
    align_phase: bool = True,
    kappa: float = sqrt(3.0),
) -> Dict[str, object]:
    matrix = _as_matrix(evals)
    jt = list(jt_values)
    if len(matrix) != len(jt):
        raise ValueError("JT values must align with eigenvalue trajectory")

    if align_phase:
        matrix = _phase_align(matrix)

    trace_abs = trace_resolvent_from_evals(matrix, safe=True)
    n = len(matrix[0]) if matrix else 1
    r_op = r_op_from_trace(trace_abs, n, kappa=kappa)
    min_dist = [min(abs(1.0 - value) for value in row) for row in matrix]
    peak_idx = _argmax(r_op)

    return {
        "JT": jt,
        "trace_abs": trace_abs,
        "r_op": r_op,
        "min_dist_to_one": min_dist,
        "peak_jt": float(jt[peak_idx]),
        "peak_r_op": float(r_op[peak_idx]),
        "peak_idx": peak_idx,
    }


def spectral_temperature(trace_abs: Sequence[float], jt: Sequence[float]) -> List[float]:
    log_trace = [log(value + _EPS) for value in trace_abs]
    first = _gradient(log_trace, jt)
    second = _gradient(first, jt)
    return [1.0 / (fabs(value) + _EPS) for value in second]


def validate_theorem_b1(
    trace_abs: Sequence[float],
    jt: Sequence[float],
    entropy: Sequence[float],
    effective_energy: Sequence[float],
    window: float = 0.05,
) -> Dict[str, float | bool]:
    t_spec = spectral_temperature(trace_abs, jt)
    from willowlab.tests.t_spec_ent import entanglement_temperature

    t_ent = entanglement_temperature(entropy, effective_energy)
    indices = [idx for idx, value in enumerate(jt) if 1.0 - window < value < 1.0 + window]
    if len(indices) < 4:
        return {"slope": float("nan"), "r2": float("nan"), "passed": False}

    a = [log(t_spec[idx] + _EPS) for idx in indices]
    b = [log(t_ent[idx] + _EPS) for idx in indices]
    slope = _polyfit_slope(a, b)
    r2 = _corrcoef(a, b) ** 2
    passed = bool(fabs(slope - 1.0) < 0.1 and r2 > 0.9)
    return {"slope": slope, "r2": r2, "passed": passed}


def validate_theorem_b2(
    evals: Sequence[Sequence[complex]],
    trace_abs: Sequence[float],
    tol: float = 1e-8,
) -> List[bool]:
    matrix = _as_matrix(evals)
    trace_list = _to_list(trace_abs)
    results: List[bool] = []
    for row, trace_value in zip(matrix, trace_list):
        if trace_value <= 1e6:
            results.append(True)
            continue
        det_value = abs(_product([1.0 - value for value in row]))
        sorted_abs = sorted(abs(value) for value in row)
        gaps = [b - a for a, b in zip(sorted_abs, sorted_abs[1:])]
        degenerate = bool(gaps and min(gaps) < tol)
        has_ep = det_value < tol or degenerate
        results.append(has_ep)
    return results


def validate_lemma_5(evals: Sequence[Sequence[complex]], jt: Sequence[float]) -> Dict[str, float | bool]:
    matrix = _as_matrix(evals)
    jt_list = _to_list(jt)
    trace_abs = trace_resolvent_from_evals(matrix, safe=False)
    numeric = _gradient(trace_abs, jt_list)

    analytic = [0.0 for _ in matrix]
    for idx in range(1, len(matrix)):
        delta = jt_list[idx] - jt_list[idx - 1]
        if fabs(delta) < _EPS:
            continue
        total = 0.0
        for curr, prev in zip(matrix[idx], matrix[idx - 1]):
            d_lambda = (curr - prev) / delta
            denom = 1.0 - curr
            total += (d_lambda / (denom * denom)).real
        analytic[idx] = total

    indices = [i for i, value in enumerate(jt_list) if 0.95 < value < 1.05]
    if len(indices) < 4:
        return {"correlation": float("nan"), "passed": False}

    numeric_vals = [numeric[i] for i in indices]
    analytic_vals = [analytic[i] for i in indices]
    corr = _corrcoef(numeric_vals, analytic_vals)
    return {"correlation": corr, "passed": bool(fabs(corr) > 0.8)}
