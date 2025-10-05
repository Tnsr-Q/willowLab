"""Helpers for Berry phase and Chern number calculations without NumPy."""

from __future__ import annotations

from cmath import phase
from math import fabs, log, pi
from typing import Dict, Iterable, List, Sequence

from willowlab import _numpy_shim as nps
from willowlab.tests.t_eta_lock import eta_lock_windows

_EPS = 1e-12


def _to_list(values: Sequence) -> List:
    if hasattr(values, "to_list"):
        return list(values.to_list())  # type: ignore[attr-defined]
    return list(values)


def _inner_product(v1: Sequence[complex], v2: Sequence[complex]) -> complex:
    return sum(a.conjugate() * b for a, b in zip(v1, v2))


def berry_connection(v1: Sequence[complex], v2: Sequence[complex]) -> complex:
    overlap = _inner_product(_to_list(v1), _to_list(v2))
    if abs(overlap) < _EPS:
        overlap = _EPS
    return 1j * phase(overlap)


def _column(matrix: Sequence[Sequence[complex]], idx: int) -> List[complex]:
    return [row[idx] for row in matrix]


def berry_phase_loop(evecs_loop: List[Sequence[Sequence[complex]]]) -> List[float]:
    if len(evecs_loop) < 2:
        raise ValueError("loop must contain at least two samples")
    n_bands = len(evecs_loop[0][0])
    phases = []
    for band in range(n_bands):
        total = 0.0
        for idx in range(len(evecs_loop) - 1):
            v_t = _column(evecs_loop[idx], band)
            v_next = _column(evecs_loop[idx + 1], band)
            total += berry_connection(v_t, v_next).imag
        v_last = _column(evecs_loop[-1], band)
        v_first = _column(evecs_loop[0], band)
        total += berry_connection(v_last, v_first).imag
        phases.append(total)
    return phases


def chern_number(berry_phases: Iterable[float]) -> int:
    total = sum(berry_phases)
    return int(round(total / (2.0 * pi)))


def quantization_check(berry_phases: Iterable[float], tol: float = 0.1) -> List[float]:
    deviations = []
    for phase_value in berry_phases:
        multiples = round(phase_value / (2.0 * pi))
        deviations.append(abs(phase_value - multiples * 2.0 * pi))
    return deviations


def validate_theorem_b4(
    berry_phases: Iterable[float],
    eta_oscillations: Iterable[float],
    chern_mod2: Iterable[int],
) -> Dict[str, float | int | bool]:
    phases = list(berry_phases)
    eta = list(eta_oscillations)
    chern_bits = list(chern_mod2)
    c = chern_number(phases)
    c_mod2 = c % 2
    eta_parity = [0 if value > 0 else 1 for value in eta]
    agreement = sum(1 for bit in eta_parity if bit == c_mod2)
    agreement_rate = agreement / len(eta_parity) if eta_parity else 0.0
    locks = eta_lock_windows(eta, chern_bits, window=5)
    lock_rate = (sum(1 for flag in locks if flag) / len(locks)) if len(locks) else 0.0
    return {
        "agreement_rate": float(agreement_rate),
        "lock_rate": float(lock_rate),
        "chern_number": int(c),
        "chern_mod2": int(c_mod2),
        "passed": bool(agreement_rate > 0.8 and lock_rate > 0.5),
    }


def c14_from_nested_loops(berry_phases_7tori: List[Iterable[float]]) -> Dict[str, float | int]:
    contributions = [chern_number(phases) for phases in berry_phases_7tori]
    c14_raw = float(sum(contributions))
    return {"c_14_raw": c14_raw, "c_14_integer": int(round(c14_raw))}


def validate_theorem_b3(berry_phases_7tori: List[Iterable[float]]) -> Dict[str, float | int | bool]:
    result = c14_from_nested_loops(berry_phases_7tori)
    return {
        "c_14_raw": result["c_14_raw"],
        "c_14_integer": result["c_14_integer"],
        "passed": bool(result["c_14_integer"] != 0),
    }


def berry_curvature_2form(evecs_grid: Sequence[Sequence[Sequence[Sequence[complex]]]]) -> List[List[float]]:
    ny = len(evecs_grid)
    nx = len(evecs_grid[0])
    n = len(evecs_grid[0][0])
    curvature = [[0.0 for _ in range(nx - 1)] for _ in range(ny - 1)]
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            for band in range(n):
                v00 = _column(evecs_grid[iy][ix], band)
                v01 = _column(evecs_grid[iy][ix + 1], band)
                v11 = _column(evecs_grid[iy + 1][ix + 1], band)
                v10 = _column(evecs_grid[iy + 1][ix], band)
                phase_val = 0.0
                phase_val += berry_connection(v00, v01).imag
                phase_val += berry_connection(v01, v11).imag
                phase_val += berry_connection(v11, v10).imag
                phase_val += berry_connection(v10, v00).imag
                curvature[iy][ix] += phase_val
    return [[value / (2.0 * pi) for value in row] for row in curvature]


def holonomy_linearity_test(
    berry_phases_loops: List[Iterable[float]],
    loop_areas: Iterable[float],
) -> Dict[str, float | bool]:
    holonomies = [sum(loop) for loop in berry_phases_loops]
    areas = list(loop_areas)
    if len(holonomies) != len(areas):
        raise ValueError("loop_areas length must match number of phase loops")
    if len(areas) < 3:
        return {"slope": float("nan"), "expected": 1.5, "passed": False}
    reference = areas[0] if areas[0] != 0 else 1.0
    predicted = [area * holonomies[0] / reference for area in areas]
    errors = [abs(h - p) for h, p in zip(holonomies, predicted)]
    log_areas = [log(abs(area) + _EPS) for area in areas]
    log_errors = [log(abs(err) + _EPS) for err in errors]
    slope = nps.polyfit(log_areas, log_errors, 1)[0]
    slope_val = float(slope)
    return {"slope": slope_val, "expected": 1.5, "passed": bool(1.3 < slope_val < 1.7)}
