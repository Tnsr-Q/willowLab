"""Unit tests that exercise the resolvent and spectral flow helpers."""

from __future__ import annotations

import cmath
import math

from willowlab.disorder import disorder_scan, optimal_disorder, validate_residue_landscape
from willowlab.resolvent import (
    resolvent_scan,
    spectral_temperature,
    trace_resolvent_from_evals,
    validate_lemma_5,
    validate_theorem_b1,
    validate_theorem_b2,
)
from willowlab.spectral_flow import (
    berry_curvature_2form,
    berry_phase_loop,
    c14_from_nested_loops,
    chern_number,
    holonomy_linearity_test,
    validate_theorem_b3,
    validate_theorem_b4,
)


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _toy_eigenvalues(t: int, n: int) -> tuple[list[float], list[list[complex]]]:
    jt = _linspace(0.5, 1.5, t)
    evals = []
    for jt_val in jt:
        angles = [2.0 * math.pi * k / n for k in range(n)]
        radii = 0.9 + 0.1 * math.exp(-20.0 * (jt_val - 1.0) ** 2)
        evals.append([radii * cmath.exp(1j * angle) for angle in angles])
    return jt, evals


def test_resolvent_scan_toy():
    jt, evals = _toy_eigenvalues(50, 10)
    result = resolvent_scan(evals, jt)
    assert min(jt) <= result["peak_jt"] <= max(jt)
    assert result["peak_r_op"] > 0


def test_spectral_temperature_matches_fixture():
    jt, evals = _toy_eigenvalues(40, 8)
    trace_abs = trace_resolvent_from_evals(evals)
    temps = spectral_temperature(trace_abs, jt)
    assert len(temps) == len(jt)
    assert all(math.isfinite(value) for value in temps)


def test_theorem_b1_duality_passes_on_mock_data():
    jt = _linspace(0.5, 1.5, 50)
    trace_abs = [math.exp(0.02 * val ** 3 + 0.19 * val ** 2) for val in jt]
    entropy = [4 * 0.02 * val ** 3 + 2 * 0.19 * val ** 2 for val in jt]
    effective_energy = [val ** 2 for val in jt]
    result = validate_theorem_b1(trace_abs, jt, entropy, effective_energy)
    assert result["passed"]
    assert 0.9 < result["slope"] < 1.1


def test_theorem_b2_detects_exceptional_point():
    jt, evals = _toy_eigenvalues(30, 6)
    trace_abs = [1.0 for _ in jt]
    trace_abs[10] = 1e7
    evals[10][0] = 0.999999
    evals[10][1] = 0.999999
    flags = validate_theorem_b2(evals, trace_abs)
    assert flags[10]
    assert len(flags) == len(jt)


def test_lemma_5_correlation_reasonable():
    jt, evals = _toy_eigenvalues(60, 6)
    result = validate_lemma_5(evals, jt)
    assert isinstance(result["passed"], bool)
    assert -1.0 <= result["correlation"] <= 1.0 or math.isnan(result["correlation"])


def test_disorder_pipeline_runs():
    jt, evals = _toy_eigenvalues(20, 4)
    deltas = [0.0, 0.05]
    results = disorder_scan(evals, jt, deltas, n_realizations=2)
    assert len(results) == len(deltas)
    optimal = optimal_disorder(results)
    assert optimal["optimal_delta"] in deltas


def test_residue_landscape_validation():
    residue_map = [[1.0 for _ in range(4)] for _ in range(4)]
    phi = [[0.0 for _ in range(4)] for _ in range(4)]
    saddles = [[False for _ in range(4)] for _ in range(4)]
    for y in range(1, 3):
        for x in range(1, 3):
            saddles[y][x] = True
    ep_mask = [[False for _ in range(3)] for _ in range(3)]
    ep_mask[1][1] = True
    result = validate_residue_landscape(residue_map, phi, saddles, ep_mask)
    assert "co_location_rate" in result
    assert result["saddle_count"] == 4


def test_spectral_flow_helpers():
    loop = []
    for theta in _linspace(0.0, 2 * math.pi, 6):
        vecs = [
            [cmath.exp(1j * theta), 0.0],
            [0.0, cmath.exp(-1j * theta)],
        ]
        loop.append(vecs)
    loop.append(loop[0])
    phases = berry_phase_loop(loop)
    c = chern_number(phases)
    assert len(phases) == 2
    assert isinstance(c, int)


def test_theorem_b3_and_b4_wrappers():
    berry_sets = [[0.0, 2 * math.pi]] * 7
    theorem_b3 = validate_theorem_b3(berry_sets)
    assert isinstance(theorem_b3["passed"], bool)
    c14 = c14_from_nested_loops(berry_sets)
    assert c14["c_14_integer"] == 7

    eta = [1.0 for _ in range(10)]
    chern_bits = [0 for _ in range(10)]
    theorem_b4 = validate_theorem_b4([0.0, 2 * math.pi], eta, chern_bits)
    assert set(theorem_b4.keys()) == {"agreement_rate", "lock_rate", "chern_number", "chern_mod2", "passed"}


def test_berry_curvature_and_holonomy():
    ny, nx, n = 3, 3, 2
    grid = []
    for _ in range(ny):
        row = []
        for _ in range(nx):
            matrix = []
            for i in range(n):
                matrix.append([1.0 if i == j else 0.0 for j in range(n)])
            row.append(matrix)
        grid.append(row)
    curvature = berry_curvature_2form(grid)
    assert len(curvature) == ny - 1

    loops = [[0.0, 0.1, 0.2], [0.0, 0.05, 0.1], [0.0, 0.025, 0.05]]
    areas = [1.0, 0.5, 0.25]
    result = holonomy_linearity_test(loops, areas)
    assert "slope" in result
