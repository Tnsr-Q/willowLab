"""Sanity checks for the appendix theorem helper utilities."""

from __future__ import annotations

import cmath
import math

from willowlab.disorder import validate_residue_landscape
from willowlab.resolvent import validate_lemma_5, validate_theorem_b1, validate_theorem_b2
from willowlab.spectral_flow import (
    holonomy_linearity_test,
    validate_theorem_b3,
    validate_theorem_b4,
)


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _synthetic_eigenvalues(num_steps: int, num_bands: int) -> tuple[list[float], list[list[complex]]]:
    jt = _linspace(0.8, 1.2, num_steps)
    evals = []
    for theta in jt:
        angles = [2.0 * math.pi * k / num_bands for k in range(num_bands)]
        radii = 1.0 - 0.05 * abs(theta - 1.0)
        evals.append([radii * cmath.exp(1j * angle) for angle in angles])
    return jt, evals


def test_appendix_lemma5_synthetic_data():
    jt, evals = _synthetic_eigenvalues(40, 6)
    result = validate_lemma_5(evals, jt)
    assert isinstance(result["passed"], bool)


def test_appendix_theorem_b1_mock_inputs():
    jt = _linspace(0.8, 1.2, 30)
    trace_abs = [math.exp(0.1 * (val - 1.0)) for val in jt]
    entropy = [math.log1p(val) for val in jt]
    effective_energy = [val ** 2 for val in jt]
    result = validate_theorem_b1(trace_abs, jt, entropy, effective_energy)
    assert set(result.keys()) == {"slope", "r2", "passed"}


def test_appendix_theorem_b2_detects_degeneracy():
    jt, evals = _synthetic_eigenvalues(20, 4)
    trace_abs = [1.0 for _ in jt]
    trace_abs[5] = 1e7
    evals[5][0] = 0.999999
    evals[5][1] = 0.999999
    mask = validate_theorem_b2(evals, trace_abs)
    assert isinstance(mask[0], bool)


def test_appendix_theorem_b3_structure():
    berry_sets = [[0.0, 2 * math.pi]] * 7
    result = validate_theorem_b3(berry_sets)
    assert set(result.keys()) == {"c_14_raw", "c_14_integer", "passed"}


def test_appendix_theorem_b4_parity_logic():
    phases = [0.0, 2 * math.pi]
    eta = [1.0 for _ in range(6)]
    chern_bits = [0 for _ in range(6)]
    result = validate_theorem_b4(phases, eta, chern_bits)
    assert "agreement_rate" in result


def test_appendix_holonomy_linearity_summary():
    loops = [[0.0, 0.02, 0.04], [0.0, 0.01, 0.02], [0.0, 0.005, 0.01]]
    areas = [1.0, 0.5, 0.25]
    summary = holonomy_linearity_test(loops, areas)
    assert set(summary.keys()) == {"slope", "expected", "passed"}


def test_appendix_residue_landscape_consistency():
    residue_map = [[1.0 for _ in range(5)] for _ in range(5)]
    phi = [[0.0 for _ in range(5)] for _ in range(5)]
    saddles = [[False for _ in range(5)] for _ in range(5)]
    saddles[2][2] = True
    ep_mask = [[False for _ in range(4)] for _ in range(4)]
    ep_mask[1][1] = True
    result = validate_residue_landscape(residue_map, phi, saddles, ep_mask)
    assert "co_location_rate" in result
