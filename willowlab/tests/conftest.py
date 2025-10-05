"""Pytest configuration and fixtures for the willowLab test-suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import math

from willowlab import _numpy_shim as np


@dataclass
class DualitySample:
    resolvent_trace: List[float]
    JT_scan_points: List[float]
    entropy: List[float]
    effective_energy: List[float]


def _build_sample() -> DualitySample:
    JT = [0.9 + 0.01 * i for i in range(21)]
    coeff_a = 0.02
    coeff_b = 0.19
    log_trace = [coeff_a * jt**3 + coeff_b * jt**2 for jt in JT]
    resolvent = [math.exp(val) for val in log_trace]
    effective_energy = [jt**2 for jt in JT]
    entropy = [4 * coeff_a * jt**3 + 2 * coeff_b * jt**2 for jt in JT]
    return DualitySample(
        resolvent_trace=np.array(resolvent),
        JT_scan_points=np.array(JT),
        entropy=np.array(entropy),
        effective_energy=np.array(effective_energy),
    )


def pytest_configure(config):  # pragma: no cover - called by pytest
    config.duality_sample = _build_sample()


def pytest_unconfigure(config):  # pragma: no cover - cleanup hook
    if hasattr(config, "duality_sample"):
        delattr(config, "duality_sample")


import pytest


@pytest.fixture
def ds() -> DualitySample:
    """Provide a deterministic dataset for duality checks."""
    return _build_sample()
