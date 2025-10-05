"""Utility helpers for η-lock tests without NumPy."""

from __future__ import annotations

from typing import Iterable, List


def eta_lock_windows(eta_series: Iterable[float], chern_mod2: Iterable[int], window: int = 5) -> List[bool]:
    eta_list = list(eta_series)
    chern_list = list(chern_mod2)
    locks: List[bool] = []
    for idx in range(max(len(eta_list) - window + 1, 0)):
        eta_slice = eta_list[idx : idx + window]
        chern_slice = chern_list[idx : idx + window]
        eta_locked = all(value > 0 for value in eta_slice) or all(value < 0 for value in eta_slice)
        chern_locked = len(set(chern_slice)) == 1
        locks.append(eta_locked and chern_locked)
    return locks


def decoder_priors_from_crossings(crossing_parity: Iterable[int]):
    priors = []
    for parity in crossing_parity:
        if parity % 2 == 1:
            priors.append({"parity_flip_bias": 0.7, "phase_flip_bias": 0.3})
        else:
            priors.append({"parity_flip_bias": 0.3, "phase_flip_bias": 0.7})
    return priors
