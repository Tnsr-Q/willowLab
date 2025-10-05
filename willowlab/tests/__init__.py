"""Test helpers for the willowLab suite."""
import importlib
import sys

try:
    import numpy  # type: ignore # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised during tests
    shim = importlib.import_module('willowlab._numpy_shim')
    sys.modules.setdefault('numpy', shim)
