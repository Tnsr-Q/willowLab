from __future__ import annotations

import builtins
import math
from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float]


class SimpleArray:
    """A tiny, list-backed array object with just enough NumPy-like behaviour."""

    def __init__(self, values: Iterable[Number]):
        self._values = list(values)

    # Basic container protocol -------------------------------------------------
    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._values)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return repr(self._values)

    __str__ = __repr__

    def __bool__(self):  # pragma: no cover - mimics NumPy semantics
        raise ValueError("The truth value of an array is ambiguous")

    def __getitem__(self, item):
        if isinstance(item, SimpleArray):
            mask = [bool(v) for v in item._values]
            return SimpleArray(v for v, flag in zip(self._values, mask) if flag)
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], bool):
            mask = [bool(v) for v in item]
            return SimpleArray(v for v, flag in zip(self._values, mask) if flag)
        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError("Only two-dimensional indexing is supported")
            row = self.__getitem__(item[0])
            if isinstance(row, SimpleArray):
                return row.__getitem__(item[1])
            raise TypeError("Indexing requires an array result")

        result = self._values[item]
        if isinstance(item, slice):
            return SimpleArray(result)
        return result

    # Internal helpers ---------------------------------------------------------
    def _coerce(self, other: Union[Sequence[Number], Number]):
        if isinstance(other, SimpleArray):
            values = other._values
        elif isinstance(other, (list, tuple)):
            values = list(other)
        elif isinstance(other, (int, float)):
            values = [other] * len(self._values)
        else:
            return NotImplemented
        if len(values) != len(self._values):
            raise ValueError("operands must have the same length")
        return values

    def _binary(self, other, op):
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        return SimpleArray(op(a, b) for a, b in zip(self._values, coerced))

    # Arithmetic operations ----------------------------------------------------
    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return SimpleArray(other - a for a in self._values)
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        return SimpleArray(b - a for a, b in zip(self._values, coerced))

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return SimpleArray(other / a for a in self._values)
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        return SimpleArray(b / a for a, b in zip(self._values, coerced))

    # Comparisons --------------------------------------------------------------
    def _compare(self, other, op):
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        return SimpleArray(op(a, b) for a, b in zip(self._values, coerced))

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __eq__(self, other):  # type: ignore[override]
        result = self._compare(other, lambda a, b: a == b)
        if result is NotImplemented:
            return False
        return result

    def __ne__(self, other):  # type: ignore[override]
        result = self._compare(other, lambda a, b: a != b)
        if result is NotImplemented:
            return True
        return result

    def __and__(self, other):
        result = self._compare(other, lambda a, b: bool(a) and bool(b))
        if result is NotImplemented:
            return NotImplemented
        return result

    def __rand__(self, other):
        return self.__and__(other)

    # Utility helpers ----------------------------------------------------------
    def to_list(self) -> List[Number]:
        return list(self._values)


def _to_sequence(values: Union[Sequence[Number], Number]) -> List[Number]:
    if isinstance(values, (list, tuple)):
        return list(values)
    if isinstance(values, SimpleArray):
        return values.to_list()
    return [values]  # type: ignore[list-item]


def array(values: Union[Sequence[Number], Number], dtype=None):
    seq = _to_sequence(values)
    if dtype is bool:
        return SimpleArray(bool(v) for v in seq)
    return SimpleArray(seq)


def unique(values: Sequence[Number]) -> List[Number]:
    seen = set()
    result: List[Number] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def sign(values: Union[Sequence[Number], Number]):
    seq = _to_sequence(values)
    result = [1 if value > 0 else -1 if value < 0 else 0 for value in seq]
    if isinstance(values, (list, tuple, SimpleArray)):
        return SimpleArray(result)
    return result[0]


def log(values: Union[Sequence[float], float]):
    if isinstance(values, (list, tuple, SimpleArray)):
        return SimpleArray(math.log(v) for v in values)
    return math.log(values)


def abs(values: Union[Sequence[Number], Number]):  # pylint: disable=redefined-builtin
    if isinstance(values, (list, tuple, SimpleArray)):
        return SimpleArray(builtins.abs(v) for v in values)
    return builtins.abs(values)


def gradient(values: Sequence[float], coordinates: Sequence[float] | None = None) -> SimpleArray:
    y = _to_sequence(values)
    n = len(y)
    if n == 0:
        return SimpleArray([])
    if coordinates is None:
        x = list(range(n))
    else:
        x = _to_sequence(coordinates)
        if len(x) != n:
            raise ValueError("x and y must have the same length")

    if n == 1:
        return SimpleArray([0.0])

    gradients: List[float] = []
    for i in range(n):
        if i == 0:
            dy = y[1] - y[0]
            dx = x[1] - x[0]
        elif i == n - 1:
            dy = y[-1] - y[-2]
            dx = x[-1] - x[-2]
        else:
            dy = y[i + 1] - y[i - 1]
            dx = x[i + 1] - x[i - 1]
        gradients.append(dy / dx if dx != 0 else 0.0)
    return SimpleArray(gradients)


def polyfit(x_values: Sequence[float], y_values: Sequence[float], degree: int) -> SimpleArray:
    if degree != 1:
        raise NotImplementedError("Only degree-1 fits are supported in this shim")

    x = _to_sequence(x_values)
    y = _to_sequence(y_values)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n == 0:
        return SimpleArray([0.0, 0.0])

    x_mean = sum(x) / n
    y_mean = sum(y) / n
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    if denominator == 0:
        slope = 0.0
    else:
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return SimpleArray([slope, intercept])


def corrcoef(x_values: Sequence[float], y_values: Sequence[float]) -> SimpleArray:
    x = _to_sequence(x_values)
    y = _to_sequence(y_values)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n == 0:
        return SimpleArray([SimpleArray([1.0, 0.0]), SimpleArray([0.0, 1.0])])

    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    x_var = sum((xi - x_mean) ** 2 for xi in x)
    y_var = sum((yi - y_mean) ** 2 for yi in y)
    denom = math.sqrt(x_var * y_var)
    if denom == 0:
        r = 0.0
    else:
        r = cov / denom
    return SimpleArray([SimpleArray([1.0, r]), SimpleArray([r, 1.0])])
