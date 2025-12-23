from abc import ABC
from typing import Union, Iterable

import numpy as np

ArrayLike = Union[float, int, np.ndarray]


class FilterBase(ABC):

    def filter(self, measurements: Iterable[ArrayLike]) -> np.ndarray:
        ...

    def __call__(self, measurements: Iterable[ArrayLike]) -> np.ndarray:
        return self.filter(measurements)

    @staticmethod
    def _to_matrix(x: ArrayLike) -> np.ndarray:
        """Приведение скаляра или массива к 2D-матрице"""
        if isinstance(x, (int, float)):
            return np.array([[x]], dtype=float)
        x = np.asarray(x, dtype=float)
        return x if x.ndim == 2 else np.atleast_2d(x)

    @staticmethod
    def _to_vector(x: ArrayLike) -> np.ndarray:
        """Приведение скаляра или массива к вектору-столбцу"""
        if isinstance(x, (int, float)):
            return np.array([[x]], dtype=float)
        x = np.asarray(x, dtype=float)
        return x.reshape(-1, 1)
