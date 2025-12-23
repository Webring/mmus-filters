from typing import Union, Iterable
import numpy as np

from filters.base import FilterBase, ArrayLike


class KalmanFilter(FilterBase):
    """
    Универсальный многомерный Калмановский фильтр.

    Поддерживает как скалярные, так и матричные модели.
    Все скаляры автоматически приводятся к матрицам размерности (1, 1).

    Модель:
        x_k   = A x_{k-1} + w
        z_k   = H x_k     + v

    где:
        w ~ N(0, Q)
        v ~ N(0, R)
    """

    def __init__(
            self,
            A: ArrayLike,
            H: ArrayLike,
            Q: ArrayLike,
            R: ArrayLike,
            x0: ArrayLike = 0.0,
            P0: ArrayLike = 1.0,
    ) -> None:
        self._A = self._to_matrix(A)
        self._H = self._to_matrix(H)
        self._Q = self._to_matrix(Q)
        self._R = self._to_matrix(R)
        self._x = self._to_vector(x0)
        self._P = self._to_matrix(P0)

    @property
    def state(self) -> np.ndarray:
        """Текущая оценка состояния x"""
        return self._x

    @property
    def covariance(self) -> np.ndarray:
        """Ковариация ошибки P"""
        return self._P

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def H(self) -> np.ndarray:
        return self._H

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @property
    def R(self) -> np.ndarray:
        return self._R

    @A.setter
    def A(self, value: ArrayLike) -> None:
        self._A = self._to_matrix(value)

    @H.setter
    def H(self, value: ArrayLike) -> None:
        self._H = self._to_matrix(value)

    @Q.setter
    def Q(self, value: ArrayLike) -> None:
        self._Q = self._to_matrix(value)

    @R.setter
    def R(self, value: ArrayLike) -> None:
        self._R = self._to_matrix(value)

    def predict(self) -> None:
        """Шаг предсказания"""
        self._x = self._A @ self._x
        self._P = self._A @ self._P @ self._A.T + self._Q

    def update(self, z: ArrayLike) -> None:
        """Шаг коррекции по измерению"""
        z = self._to_vector(z)

        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)

        y = z - self._H @ self._x
        self._x = self._x + K @ y
        self._P = (np.eye(self._P.shape[0]) - K @ self._H) @ self._P

    def filter(self, measurements: Iterable[ArrayLike]) -> np.ndarray:
        """
        Прогоняет фильтр по последовательности измерений.

        Returns:
            ndarray: массив оценок состояния
        """
        estimates = []

        for z in measurements:
            self.predict()
            self.update(z)
            estimates.append(self._x.ravel())

        return np.array(estimates)
