from typing import Iterable

import numpy as np

from filters.base import FilterBase, ArrayLike


class YazvinskyFilter(FilterBase):
    """
        Адаптивный фильтр Калмана (Язвицкого), общий матричный случай.

        Модель:
            x_k = Φ x_{k-1} + Γ w_k
            z_k = H x_k     + v_k

        где:
            w_k ~ N(0, Q) — оценивается адаптивно
            v_k ~ N(0, R)
        """

    def __init__(
            self,
            Phi: ArrayLike,
            H: ArrayLike,
            R: ArrayLike,
            Gamma: ArrayLike,
            x0: ArrayLike,
            P0: ArrayLike,
    ) -> None:
        self._Phi = self._to_matrix(Phi)
        self._H = self._to_matrix(H)
        self._R = self._to_matrix(R)
        self._Gamma = self._to_matrix(Gamma)

        self._x = self._to_vector(x0)
        self._P = self._to_matrix(P0)

        n = self._x.shape[0]
        q = self._Gamma.shape[1]

        # Адаптивная ковариация шума процесса
        self._Q = np.zeros((q, q))

        self._I = np.eye(n)


    @property
    def state(self) -> np.ndarray:
        """Текущее состояние (n, 1)"""
        return self._x

    @property
    def covariance(self) -> np.ndarray:
        """Ковариация ошибки P"""
        return self._P

    @property
    def Q(self) -> np.ndarray:
        """Адаптивная оценка ковариации шума процесса"""
        return self._Q



    def predict(self) -> None:
        """Шаг прогноза"""
        self._x = self._Phi @ self._x
        self._P = (
                self._Phi @ self._P @ self._Phi.T
                + self._Gamma @ self._Q @ self._Gamma.T
        )

    def update(self, z: ArrayLike) -> None:
        """Шаг коррекции + адаптация Q"""
        z = self._to_vector(z)

        # ===== Инновация =====
        v = z - self._H @ self._x

        # ===== Адаптивная оценка Q (Язвицкий) =====
        HG = self._H @ self._Gamma
        denom = HG.T @ HG

        if np.linalg.matrix_rank(denom) == denom.shape[0]:
            Q_hat = (
                    np.linalg.inv(denom)
                    @ (
                            HG.T
                            @ (
                                    v @ v.T
                                    - self._H @ self._Phi @ self._P @ self._Phi.T @ self._H.T
                                    - self._R
                            )
                            @ HG
                    )
                    @ np.linalg.inv(denom)
            )

            # Условие положительной полуопределённости
            self._Q = np.maximum(Q_hat, 0.0)

        # ===== Калмановский коэффициент =====
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # ===== Коррекция =====
        self._x = self._x + K @ v
        self._P = (self._I - K @ self._H) @ self._P

    # =========================
    # Batch-фильтрация
    # =========================

    def filter(self, measurements: Iterable[ArrayLike]) -> np.ndarray:
        """
        Прогон фильтра по последовательности измерений.

        Returns
        -------
        ndarray (N, n)
            Оценки состояния
        """
        estimates = []

        for z in measurements:
            self.predict()
            self.update(z)
            estimates.append(self._x.ravel())

        return np.asarray(estimates)