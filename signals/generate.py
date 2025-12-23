from typing import Callable

import numpy as np


def normally_noisy(
        func: Callable,
        start: float = 0,
        end: float = 1,
        density: int = 100,
        noise_sigma: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерируем сигнал с нормально распределенным шумом и заданными параметрами

    :param func:
    :param start:
    :param end:
    :param density:
    :param noise_sigma:
    :return:
    """
    t = np.linspace(start, end, density)
    true_signal = func(t)
    noise = np.random.normal(0, noise_sigma, len(true_signal))
    signal = true_signal + noise
    return t, true_signal, signal
