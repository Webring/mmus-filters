# Фильтры Калмана и Язвинского

## Запуск
Установи uv
### Установка зависимостей
```bash
uv sync
```

### Запуск скрипта работы со микрофоном
```bash
uv run main.py
```
или сразу с нужным устройством (1 - id микрофона)
```bash
uv run main.py 1
```

### Запуск скрипта для подбора параметров Калмана
```bash
uv run .\kalman-params-main.py
```


## Примеры с формулами
```python
# Одномерный фильтр Калмана
def kalman_filter_1d(z, A=1, H=1, Q=0, R=1):
    x = 0.0
    P = 1.0
    filtered = []

    for measurement in z:
        #
        x_pred = A * x
        P_pred = A * P * A + Q

        #
        K = P_pred / (P_pred + R)
        x = x_pred + K * (measurement - H * x_pred)
        P = (1 - K * H) * P_pred

        filtered.append(x)

    return np.array(filtered)
```

```python
def adaptive_kalman_filter_general(
        z,
        Phi,
        H,
        R,
        Gamma,
        x0,
        P0
):
    """
    Адаптивный фильтр Калмана (Язвицкого), общий матричный случай

    Параметры
    ----------
    z : ndarray (N, m)
        Последовательность измерений
    Phi : ndarray (n, n)
        Матрица перехода состояния
    H : ndarray (m, n)
        Матрица наблюдений
    R : ndarray (m, m)
        Ковариация шума измерений
    Gamma : ndarray (n, q)
        Матрица шума процесса
    x0 : ndarray (n,)
        Начальная оценка состояния
    P0 : ndarray (n, n)
        Начальная ковариация ошибки

    Возвращает
    ----------
    x_hat : ndarray (N, n)
        Сглаженные оценки состояния
    """

    z = np.asarray(z)
    N = z.shape[0]
    n = x0.shape[0]

    x_hat = np.zeros((N, n))

    x = x0.copy()
    P = P0.copy()

    # Начальная ковариация шума процесса
    Q = np.zeros((Gamma.shape[1], Gamma.shape[1]))

    I = np.eye(n)

    for k in range(N):
        # ===== ПРОГНОЗ =====
        x_pred = Phi @ x
        P_pred = Phi @ P @ Phi.T + Gamma @ Q @ Gamma.T

        # ===== ИННОВАЦИЯ =====
        v = z[k] - H @ x_pred

        # ===== АДАПТИВНАЯ ОЦЕНКА Q (формула 13) =====
        HG = H @ Gamma
        denom = (HG.T @ HG)

        if np.linalg.matrix_rank(denom) == denom.shape[0]:
            Q_hat = np.linalg.inv(denom) @ (
                    HG.T @ (
                    np.outer(v, v)
                    - H @ Phi @ P @ Phi.T @ H.T
                    - R
            ) @ HG
            ) @ np.linalg.inv(denom)

            # Условие (14): положительная полуопределённость
            Q = np.maximum(Q_hat, 0)

        # ===== КОЭФФИЦИЕНТ КАЛМАНА =====
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # ===== КОРРЕКЦИЯ =====
        x = x_pred + K @ v
        P = (I - K @ H) @ P_pred

        x_hat[k] = x

    return x_hat
```