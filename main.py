import sys

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from abc import ABC, abstractmethod
from matplotlib.animation import FuncAnimation

from filters.yazvinsky import YazvinskyFilter


# ============================================================
# БАЗОВЫЙ ИСТОЧНИК
# ============================================================

class SignalSource(ABC):
    def __init__(self, history=300):
        self.buffer = deque([0.0] * history, maxlen=history)
        self.sample_index = 0

    def _append(self, value: float):
        self.buffer.append(value)
        self.sample_index += 1

    def get_buffer(self):
        return self.buffer

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


# ============================================================
# МИКРОФОН (RAW)
# ============================================================

class MicrophoneSource(SignalSource):
    def __init__(
            self,
            device,
            samplerate=44100,
            blocksize=1024,
            gain=1.0,
            history=300
    ):
        super().__init__(history)
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.gain = gain
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        volume = np.sqrt(np.mean(indata ** 2))
        self._append(volume * self.gain)

    def start(self):
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


# ============================================================
# ТВОЙ КАЛМАН-ФИЛЬТР (УПРОЩЁННО)
# ============================================================

class KalmanFilter:
    def __init__(self, A, H, Q, R, x0=0.0, P0=1.0):
        self.A = np.atleast_2d(A)
        self.H = np.atleast_2d(H)
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        self.x = np.atleast_2d(x0).reshape(-1, 1)
        self.P = np.atleast_2d(P0)

    @property
    def state(self):
        return self.x

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.atleast_2d(z).reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P


# ============================================================
# МИКРОФОН С КАЛМАНОМ (ВСЁ В ОДНОМ CALLBACK)
# ============================================================

class KalmanMicrophoneSource(MicrophoneSource):
    def __init__(
            self,
            device,
            kalman: KalmanFilter,
            samplerate=44100,
            blocksize=1024,
            gain=1.0,
            history=300
    ):
        super().__init__(device, samplerate, blocksize, gain, history)
        self.kalman = kalman
        self.filtered_buffer = deque([0.0] * history, maxlen=history)

    def _callback(self, indata, frames, time_info, status):
        # RAW
        volume = np.sqrt(np.mean(indata ** 2))
        raw = volume * self.gain
        self._append(raw)

        # KALMAN (СТРОГО 1:1)
        self.kalman.predict()
        self.kalman.update(raw)

        filtered = float(self.kalman.state.ravel()[0])
        self.filtered_buffer.append(filtered)


# ============================================================
# ОТОБРАЖЕНИЕ
# ============================================================

def plot_microphone(mic: KalmanMicrophoneSource):
    fig, ax = plt.subplots()

    raw_line, = ax.plot(mic.buffer, label="Raw")
    kalman_line, = ax.plot(mic.filtered_buffer, label="Kalman")

    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()

    def update(frame):
        raw_line.set_ydata(mic.buffer)
        kalman_line.set_ydata(mic.filtered_buffer)

        raw_line.set_xdata(range(len(mic.buffer)))
        kalman_line.set_xdata(range(len(mic.filtered_buffer)))

        # ax.relim()
        # ax.autoscale_view()
        return raw_line, kalman_line

    ani = FuncAnimation(fig, update, interval=30)
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) == 2:
        device_id = int(sys.argv[1])
    else:
        devices = sd.query_devices()
        input_devices = [
            (i, d['name'], d['max_input_channels'])
            for i, d in enumerate(devices)
            if d['max_input_channels'] > 0
        ]

        for idx, name, ch in input_devices:
            print(f"{idx}: {name} (inputs: {ch})")

        device_id = int(input("Enter device id (-1 to quit): "))

    if device_id == -1:
        exit("EXITING")

    filter = YazvinskyFilter(
        1.0,
        1.0,
        0.2,
        5,
        x0=0.0,
        P0=1.0
    )

    mic = KalmanMicrophoneSource(
        device=device_id,
        kalman=filter,
        gain=1.0
    )

    try:
        mic.start()
        plot_microphone(mic)
    except sd.PortAudioError:
        print("Невозможно открыть указанный микрофон")
    finally:
        mic.stop()
