import numpy as np
import sounddevice as sd
from datetime import datetime, timedelta

from signal_sources.base import SignalSource


class MicrophoneSource(SignalSource):
    def __init__(
        self,
        device,
        samplerate=44100,
        blocksize=1024,
        gain=1.0,
        livetime: timedelta = timedelta(seconds=5),
    ):
        super().__init__(livetime=livetime)

        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.gain = gain
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # при желании можно логировать
            pass

        # RMS громкость
        volume = np.sqrt(np.mean(indata ** 2)) * self.gain

        # timestamp — текущее время
        ts = datetime.now()

        self._append(volume, ts)

    def start(self):
        if self.stream is not None:
            return

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
