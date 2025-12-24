import threading
from datetime import datetime, timedelta
from typing import Callable, Optional

import serial
from signal_sources.base import SignalSource


class SerialSource(SignalSource):
    def __init__(
            self,
            port: str,
            data_extractor: Callable[[str], Optional[float]] = float,
            baudrate: int = 9600,
            livetime: timedelta = timedelta(seconds=5),
            interval: float = 1.0,
    ):
        super().__init__(livetime=livetime, title="SerialSource")

        self.port = port
        self.data_extractor = data_extractor
        self.baudrate = baudrate
        self.interval = interval

        self.ser: serial.Serial | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def _read_loop(self):
        """Фоновое чтение из COM-порта"""
        while self._running:
            try:
                ts = datetime.now()
                line = self.ser.readline().decode("utf-8").strip()

                if not line:
                    continue

                try:
                    value = self.data_extractor(line)
                except Exception as e:
                    value = None
                    print(f"Serial extraction error: '{e}' on line '{line}'")

                if not value:
                    print(f"Serial message: {line}")
                    continue

                self._append(value, ts)

            except serial.SerialException:
                break

    def start(self):
        if self._running:
            return

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.interval,
        )

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)

        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
