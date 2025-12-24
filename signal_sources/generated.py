import threading
from datetime import datetime, timedelta
from typing import Callable

from signal_sources.base import SignalSource


class GeneratedSource(SignalSource):
    def __init__(
            self,
            func: Callable[[datetime], float],
            interval: float = 0.05,
            livetime: timedelta = timedelta(seconds=5),
    ):
        super().__init__(livetime=livetime, title="GeneratedFunc")

        self.interval = interval
        self.func = func

        self._thread: threading.Thread | None = None
        self._running = False

    def _run(self):
        """
        Основной цикл генерации синуса
        """
        while self._running:
            now = datetime.now()

            value = self.func(now)

            self._append(value, now)

            threading.Event().wait(self.interval)

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread = None
