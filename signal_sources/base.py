from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import heapq
from typing import List, Tuple


class SignalSource(ABC):
    def __init__(self, livetime: timedelta, title="MyGraph"):
        self._title = title
        self.livetime = livetime
        self._queue: List[Tuple[datetime, float]] = []

    def _append(self, value: float, ts: datetime | None = None):
        """
        Добавить значение с таймстемпом.
        Если ts не передан — используется текущее время.
        """
        if ts is None:
            ts = datetime.now()

        heapq.heappush(self._queue, (ts, value))
        self._cleanup()

    def _cleanup(self):
        """
        Удаляет все элементы, которые старше livetime
        """
        if self.livetime is None:
            return

        threshold = datetime.now() - self.livetime

        while self._queue and self._queue[0][0] < threshold:
            heapq.heappop(self._queue)

    def get_buffer(self) -> List[Tuple[datetime, float]]:
        """
        Возвращает актуальный список значений
        (отсортирован по времени)
        """
        self._cleanup()
        return sorted(list(self._queue), key=lambda x: x[0])


    def get_values(self) -> List[float]:
        """
        Только значения (без ts)
        """
        self._cleanup()
        return [value for _, value in self._queue]

    def get_latest(self) -> Tuple[datetime, float] | None:
        """
        Последнее по времени значение
        """
        self._cleanup()
        if not self._queue:
            return None
        return max(self._queue, key=lambda x: x[0])

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title: str):
        self._title = title

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass
