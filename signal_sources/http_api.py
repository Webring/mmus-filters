import asyncio
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional, Any

import httpx

from signal_sources.base import SignalSource


class ApiSource(SignalSource):
    def __init__(
        self,
        url: str,
        data_extractor: Callable[[str], Optional[float]],
        livetime: timedelta = timedelta(seconds=5),
        interval: float = 1.0,
        timeout: float = 5.0,
        headers: Optional[dict] = None,
    ):
        super().__init__(livetime=livetime, title="ApiSource")

        self.url = url
        self.data_extractor = data_extractor
        self.interval = interval
        self.timeout = timeout
        self.headers = headers or {}

        self._running = False

        # async-сущности (живут в отдельном потоке)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task | None = None
        self._thread: threading.Thread | None = None

    async def _poll_loop(self):
        assert self._client is not None

        while self._running:
            ts = datetime.now()

            try:
                resp = await self._client.get(self.url, timeout=self.timeout)
                resp.raise_for_status()

                payload = resp.text

                try:
                    value = self.data_extractor(payload)
                except Exception as e:
                    value = None
                    print(f"API extraction error: '{e}' on payload {payload}")

                if value is None:
                    print(f"API message: {payload}")
                else:
                    self._append(value, ts)

            except httpx.RequestError as e:
                print(f"API request error: {e}")
            except httpx.HTTPStatusError as e:
                print(f"API HTTP error: {e}")

            await asyncio.sleep(self.interval)

    def _run_loop(self):
        """
        Отдельный поток с собственным event loop.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._client = httpx.AsyncClient(headers=self.headers)

        self._task = self._loop.create_task(self._poll_loop())
        try:
            self._loop.run_until_complete(self._task)
        finally:
            self._loop.run_until_complete(self._client.aclose())
            self._loop.close()

    # ---- СИНХРОННЫЙ start ----
    def start(self):
        if self._running:
            return

        self._running = True

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )
        self._thread.start()

    # ---- СИНХРОННЫЙ stop ----
    def stop(self):
        if not self._running:
            return

        self._running = False

        # аккуратно завершить задачу и поток
        if self._loop and self._task:
            # пробуждаем цикл, чтобы он увидел флаг
            asyncio.run_coroutine_threadsafe(
                asyncio.sleep(0),
                self._loop,
            )

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._loop = None
        self._task = None
        self._client = None
