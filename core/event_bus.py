from __future__ import annotations
import asyncio, uuid, time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict

Handler = Callable[[dict], Awaitable[None]]


class Topic(str, Enum):
    INTENT = "intent"
    EFFECT = "effect"
    ERROR = "error"
    TRACE = "trace"
    COUNCIL = "council"
    SYSTEM = "system"


class EventBus:
    def __init__(self):
        self.queues: Dict[Topic, asyncio.Queue] = {t: asyncio.Queue(maxsize=1000) for t in Topic}
        self.subscribers: Dict[Topic, list[Handler]] = {t: [] for t in Topic}
        self._tasks: list[asyncio.Task] = []
        self._started = False
        self._start_lock = asyncio.Lock()

    async def _drain(self, topic: Topic):
        while True:
            msg = await self.queues[topic].get()
            for handler in list(self.subscribers[topic]):
                try:
                    await handler(msg)
                except Exception as exc:
                    await self.publish(
                        Topic.ERROR,
                        {
                            "id": str(uuid.uuid4()),
                            "ts": time.time(),
                            "error": str(exc),
                            "during": topic.value,
                        },
                    )

    async def start(self):
        async with self._start_lock:
            if self._started:
                return
            loop = asyncio.get_running_loop()
            for topic in Topic:
                self._tasks.append(loop.create_task(self._drain(topic)))
            self._started = True

    async def stop(self):
        if not self._started:
            return
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        self._started = False

    def subscribe(self, topic: Topic, handler: Handler):
        self.subscribers[topic].append(handler)

    async def publish(self, topic: Topic, payload: Dict[str, Any]):
        if "id" not in payload:
            payload["id"] = str(uuid.uuid4())
        if "ts" not in payload:
            payload["ts"] = time.time()
        await self.queues[topic].put(payload)
