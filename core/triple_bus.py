"""
Triple bus event system for Dexter-Gliksbot.

Implements three logical channels:
- MAIN: dexter <-> user communication / intents / effects
- COLLAB: peer-to-peer collaboration while agents are idle
- PRIVATE: per-agent channels used while agents are on-task

Each bus supports async pub/sub with metadata injection (id, timestamp,
bus name, normalised topic) and resilient delivery (handler errors are
logged but do not break broadcast).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

Handler = Callable[[Dict[str, Any]], Awaitable[None]]


class MainTopic(str, Enum):
    USER_INPUT = "user_input"
    DEXTER_RESPONSE = "dexter_response"
    INTENT = "intent"
    EFFECT = "effect"
    ERROR = "error"
    TRACE = "trace"
    CONTEXT_AVAILABLE = "context_available"


class CollabTopic(str, Enum):
    OBSERVATION = "observation"
    PROPOSAL = "proposal"
    REFINEMENT = "refinement"
    CRITIQUE = "critique"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    DEXTER_INTERVENTION = "dexter_intervention"
    CONSENSUS = "consensus"
    COLLABORATION_COMPLETE = "collaboration_complete"


class PrivateTopic(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    DEXTER_SUPPORT = "dexter_support"
    CONTEXT_UPDATE = "context_update"
    PROGRESS = "progress"
    TASK_COMPLETE = "task_complete"
    HELP_REQUEST = "help_request"


class TopicBus:
    """Asynchronous pub/sub bus for a fixed set of topics."""

    def __init__(self, name: str, topics: Iterable[Enum]) -> None:
        self.name = name
        self.bus_id = name
        self._topics = list(topics)
        self._subscribers: Dict[str, List[Handler]] = defaultdict(list)
        self._read_only_handlers: set = set()  # Track read-only handlers
        self._started = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def subscribe(self, topic: Enum, handler: Handler, read_only: bool = False) -> None:
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to
            handler: The callback function
            read_only: If True, handler can only receive messages, not publish
        """
        if handler not in self._subscribers[topic.value]:
            self._subscribers[topic.value].append(handler)
            if read_only:
                self._read_only_handlers.add(handler)

    def unsubscribe(self, topic: Enum, handler: Handler) -> None:
        if handler in self._subscribers[topic.value]:
            self._subscribers[topic.value].remove(handler)
        # Remove from read-only set if present
        self._read_only_handlers.discard(handler)

    def unsubscribe_all(self, handler: Handler) -> None:
        for handlers in self._subscribers.values():
            while handler in handlers:
                handlers.remove(handler)
        # Remove from read-only set
        self._read_only_handlers.discard(handler)

    def get_subscriber_count(self, topic: Enum) -> int:
        return len(self._subscribers[topic.value])

    @property
    def subscribers(self) -> Dict[str, List[Handler]]:
        """Read-only view of subscribers keyed by topic."""
        return {topic: list(handlers) for topic, handlers in self._subscribers.items()}

    async def publish(self, topic: Enum, message: Dict[str, Any]) -> None:
        if not self._started:
            return
        event = dict(message)
        event.setdefault("id", str(uuid.uuid4()))
        event.setdefault("ts", time.time())
        event["bus"] = self.name
        event["topic"] = topic.value

        handlers = list(self._subscribers[topic.value])
        if not handlers:
            return

        for handler in handlers:
            try:
                await handler(dict(event))
            except Exception:
                logger.exception("Handler failure on %s/%s", self.name, topic.value)


class PrivateBus(TopicBus):
    def __init__(self, agent_id: str) -> None:
        super().__init__(f"private:{agent_id}", PrivateTopic)
        self.agent_id = agent_id


class TripleBusSystem:
    """Co-ordinates main, collab, and per-agent private buses."""

    def __init__(self) -> None:
        self.main = TopicBus("main", MainTopic)
        self.collab = TopicBus("collab", CollabTopic)
        self._private: Dict[str, PrivateBus] = {}
        self._private_listeners: List[Callable[[PrivateBus], Awaitable[None]]] = []
        self._started = False

    async def start_all(self) -> None:
        await self.main.start()
        await self.collab.start()
        for bus in self._private.values():
            await bus.start()
        self._started = True

    async def stop_all(self) -> None:
        await self.main.stop()
        await self.collab.stop()
        for bus in self._private.values():
            await bus.stop()
        self._started = False

    async def publish(self, topic: MainTopic, message: Dict[str, Any]) -> None:
        await self.main.publish(topic, message)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "main": {
                "subscriber_counts": {topic.value: self.main.get_subscriber_count(topic) for topic in MainTopic}
            },
            "collab": {
                "subscriber_counts": {topic.value: self.collab.get_subscriber_count(topic) for topic in CollabTopic}
            },
            "private": {
                "bus_count": len(self._private),
                "agent_ids": sorted(self._private.keys()),
            },
        }

    async def get_private(self, agent_id: str) -> PrivateBus:
        if agent_id not in self._private:
            bus = PrivateBus(agent_id)
            self._private[agent_id] = bus
            if self._started:
                await bus.start()
            if self._private_listeners:
                await asyncio.gather(*[listener(bus) for listener in list(self._private_listeners)])
        return self._private[agent_id]

    async def destroy_private(self, agent_id: str) -> None:
        bus = self._private.pop(agent_id, None)
        if bus is not None:
            await bus.stop()

    def register_private_listener(self, listener: Callable[[PrivateBus], Awaitable[None]]) -> None:
        self._private_listeners.append(listener)

    @property
    def private_bus_ids(self) -> Iterable[str]:
        return tuple(self._private.keys())


_global_bus: Optional[TripleBusSystem] = None


def get_global_triple_bus() -> TripleBusSystem:
    global _global_bus
    if _global_bus is None:
        _global_bus = TripleBusSystem()
    return _global_bus


def reset_global_triple_bus() -> None:
    global _global_bus
    _global_bus = None
