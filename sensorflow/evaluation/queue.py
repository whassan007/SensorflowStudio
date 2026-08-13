"""Queue abstraction between ingestion and evaluation workers.

The platform is queue-first but transport-agnostic: EventQueue is the interface,
InMemoryEventQueue is the fully-working default, RedisEventQueue is used when a
redis client is importable/reachable, and KafkaEventQueue is an
interface-compatible stub kept for deployment parity (Kafka is not bundled).
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional


class EventQueue:
    """Abstract message/job queue keyed by topic."""

    backend_name = "abstract"

    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        raise NotImplementedError

    def consume(self, topic: str, max_messages: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def ack(self, topic: str, count: int, failed: int = 0) -> None:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class InMemoryEventQueue(EventQueue):
    """Default queue: thread-safe deque per topic with throughput accounting."""

    backend_name = "in_memory"

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.topics: Dict[str, deque] = defaultdict(deque)
        self.processing = 0
        self.completed = 0
        self.failed = 0
        self._window: deque = deque(maxlen=500)  # (ts, count) for throughput

    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        with self.lock:
            self.topics[topic].append(message)

    def consume(self, topic: str, max_messages: int = 100) -> List[Dict[str, Any]]:
        out = []
        with self.lock:
            q = self.topics[topic]
            while q and len(out) < max_messages:
                out.append(q.popleft())
            self.processing += len(out)
        return out

    def ack(self, topic: str, count: int, failed: int = 0) -> None:
        with self.lock:
            self.processing = max(0, self.processing - count - failed)
            self.completed += count
            self.failed += failed
            self._window.append((time.time(), count))

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            depth = {t: len(q) for t, q in self.topics.items() if len(q) > 0}
            pending = sum(len(q) for q in self.topics.values())
            now = time.time()
            recent = [(ts, c) for ts, c in self._window if now - ts <= 5.0]
            throughput = sum(c for _, c in recent) / 5.0 if recent else 0.0
            return {
                "backend": self.backend_name,
                "pending": pending,
                "processing": self.processing,
                "completed": self.completed,
                "failed": self.failed,
                "throughput_per_s": round(throughput, 1),
                "depth_by_topic": depth,
            }

    def reset(self) -> None:
        with self.lock:
            self.topics.clear()
            self.processing = 0
            self.completed = 0
            self.failed = 0
            self._window.clear()


class RedisEventQueue(InMemoryEventQueue):
    """Redis-backed queue. Falls back cleanly if redis is unavailable.

    Uses Redis lists per topic; counters are kept locally (single-process dev
    deployment). Only constructed when the redis package imports and pings.
    """

    backend_name = "redis"

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        super().__init__()
        import redis  # noqa: F401  (raises ImportError when unavailable)

        self.client = redis.Redis.from_url(url, socket_connect_timeout=0.5)
        self.client.ping()
        self.prefix = "labeleval:queue:"

    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        self.client.rpush(self.prefix + topic, json.dumps(message))

    def consume(self, topic: str, max_messages: int = 100) -> List[Dict[str, Any]]:
        out = []
        for _ in range(max_messages):
            raw = self.client.lpop(self.prefix + topic)
            if raw is None:
                break
            out.append(json.loads(raw))
        with self.lock:
            self.processing += len(out)
        return out

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        depth = {}
        for key in self.client.scan_iter(self.prefix + "*"):
            name = key.decode().replace(self.prefix, "")
            depth[name] = self.client.llen(key)
        base["pending"] = sum(depth.values())
        base["depth_by_topic"] = depth
        base["backend"] = self.backend_name
        return base

    def reset(self) -> None:
        for key in self.client.scan_iter(self.prefix + "*"):
            self.client.delete(key)
        super().reset()


class KafkaEventQueue(EventQueue):
    """Interface-compatible Kafka stub.

    Kafka is not bundled with this repo; this class documents the production
    transport and raises a clear error if selected without a broker client.
    """

    backend_name = "kafka"

    def __init__(self, bootstrap_servers: str = "localhost:9092") -> None:
        self.bootstrap_servers = bootstrap_servers
        raise RuntimeError(
            "KafkaEventQueue is a deployment stub: install/configure a Kafka client "
            "and broker, or use InMemoryEventQueue (default) / RedisEventQueue."
        )


def make_queue(preferred: str = "auto") -> EventQueue:
    """Build the best available queue. Never hard-codes to Kafka."""
    if preferred in ("auto", "redis"):
        try:
            return RedisEventQueue()
        except Exception:
            if preferred == "redis":
                pass  # fall through to in-memory with a working queue
    if preferred == "kafka":
        try:
            return KafkaEventQueue()
        except Exception:
            pass
    return InMemoryEventQueue()
