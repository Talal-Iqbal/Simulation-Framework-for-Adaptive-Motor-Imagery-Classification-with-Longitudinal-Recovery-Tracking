"""In-process counters exposed at /metrics in Prometheus text format."""

from __future__ import annotations

import threading
from collections import defaultdict


class MetricsCollector:
    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._counters[name] = value

    def render_prometheus(self) -> str:
        lines = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


_GLOBAL_METRICS = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _GLOBAL_METRICS
