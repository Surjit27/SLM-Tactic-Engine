"""Latency measurement helpers for prototype inference pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LatencyRecorder:
    samples: dict[str, list[float]] = field(default_factory=dict)

    def start(self) -> float:
        return time.perf_counter()

    def stop(self, name: str, start_time: float) -> None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.samples.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, values in self.samples.items():
            arr = np.array(values, dtype=np.float64)
            out[key] = {
                "count": float(len(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "mean_ms": float(np.mean(arr)),
            }
        return out
