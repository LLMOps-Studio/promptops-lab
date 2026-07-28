import math
from collections.abc import Sequence

from pydantic import BaseModel


class LatencyStats(BaseModel):
    """p50/p95/p99 latency (seconds) over a set of timed requests."""

    p50: float
    p95: float
    p99: float
    n: int


def _percentile(sorted_samples: list, pct: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    k = (len(sorted_samples) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_samples[int(k)]
    return sorted_samples[f] * (c - k) + sorted_samples[c] * (k - f)


def compute_latency_stats(samples_seconds: Sequence[float]) -> LatencyStats:
    """
    p50/p95/p99 latency from a list of per-request timings, in seconds.

    Reports percentiles rather than a mean -- LLM latency is long-tailed,
    and a handful of very slow requests can be invisible in an average
    while showing up clearly at p95/p99.

    Raises:
        ValueError: if samples_seconds is empty.
    """
    if not samples_seconds:
        raise ValueError("samples_seconds must not be empty")
    sorted_samples = sorted(samples_seconds)
    return LatencyStats(
        p50=_percentile(sorted_samples, 0.50),
        p95=_percentile(sorted_samples, 0.95),
        p99=_percentile(sorted_samples, 0.99),
        n=len(sorted_samples),
    )
