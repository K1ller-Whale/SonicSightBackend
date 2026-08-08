"""Statistics helpers. Percentiles are named exactly (never 'average')."""

import numpy as np

PERCENTILES = (50, 90, 95, 99)


def stats_bundle(values):
    """Full descriptive bundle for a series: p50/p90/p95/p99/max/min/mean/n."""
    if not len(values):
        return {"count": 0}
    a = np.asarray(values, dtype=np.float64)
    out = {"count": int(a.size), "mean": float(a.mean()),
           "min": float(a.min()), "max": float(a.max())}
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def percentile(values, p):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return None
    return float(np.percentile(a, p))


def least_squares_slope(t_seconds, values):
    """Slope of values over time (units of value per minute)."""
    t = np.asarray(t_seconds, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.size < 2:
        return None
    slope_per_sec = np.polyfit(t, v, 1)[0]
    return float(slope_per_sec * 60.0)


def proportion(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


class Series:
    """Timestamped sample series (monotonic seconds, value)."""

    def __init__(self, name):
        self.name = name
        self.t = []
        self.v = []

    def add(self, t_seconds, value):
        self.t.append(float(t_seconds))
        self.v.append(float(value))

    def bundle(self):
        return stats_bundle(self.v)

    def window(self, t_from, t_to):
        pairs = [(t, v) for t, v in zip(self.t, self.v) if t_from <= t < t_to]
        return [v for _, v in pairs]

    def to_rows(self):
        return list(zip(self.t, self.v))
