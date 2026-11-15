"""Convergence diagnostics: dG(t), rebinding events, replica statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from ..engine.grid import BiasGrid
from ..engine.hills import Hills
from ..funnel.geometry import Funnel
from .binding import binding_free_energy
from .fes import projection_profile


def dg_time_series(hills: Hills, grid: BiasGrid, funnel: Funnel, fes_scale: float, kT: float,
                   every: int, **binding_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """dG0 from the bias accumulated up to each multiple of ``every`` hills.

    Returns ``(times_ps, dG)``; the last point always uses all hills.
    """
    work = grid.empty_copy()
    stops = list(range(every, len(hills) + 1, every))
    if not stops or stops[-1] != len(hills):
        stops.append(len(hills))
    times, values, done = [], [], 0
    for stop in stops:
        part = hills[done:stop]
        work.add_hills(part.s, part.rho, part.height)
        done = stop
        profile = projection_profile(work, fes_scale, kT)
        values.append(binding_free_energy(work.s, profile, funnel, kT, **binding_kwargs).dG)
        times.append(hills.time[stop - 1])
    return np.array(times), np.array(values)


@dataclass
class Transitions:
    unbinding: int
    rebinding: int


def count_transitions(s: np.ndarray, bound_below: float, unbound_above: float) -> Transitions:
    """Count bound -> unbound and unbound -> bound passages with hysteresis:
    the state changes only when ``s`` crosses the far threshold."""
    s = np.asarray(s, dtype=float)
    state, unbinding, rebinding = None, 0, 0
    for value in s:
        if value < bound_below:
            if state == "unbound":
                rebinding += 1
            state = "bound"
        elif value > unbound_above:
            if state == "bound":
                unbinding += 1
            state = "unbound"
    return Transitions(unbinding, rebinding)


@dataclass
class ReplicaSummary:
    n: int
    mean: float
    sem: Optional[float]
    ci95: Optional[tuple[float, float]]


def summarize_replicas(values: Sequence[float]) -> ReplicaSummary:
    """Mean over independent replicas with a Student-t 95% confidence interval."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return ReplicaSummary(int(values.size), mean, None, None)
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    half = float(stats.t.ppf(0.975, values.size - 1) * sem)
    return ReplicaSummary(int(values.size), mean, sem, (mean - half, mean + half))


def plateau_statistics(times: np.ndarray, dg: np.ndarray, fraction: float = 0.3) -> dict:
    """Mean and drift of dG(t) over the last ``fraction`` of the run."""
    n = max(2, int(round(len(dg) * fraction)))
    tail_t, tail = np.asarray(times[-n:], dtype=float), np.asarray(dg[-n:], dtype=float)
    slope = np.polyfit(tail_t, tail, 1)[0] if np.ptp(tail_t) > 0 else 0.0
    return dict(mean=float(tail.mean()), std=float(tail.std()),
                drift=float(slope * (tail_t[-1] - tail_t[0])))
