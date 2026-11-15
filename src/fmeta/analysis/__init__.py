"""Analysis of funnel metadynamics runs: free-energy surfaces, dG0, convergence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..engine.grid import BiasGrid, BiasSettings
from ..engine.hills import read_colvar, read_hills, sum_hills
from ..engine.runner import load_run_metadata
from ..funnel.geometry import Funnel
from .binding import BindingEstimate, binding_free_energy, default_unbound_window
from .convergence import (count_transitions, dg_time_series, plateau_statistics,
                          summarize_replicas)
from .fes import free_energy_surface, projection_profile, reweighted_profile

KCAL = 4.184

__all__ = ["analyze_run", "analyze_replicas", "binding_free_energy", "BindingEstimate",
           "free_energy_surface", "projection_profile", "reweighted_profile",
           "dg_time_series", "count_transitions", "summarize_replicas"]

# Heuristic thresholds for calling a run converged; see ``analyze_run``.
MIN_REBINDING = 3
MAX_DRIFT = 4.184        # kJ/mol (1 kcal/mol): linear trend of dG(t) across the plateau window
MAX_ROUGHNESS = 2.0      # kJ/mol, unbound-window flatness
MAX_REWEIGHT_GAP = 4.0   # kJ/mol, bias-sum vs reweighted estimate


def _reweighted_estimate(colvar: dict, grid: BiasGrid, funnel: Funnel, kT: float,
                         fraction: float, **binding_kwargs) -> Optional[float]:
    n = colvar["s"].size
    start = int(n * (1.0 - fraction))
    edges = np.linspace(grid.s_min, grid.s_max, max(20, int((grid.s_max - grid.s_min) / 0.05)) + 1)
    profile = reweighted_profile(colvar["s"][start:], colvar["rho"][start:], grid, kT, edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    sampled = np.isfinite(profile)
    unbound = binding_kwargs.get("unbound") or default_unbound_window(funnel)
    needed = (centres >= unbound[0]) & (centres <= unbound[1])
    if sampled.sum() < 4 or not sampled[needed].all():
        return None
    # Unsampled bins hold no Boltzmann weight: keep them out of the integrals.
    profile = np.where(sampled, profile, profile[sampled].max() + 50.0 * kT)
    return binding_free_energy(centres, profile, funnel, kT, **binding_kwargs).dG


def analyze_run(workdir, n_points: int = 20, plateau_fraction: float = 0.5,
                site_max: Optional[float] = None,
                unbound: Optional[tuple[float, float]] = None, write: bool = True) -> dict:
    """Analyse one run directory and return (and by default write) the results.

    The headline number is ``dG_plateau``: the mean of dG0(t) over the last
    ``plateau_fraction`` of the run. The estimate from the final bias alone
    (``dG_final``) still fluctuates as hills are added and is noisier.
    """
    workdir = Path(workdir)
    meta = load_run_metadata(workdir)
    funnel = Funnel.from_dict(meta["funnel"])
    settings = BiasSettings(**meta["bias"])
    grid = BiasGrid(**meta["grid"])
    hills = read_hills(workdir / "HILLS")
    colvar = read_colvar(workdir / "COLVAR")
    if len(hills) == 0:
        raise ValueError(f"{workdir}/HILLS contains no hills")
    kT, scale = settings.kT, settings.fes_scale
    kwargs = dict(site_max=site_max, unbound=unbound)

    times, dg = dg_time_series(hills, grid, funnel, scale, kT,
                               every=max(1, len(hills) // n_points), **kwargs)
    final_grid = sum_hills(hills, grid)
    profile = projection_profile(final_grid, scale, kT)
    estimate = binding_free_energy(final_grid.s, profile, funnel, kT, **kwargs)
    plateau = plateau_statistics(times, dg, plateau_fraction)

    s_bound = float(final_grid.s[np.argmin(profile)])
    bound_below = min(s_bound + 0.3, estimate.site_max)
    transitions = count_transitions(colvar["s"], bound_below, estimate.unbound[0])
    dg_reweighted = _reweighted_estimate(colvar, final_grid, funnel, kT, plateau_fraction, **kwargs)

    problems = []
    if transitions.rebinding < MIN_REBINDING:
        problems.append(f"only {transitions.rebinding} rebinding event(s); need >= {MIN_REBINDING}")
    if abs(plateau["drift"]) > MAX_DRIFT:
        problems.append(f"dG still drifting ({plateau['drift']:+.2f} kJ/mol over the plateau window)")
    if estimate.unbound_roughness > MAX_ROUGHNESS:
        problems.append(f"unbound region not flat (std {estimate.unbound_roughness:.2f} kJ/mol)")
    if dg_reweighted is not None and abs(dg_reweighted - plateau["mean"]) > MAX_REWEIGHT_GAP:
        problems.append(f"reweighted estimate differs by {dg_reweighted - plateau['mean']:+.2f} kJ/mol")

    result = dict(
        workdir=str(workdir),
        simulated_ns=float(hills.time[-1] / 1000.0),
        n_hills=len(hills),
        temperature=settings.temperature,
        dG_plateau=plateau["mean"], dG_plateau_std=plateau["std"],
        dG_plateau_kcal=plateau["mean"] / KCAL,
        dG_final=estimate.dG, dG_simple=estimate.dG_simple, dG_reweighted=dg_reweighted,
        drift=plateau["drift"], k_bind_nm3=estimate.k_bind, well_depth=estimate.well_depth,
        s_bound=s_bound, site_max=estimate.site_max, unbound=list(estimate.unbound),
        unbound_roughness=estimate.unbound_roughness,
        unbinding_events=transitions.unbinding, rebinding_events=transitions.rebinding,
        last_hill_height=float(hills.height[-1]),
        converged=not problems, problems=problems,
        dG_time_ns=(times / 1000.0).tolist(), dG_series=dg.tolist(),
    )
    if write:
        with open(workdir / "result.json", "w") as handle:
            json.dump(result, handle, indent=2)
        np.savez(workdir / "fes.npz", s=final_grid.s, rho=final_grid.rho,
                 fes=free_energy_surface(final_grid, scale), profile=profile,
                 time_ns=times / 1000.0, dG=dg)
    return result


def analyze_replicas(workdirs: Sequence, write_to=None, **kwargs) -> dict:
    """Analyse independent replicas and combine their plateau estimates."""
    runs = [analyze_run(w, **kwargs) for w in workdirs]
    summary = summarize_replicas([r["dG_plateau"] for r in runs])
    result = dict(
        n_replicas=summary.n, dG=summary.mean, dG_kcal=summary.mean / KCAL,
        sem=summary.sem, ci95=None if summary.ci95 is None else list(summary.ci95),
        converged=all(r["converged"] for r in runs),
        problems={r["workdir"]: r["problems"] for r in runs if r["problems"]},
        replicas=[{k: r[k] for k in ("workdir", "simulated_ns", "dG_plateau", "dG_plateau_std",
                                      "dG_final", "rebinding_events", "converged")} for r in runs],
    )
    if write_to is not None:
        with open(write_to, "w") as handle:
            json.dump(result, handle, indent=2)
    return result
