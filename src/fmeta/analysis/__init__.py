"""Analysis of funnel metadynamics runs: free-energy surfaces, dG0, convergence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..engine.grid import BiasGrid, BiasSettings
from ..engine.hills import merge_hills, read_colvar, read_hills
from ..engine.runner import load_run_metadata
from ..funnel.geometry import Funnel
from .binding import BindingEstimate, binding_free_energy, default_unbound_window, empty_funnel_free_energy
from .convergence import count_transitions, dg_time_series, plateau_statistics, scan_hills, summarize_replicas
from .fes import free_energy_surface, projection_profile, reweighted_profile, reweighting_offset

KCAL = 4.184

__all__ = ["analyze_run", "analyze_replicas", "walker_directories", "binding_free_energy", "BindingEstimate",
           "free_energy_surface", "projection_profile", "reweighted_profile", "reweighting_offset",
           "dg_time_series", "count_transitions", "summarize_replicas"]

# Heuristic thresholds for calling a run converged; see ``analyze_run``.
MIN_REBINDING = 3
MAX_DRIFT = 4.184        # kJ/mol (1 kcal/mol): linear trend of dG(t) across the plateau window
MAX_ROUGHNESS = 2.0      # kJ/mol, unbound-window flatness
MAX_REWEIGHT_GAP = 4.0   # kJ/mol, bias-sum vs reweighted estimate
MIN_BINDING_EXCESS = 1.0  # kT by which dG must beat an empty funnel of the same shape


def _reweighted_estimate(colvars: list, offset_times: np.ndarray, offsets: np.ndarray,
                         grid: BiasGrid, funnel: Funnel, kT: float, discard: float,
                         **binding_kwargs) -> Optional[float]:
    """dG0 from the reweighted histogram of the trajectory itself. It shares the
    trajectory with the bias-sum estimate but not the route from data to W(s)."""
    start = discard * max(colvar["time"][-1] for colvar in colvars if colvar["time"].size)
    keep = [colvar["time"] > start for colvar in colvars]
    s, time, bias = (np.concatenate([colvar[key][mask] for colvar, mask in zip(colvars, keep)])
                     for key in ("s", "time", "bias"))
    if s.size < 100:
        return None
    log_weights = (bias - np.interp(time, offset_times, offsets)) / kT
    edges = np.linspace(grid.s_min, grid.s_max, max(20, int((grid.s_max - grid.s_min) / 0.05)) + 1)
    profile = reweighted_profile(s, log_weights, kT, edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    sampled = np.isfinite(profile)
    unbound = binding_kwargs.get("unbound") or default_unbound_window(funnel)
    needed = (centres >= unbound[0]) & (centres <= unbound[1])
    if sampled.sum() < 4 or not sampled[needed].all():
        return None
    # Unsampled bins hold no Boltzmann weight: keep them out of the integrals.
    profile = np.where(sampled, profile, profile[sampled].max() + 50.0 * kT)
    return binding_free_energy(centres, profile, funnel, kT, **binding_kwargs).dG


def walker_directories(workdir) -> list:
    """Sub-directories ``w0, w1, ...`` of a run that used multiple walkers."""
    workdir = Path(workdir)
    found = [d for d in workdir.glob("w*") if d.name[1:].isdigit() and (d / "HILLS").exists()]
    return sorted(found, key=lambda d: int(d.name[1:]))


def analyze_run(workdir, n_points: int = 20, plateau_fraction: float = 0.5,
                reweight_discard: float = 0.2, site_max: Optional[float] = None,
                unbound: Optional[tuple[float, float]] = None, write: bool = True) -> dict:
    """Analyse one run directory and return (and by default write) the results.

    The headline number is ``dG_plateau``: the mean of dG0(t) over the last
    ``plateau_fraction`` of the run. The estimate from the final bias alone
    (``dG_final``) still fluctuates as hills are added and is noisier.
    ``dG_reweighted`` comes from the reweighted histogram of the trajectory
    (skipping the first ``reweight_discard`` of it) and serves as a cross-check.
    """
    workdir = Path(workdir)
    # With multiple walkers the bias is the sum over all of them, ordered in time.
    sources = walker_directories(workdir) or [workdir]
    meta = load_run_metadata(sources[0])
    funnel = Funnel.from_dict(meta["funnel"])
    settings = BiasSettings(**meta["bias"])
    grid = BiasGrid(**meta["grid"])
    hills = merge_hills([read_hills(d / "HILLS") for d in sources])
    colvars = [read_colvar(d / "COLVAR") for d in sources]
    if len(hills) == 0:
        raise ValueError(f"{workdir}/HILLS contains no hills")
    kT, scale = settings.kT, settings.fes_scale
    kwargs = dict(site_max=site_max, unbound=unbound)

    # One pass over the hills: dG(t) at n_points times, c(t) ten times as often.
    fine = max(1, len(hills) // (10 * n_points))
    coarse = max(1, len(hills) // n_points)
    times, dg, offset_times, offsets = [], [], [0.0], [0.0]
    for n, time_ps, work in scan_hills(hills, grid, fine):
        offset_times.append(time_ps)
        offsets.append(reweighting_offset(work, scale, kT))
        if n % coarse == 0 or n == len(hills):
            times.append(time_ps)
            dg.append(binding_free_energy(work.s, projection_profile(work, scale, kT),
                                          funnel, kT, **kwargs).dG)
    times, dg = np.array(times), np.array(dg)
    final_grid = work
    profile = projection_profile(final_grid, scale, kT)
    estimate = binding_free_energy(final_grid.s, profile, funnel, kT, **kwargs)
    plateau = plateau_statistics(times, dg, plateau_fraction)

    s_bound = float(final_grid.s[np.argmin(profile)])
    bound_below = min(s_bound + 0.3, estimate.site_max)
    counted = [count_transitions(colvar["s"], bound_below, estimate.unbound[0])
               for colvar in colvars]
    unbinding = sum(c.unbinding for c in counted)
    rebinding = sum(c.rebinding for c in counted)
    dg_reweighted = _reweighted_estimate(colvars, np.array(offset_times), np.array(offsets),
                                         final_grid, funnel, kT, reweight_discard, **kwargs)

    # Does the answer depend on where the bound region is said to end? For a real
    # binder the basin dominates the integral and it does not.
    shifted = [binding_free_energy(final_grid.s, profile, funnel, kT,
                                   site_max=estimate.site_max + delta, unbound=unbound).dG
               for delta in (-0.2, 0.2)]
    site_sensitivity = float(max(abs(value - estimate.dG) for value in shifted))
    # What would the same analysis report if nothing bound at all? Then K_b is just
    # the volume of funnel that was labelled "bound", and dG is set by that label.
    dg_empty = empty_funnel_free_energy(final_grid.s, funnel, kT, estimate.site_max)
    binding_excess = dg_empty - plateau["mean"]

    problems = []
    if binding_excess < MIN_BINDING_EXCESS * kT:
        problems.append(f"dG is within {MIN_BINDING_EXCESS:.0f} kT of what an empty funnel gives "
                        f"({dg_empty:.2f} kJ/mol): no binding beyond the funnel volume, and the "
                        "value depends on where the bound region is said to end")
    if rebinding < MIN_REBINDING:
        problems.append(f"only {rebinding} rebinding event(s); need >= {MIN_REBINDING}")
    if abs(plateau["drift"]) > MAX_DRIFT:
        problems.append(f"dG still drifting ({plateau['drift']:+.2f} kJ/mol over the plateau window)")
    if estimate.unbound_roughness > MAX_ROUGHNESS:
        problems.append(f"unbound region not flat (std {estimate.unbound_roughness:.2f} kJ/mol)")
    if dg_reweighted is not None and abs(dg_reweighted - plateau["mean"]) > MAX_REWEIGHT_GAP:
        problems.append(f"reweighted estimate differs by {dg_reweighted - plateau['mean']:+.2f} kJ/mol")

    result = dict(
        workdir=str(workdir),
        simulated_ns=float(sum(c["time"][-1] for c in colvars if c["time"].size) / 1000.0),
        n_walkers=len(sources),
        n_hills=len(hills),
        temperature=settings.temperature,
        dG_plateau=plateau["mean"], dG_plateau_std=plateau["std"],
        dG_plateau_kcal=plateau["mean"] / KCAL,
        dG_final=estimate.dG, dG_simple=estimate.dG_simple, dG_reweighted=dg_reweighted,
        drift=plateau["drift"], k_bind_nm3=estimate.k_bind, well_depth=estimate.well_depth,
        s_bound=s_bound, site_max=estimate.site_max, unbound=list(estimate.unbound),
        unbound_roughness=estimate.unbound_roughness, site_sensitivity=site_sensitivity,
        dG_empty_funnel=dg_empty, binding_excess=float(binding_excess),
        unbinding_events=unbinding, rebinding_events=rebinding,
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
