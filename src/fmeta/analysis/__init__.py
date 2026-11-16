"""Analysis of funnel metadynamics runs: free-energy surfaces, dG0, convergence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..engine.grid import BiasGrid, BiasSettings
from ..engine.hills import Hills, merge_hills, read_colvar, read_hills
from ..engine.runner import load_run_metadata
from ..funnel.geometry import Funnel
from .binding import BindingEstimate, binding_free_energy, default_unbound_window, empty_funnel_free_energy
from .convergence import count_transitions, dg_time_series, plateau_statistics, summarize_replicas
from .fes import free_energy_surface, projection_profile, reweighted_profile, sliced_log_weights

KCAL = 4.184

__all__ = ["analyze_run", "analyze_replicas", "walker_directories", "binding_free_energy", "BindingEstimate",
           "free_energy_surface", "projection_profile", "reweighted_profile", "sliced_log_weights",
           "dg_time_series", "count_transitions", "summarize_replicas"]

# Heuristic thresholds for calling a run converged; see ``analyze_run``.
MIN_REBINDING = 3
MAX_DRIFT = 4.184        # kJ/mol (1 kcal/mol): linear trend of dG(t) across the plateau window
MAX_ROUGHNESS = 2.0      # kJ/mol, unbound-window flatness
MAX_REWEIGHT_GAP = 4.0   # kJ/mol, between two estimates of the same run
MIN_BINDING_EXCESS = 1.0  # kT by which dG must beat an empty funnel of the same shape

# The reweighting treats the run as a sequence of time slices, each with the bias of its middle.
SLICE_PS = 250.0         # width of a slice; wider when a run would need more than MAX_SLICES
MAX_SLICES = 300
MAX_SAMPLES = 40000      # more samples than this are thinned evenly


def _sample_estimate(colvars: list, hills: Hills, grid: BiasGrid, funnel: Funnel, kT: float,
                     start: float, **binding_kwargs) -> Optional[float]:
    """dG0 from the samples after ``start`` (ps) alone, or None when they cannot give one.

    The time after ``start`` is cut into slices. Within a slice the bias is taken as constant
    (the hills up to its middle), which makes the run a set of ensembles with known biases;
    :func:`sliced_log_weights` then gives every sample its weight in the unbiased ensemble and
    the weighted histogram along s gives W(s). The hills before ``start`` enter only as part
    of the bias the later samples felt: nothing assumes that they had filled anything."""
    end = max((float(colvar["time"][-1]) for colvar in colvars if colvar["time"].size), default=0.0)
    s, rho, time = (np.concatenate([colvar[key][colvar["time"] > start] for colvar in colvars])
                    for key in ("s", "rho", "time"))
    if s.size < 100:
        return None
    thin = -(-s.size // MAX_SAMPLES)
    s, rho, time = s[::thin], rho[::thin], time[::thin]
    n_slices = int(np.clip(np.ceil((end - start) / SLICE_PS), 1, MAX_SLICES))
    edges = np.linspace(start, end, n_slices + 1)
    counts = np.bincount(np.clip(np.searchsorted(edges, time, side="left") - 1, 0, n_slices - 1),
                         minlength=n_slices)
    work, done, reduced = grid.empty_copy(), 0, []
    for middle, count in zip(0.5 * (edges[1:] + edges[:-1]), counts):
        stop = int(np.searchsorted(hills.time, middle, side="right"))
        if stop > done:
            part = hills[done:stop]
            work.add_hills(part.s, part.rho, part.height)
            done = stop
        if count:
            reduced.append(work.interpolate(s, rho) / kT)
    try:
        log_weights = sliced_log_weights(np.array(reduced), counts[counts > 0])
    except ArithmeticError:
        return None

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
                discard_ns: float = 2.0, site_max: Optional[float] = None,
                unbound: Optional[tuple[float, float]] = None, write: bool = True) -> dict:
    """Analyse one run directory and return (and by default write) the results.

    The headline number ``dG`` uses only the samples after the first ``discard_ns``, the part
    of a run in which the bias changes most (of a run shorter than twice that: its second
    half). It is ``dG_reweighted``, the estimate from the reweighted histogram of those
    samples. Two cross-checks stand beside it: ``dG_second_half``, the same estimate from the
    second half of the run alone, and ``dG_plateau``, the mean over the last
    ``plateau_fraction`` of the run of dG0(t) from the bias accumulated since the start.
    Where the samples give no estimate, ``dG`` is ``dG_plateau`` and the run is not converged.
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

    times, dg = dg_time_series(hills, grid, funnel, scale, kT,
                               max(1, round(len(hills) / n_points)), **kwargs)
    final_grid = grid.empty_copy()
    final_grid.add_hills(hills.s, hills.rho, hills.height)
    profile = projection_profile(final_grid, scale, kT)
    estimate = binding_free_energy(final_grid.s, profile, funnel, kT, **kwargs)
    plateau = plateau_statistics(times, dg, plateau_fraction)

    s_bound = float(final_grid.s[np.argmin(profile)])
    bound_below = min(s_bound + 0.3, estimate.site_max)
    counted = [count_transitions(colvar["s"], bound_below, estimate.unbound[0])
               for colvar in colvars]
    unbinding = sum(c.unbinding for c in counted)
    rebinding = sum(c.rebinding for c in counted)

    half = 0.5 * max((float(colvar["time"][-1]) for colvar in colvars if colvar["time"].size), default=0.0)
    start = min(1000.0 * discard_ns, half)
    dg_reweighted = _sample_estimate(colvars, hills, grid, funnel, kT, start, **kwargs)
    dg_second_half = (dg_reweighted if start == half
                      else _sample_estimate(colvars, hills, grid, funnel, kT, half, **kwargs))
    headline = plateau["mean"] if dg_reweighted is None else dg_reweighted

    # Does the answer depend on where the bound region is said to end? For a real
    # binder the basin dominates the integral and it does not.
    shifted = [binding_free_energy(final_grid.s, profile, funnel, kT,
                                   site_max=estimate.site_max + delta, unbound=unbound).dG
               for delta in (-0.2, 0.2)]
    site_sensitivity = float(max(abs(value - estimate.dG) for value in shifted))
    # What would the same analysis report if nothing bound at all? Then K_b is just
    # the volume of funnel that was labelled "bound", and dG is set by that label.
    dg_empty = empty_funnel_free_energy(final_grid.s, funnel, kT, estimate.site_max)
    binding_excess = dg_empty - headline

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
    if dg_reweighted is None:
        problems.append(f"no estimate from the samples after the first {start / 1000.0:.1f} ns (they do "
                        "not cover the unbound region): dG is the mean of dG(t) from the bias")
    else:
        if abs(dg_reweighted - plateau["mean"]) > MAX_REWEIGHT_GAP:
            problems.append(f"reweighted estimate differs by {dg_reweighted - plateau['mean']:+.2f} kJ/mol "
                            "from the mean of dG(t) from the bias")
        if dg_second_half is not None and abs(dg_second_half - dg_reweighted) > MAX_REWEIGHT_GAP:
            problems.append(f"dG changes by {dg_second_half - dg_reweighted:+.2f} kJ/mol when only the "
                            "second half of the run is used")

    result = dict(
        workdir=str(workdir),
        simulated_ns=float(sum(c["time"][-1] for c in colvars if c["time"].size) / 1000.0),
        n_walkers=len(sources),
        n_hills=len(hills),
        temperature=settings.temperature,
        dG=float(headline), dG_kcal=float(headline) / KCAL,
        dG_source="bias" if dg_reweighted is None else "samples", discard_ns=start / 1000.0,
        dG_reweighted=dg_reweighted, dG_second_half=dg_second_half,
        dG_plateau=plateau["mean"], dG_plateau_std=plateau["std"],
        dG_final=estimate.dG, dG_simple=estimate.dG_simple,
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
    """Analyse independent replicas and combine their estimates."""
    runs = [analyze_run(w, **kwargs) for w in workdirs]
    summary = summarize_replicas([r["dG"] for r in runs])
    result = dict(
        n_replicas=summary.n, dG=summary.mean, dG_kcal=summary.mean / KCAL,
        sem=summary.sem, ci95=None if summary.ci95 is None else list(summary.ci95),
        converged=all(r["converged"] for r in runs),
        problems={r["workdir"]: r["problems"] for r in runs if r["problems"]},
        replicas=[{k: r[k] for k in ("workdir", "simulated_ns", "dG", "dG_source", "dG_reweighted",
                                      "dG_plateau", "dG_plateau_std", "dG_final", "rebinding_events",
                                      "converged")}
                  for r in runs],
    )
    if write_to is not None:
        with open(write_to, "w") as handle:
            json.dump(result, handle, indent=2)
    return result
