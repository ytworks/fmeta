"""Analysis of funnel metadynamics runs: free-energy surfaces, dG0 of a set of replicas, diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..engine.grid import BiasGrid, BiasSettings
from ..engine.hills import Hills, merge_hills, read_colvar, read_hills
from ..engine.runner import load_run_metadata
from ..funnel.geometry import KB, Funnel
from .binding import BindingEstimate, binding_free_energy, default_unbound_window, empty_funnel_free_energy
from .convergence import count_transitions, dg_time_series, plateau_statistics, summarize_replicas
from .fes import free_energy_surface, projection_profile, reweighted_profile, sliced_log_weights
from .walkers import SharedBias, hills_on_grid

KCAL = 4.184

__all__ = ["analyze_run", "analyze_replicas", "walker_directories", "binding_free_energy", "BindingEstimate",
           "free_energy_surface", "projection_profile", "reweighted_profile", "sliced_log_weights",
           "dg_time_series", "count_transitions", "summarize_replicas"]

# Thresholds of the diagnostics of a replica: notes beside its value, never a verdict (see
# ``analyze_run``). Only MIN_BINDING_EXCESS decides something: what is reported as no binding.
MIN_REBINDING = 3
MAX_DRIFT = 4.184        # kJ/mol (1 kcal/mol): change of dG(t) from the bias over the last part of a run
MAX_ROUGHNESS = 2.0      # kJ/mol, unbound-window flatness
MAX_REWEIGHT_GAP = 4.0   # kJ/mol, between two estimates of the same run
MIN_BINDING_EXCESS = 1.0  # kT by which dG must beat an empty funnel of the same shape

# The reweighting treats each walker of a run as a sequence of time slices, each with the bias
# that walker had at its middle.
SLICE_PS = 250.0         # width of a slice; wider when a run would need more than MAX_SLICES
MAX_SLICES = 300         # of all walkers together
MAX_SAMPLES = 40000      # more samples than this are thinned evenly


def _cycles(colvar: dict, own: Hills) -> np.ndarray:
    """The number of its walker's own hills that each sample felt (a sample taken at the time
    of a hill precedes it)."""
    return np.searchsorted(own.time, colvar["time"], side="left")


def _slice_samples(colvars: list, parts: list, on_grid: list, start: float):
    """Select samples and the SYNC-confirmed bias at the midpoint of each time slice."""
    samples, biases = [], []         # the samples of each walker; (walker, hills on its grid) of each slice
    for walker, (colvar, own, table) in enumerate(zip(colvars, parts, on_grid)):
        time = colvar["time"]
        if not time.size:
            continue
        n_slices = int(np.clip(np.ceil((time[-1] - start) / SLICE_PS), 1, max(1, MAX_SLICES // len(colvars))))
        edges = np.linspace(start, time[-1], n_slices + 1)
        on = table[np.searchsorted(own.time, 0.5 * (edges[1:] + edges[:-1]), side="right")]
        index = np.clip(np.searchsorted(edges, time, side="left") - 1, 0, n_slices - 1)
        keep = (time > start) & (table[_cycles(colvar, own)] >= 0).all(axis=1) & (on[index] >= 0).all(axis=1)
        samples.append((colvar["s"][keep], colvar["rho"][keep], len(biases) + index[keep]))
        biases += [(walker, counts) for counts in on]
    if not samples:
        return None
    s, rho, ensemble = (np.concatenate([sample[k] for sample in samples]) for k in range(3))
    if s.size < 100:
        return None
    thin = -(-s.size // MAX_SAMPLES)
    s, rho, ensemble = s[::thin], rho[::thin], ensemble[::thin]
    counts = np.bincount(ensemble, minlength=len(biases))
    return s, rho, biases, counts


def _sample_estimate(colvars: list, parts: list, on_grid: list, grid: BiasGrid, funnel: Funnel, kT: float,
                     start: float, **binding_kwargs) -> Optional[float]:
    """Reweight samples from the bias each walker felt. Hills before the cutoff enter only as
    part of the subsequent walker-specific biases, reconstructed from their SYNC histories."""
    selected = _slice_samples(colvars, parts, on_grid, start)
    if selected is None:
        return None
    s, rho, biases, counts = selected
    shared, current, reduced = None, None, []
    for (walker, on), count in zip(biases, counts):
        if walker != current:
            shared, current = SharedBias(grid, parts), walker
        if (on >= 0).all():
            work = shared.at(on)
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

    The framework headline ``dG`` uses the accumulated WT bias times ``-gamma / (gamma - 1)``
    to estimate a free-energy profile. Integrating its binding region relative to bulk gives
    dG0(t). The retained fraction (one half by default) and five-block error are choices of
    fmeta. Limongelli, Bonomi and Parrinello (PNAS 2013, Fig. S3) used trajectory reweighting
    and reported a standard deviation. Here ``dG_plateau`` is the mean, ``dG_error`` its block
    standard error, and ``dG_plateau_std`` the standard deviation over the retained fraction.

    Two cross-checks from the samples stand beside it: ``dG_reweighted``, from the reweighted
    histogram of the samples after the first ``discard_ns`` (of a run shorter than twice that:
    its second half), and ``dG_second_half``, the same from the second half of the run alone.
    A sample of a walker is weighted with the bias that walker had: its own hills and the hills
    its ``SYNC`` record says it had read from the others. Samples whose bias cannot be rebuilt
    are left out, and walkers without a record get neither cross-check.

    ``diagnostics`` are notes on what the replica did (rebinding events, the change of dG(t)
    from the bias, the two cross-checks). The bias fills the free-energy surface for as long as
    the run lasts, so dG(t), the difference between two regions of that surface, moves by
    principle: the notes say how far the filling went and decide nothing. ``no_binding`` says
    that the value is within 1 kT of what the empty funnel gives, and then it is not a binding
    free energy.
    """
    workdir = Path(workdir)
    # With multiple walkers the bias is the sum over all of them, ordered in time.
    sources = walker_directories(workdir) or [workdir]
    meta = load_run_metadata(sources[0])
    funnel = Funnel.from_dict(meta["funnel"])
    settings = BiasSettings(**meta["bias"])
    grid = BiasGrid(**meta["grid"])
    parts = [read_hills(d / "HILLS") for d in sources]
    hills = merge_hills(parts)
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
    # A sample is weighted with the bias its own walker had: walkers read each other's hills as
    # wall-clock time allows, so the hills of all of them up to a time are not that bias.
    on_grid = hills_on_grid(sources, parts)
    dg_reweighted = dg_second_half = None
    unknown = 0
    if on_grid is not None:
        dg_reweighted = _sample_estimate(colvars, parts, on_grid, grid, funnel, kT, start, **kwargs)
        dg_second_half = (dg_reweighted if start == half
                          else _sample_estimate(colvars, parts, on_grid, grid, funnel, kT, half, **kwargs))
        unknown = sum(int((table[_cycles(colvar, own)] < 0).any(axis=1).sum())
                      for colvar, own, table in zip(colvars, parts, on_grid))
    headline = plateau["mean"]

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

    no_binding = bool(binding_excess < MIN_BINDING_EXCESS * kT)
    notes = []
    if no_binding:
        notes.append(f"dG is within {MIN_BINDING_EXCESS:.0f} kT of what an empty funnel gives "
                        f"({dg_empty:.2f} kJ/mol): no binding beyond the funnel volume, and the "
                        "value depends on where the bound region is said to end")
    if rebinding < MIN_REBINDING:
        notes.append(f"{rebinding} rebinding event(s), fewer than {MIN_REBINDING}")
    if abs(plateau["drift"]) > MAX_DRIFT:
        notes.append(f"dG(t) from the bias changed by {plateau['drift']:+.2f} kJ/mol over the last "
                     f"{100 * plateau_fraction:.0f}% of the run")
    if estimate.unbound_roughness > MAX_ROUGHNESS:
        notes.append(f"unbound region not flat (std {estimate.unbound_roughness:.2f} kJ/mol)")
    if on_grid is None:
        notes.append("the walkers kept no record of the hills they had read from each other (a run from "
                     "before the record existed): the bias a sample felt cannot be rebuilt, and dG has no "
                     "cross-check from the samples")
    elif dg_reweighted is None:
        notes.append(f"no estimate from the samples after the first {start / 1000.0:.1f} ns (they do "
                     "not cover the unbound region): dG has no cross-check from the samples")
    if unknown:
        notes.append(f"{unknown} sample(s) are left out of the estimates from the samples: the bias they felt "
                     "cannot be rebuilt (a restart removed hills that their walker had read from another, or "
                     "they were read before the record began)")
    checks = [(f"after the first {start / 1000.0:.1f} ns", dg_reweighted)]
    if start < half:
        checks.append(("of the second half of the run", dg_second_half))
    for label, value in checks:
        if value is not None and abs(value - headline) > MAX_REWEIGHT_GAP:
            notes.append(f"the estimate from the samples {label} differs by {value - headline:+.2f} "
                         "kJ/mol from dG")

    result = dict(
        workdir=str(workdir),
        simulated_ns=float(sum(c["time"][-1] for c in colvars if c["time"].size) / 1000.0),
        n_walkers=len(sources),
        n_hills=len(hills),
        temperature=settings.temperature,
        dG=float(headline), dG_kcal=float(headline) / KCAL, dG_error=plateau["error"],
        dG_plateau=plateau["mean"], dG_plateau_std=plateau["std"],
        discard_ns=start / 1000.0, dG_reweighted=dg_reweighted, dG_second_half=dg_second_half,
        dG_final=estimate.dG, dG_simple=estimate.dG_simple,
        drift=plateau["drift"], k_bind_nm3=estimate.k_bind, well_depth=estimate.well_depth,
        s_bound=s_bound, site_max=estimate.site_max, unbound=list(estimate.unbound),
        unbound_roughness=estimate.unbound_roughness, site_sensitivity=site_sensitivity,
        dG_empty_funnel=dg_empty, binding_excess=float(binding_excess),
        unbinding_events=unbinding, rebinding_events=rebinding,
        last_hill_height=float(hills.height[-1]),
        no_binding=no_binding, diagnostics=notes,
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
    """Analyse independent replicas as one measurement: its value ``dG`` is the mean of their
    dG0, its uncertainty the standard error ``sem`` and the 95% Student-t interval ``ci95`` of
    that mean (None for one replica).

    A replica of the default length does not converge on its own (owner, 2026-11-16; FM-D9,
    second amendment), so nothing here passes or fails. The scatter between the replicas is
    the measure of what the run length leaves open, and ``diagnostics`` holds the notes on
    each replica. ``no_binding`` says that the value of the set is within 1 kT of what the
    empty funnel gives: then it is not a binding free energy."""
    runs = [analyze_run(w, **kwargs) for w in workdirs]
    summary = summarize_replicas([r["dG"] for r in runs])
    dg_empty = float(np.mean([r["dG_empty_funnel"] for r in runs]))
    kT = KB * runs[0]["temperature"]
    result = dict(
        n_replicas=summary.n, dG=summary.mean, dG_kcal=summary.mean / KCAL,
        sem=summary.sem, ci95=None if summary.ci95 is None else list(summary.ci95),
        dG_empty_funnel=dg_empty, no_binding=bool(dg_empty - summary.mean < MIN_BINDING_EXCESS * kT),
        diagnostics={r["workdir"]: r["diagnostics"] for r in runs if r["diagnostics"]},
        replicas=[{k: r[k] for k in ("workdir", "simulated_ns", "dG", "dG_error", "dG_plateau_std",
                                      "dG_reweighted", "dG_second_half", "dG_final",
                                      "rebinding_events", "no_binding")}
                  for r in runs],
    )
    if write_to is not None:
        with open(write_to, "w") as handle:
            json.dump(result, handle, indent=2)
    return result
