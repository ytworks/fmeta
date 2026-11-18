"""Explicit-region, standard-state binding measurements from walker-reweighted samples.

Each estimate compares the *whole* binding basin with a bulk number density. With an
effective area A(s), the latter is sum_U[w/A(s)] divided by the length of U. A separate
measurement of the bulk density counts only samples inside a wall-free disk and divides by
the volume of that disk over U. Both routes use exactly the same bound population and 1 M
standard state. They share a trajectory and are not independent uncertainty estimates.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from ..engine.grid import BiasGrid, BiasSettings
from ..engine.hills import read_colvar, read_hills
from ..engine.runner import load_run_metadata
from ..funnel.geometry import KB, STANDARD_VOLUME, Funnel
from . import KCAL, MIN_BINDING_EXCESS, _cycles, _slice_samples, walker_directories
from .convergence import plateau_statistics, summarize_replicas
from .fes import projection_profile, reweighted_profile, sliced_log_weights
from .walkers import SharedBias, hills_on_grid


def _region_grid(s, bounds):
    """Integrate to the prescribed ends even when they fall between grid nodes."""
    lo, hi = bounds
    return np.r_[lo, s[(s > lo) & (s < hi)], hi]


def profile_binding(s, profile, funnel, kT, site, unbound):
    """Soft-wall generalization of the FMAP integral on a 1D PMF (kJ/mol)."""
    s, profile = np.asarray(s), np.asarray(profile)
    if (s.ndim != 1 or s.size < 2 or profile.shape != s.shape or not np.all(np.diff(s) > 0) or
            not (s[0] <= site[0] < site[1] <= unbound[0] < unbound[1] <= s[-1])):
        raise ValueError("the PMF grid must span the two ordered finite integration windows")
    site_s, bulk_s = (_region_grid(s, bounds) for bounds in (site, unbound))

    def log_integral(x, log_values):
        peak = np.max(log_values)
        return peak + np.log(np.trapezoid(np.exp(log_values - peak), x))

    site_log = -np.interp(site_s, s, profile) / kT
    bulk_log = -np.interp(bulk_s, s, profile) / kT - np.log(funnel.effective_area(bulk_s, kT))
    log_density = log_integral(bulk_s, bulk_log) - np.log(np.ptp(bulk_s))
    return float(-kT * (log_integral(site_s, site_log) - log_density - np.log(STANDARD_VOLUME)))


def _weights_on_slices(s, rho, biases, counts, parts, grid, kT, n_fit):
    """Build each source's successive prefix sums once, even when eight walkers share hills."""
    occupied = np.flatnonzero(counts)
    prefixes = np.array([biases[i][1] for i in occupied], dtype=int)
    reduced = np.zeros((len(occupied), s.size))
    for walker, hills in enumerate(parts):
        seen = prefixes[:, walker]
        order = np.argsort(seen, kind="stable")
        work, done = grid.empty_copy(), 0
        contribution = None
        for index in order:
            n = seen[index]
            if n > done:
                new = hills[done:n]
                work.add_hills(new.s, new.rho, new.height)
                done = n
                contribution = work.interpolate(s, rho)
            if n:
                reduced[index] += contribution
    reduced /= kT
    return sliced_log_weights(reduced[:, :n_fit], counts[occupied], fast=True, evaluation_bias=reduced)


def _sample_support(log_weights):
    if not log_weights.size:
        return dict(count=0, ess=0.0)
    centred = log_weights - log_weights.max()
    return dict(count=int(centred.size),
                ess=float(np.exp(2 * logsumexp(centred) - logsumexp(2 * centred))))


def _dcd_site_rows(colvar, meta, start, stop, site):
    """Saved site-frame indices and their COLVAR rows. Frame 0 is one save interval after startup."""
    interval = (meta["settings"].get("trajectory_hills") or 0) * meta["bias"]["pace"] * (meta.get("timestep_ps") or 0)
    if interval <= 0:
        return None
    frames = int(np.floor((stop + 1.e-6) / interval))
    index = np.arange(frames)
    times = (index + 1) * interval
    post = times > start + 1.e-6
    rows = np.searchsorted(colvar["time"], times[post] - 1.e-6)
    if rows.size and (rows[-1] >= len(colvar["time"]) or
                      not np.allclose(colvar["time"][rows], times[post], rtol=0., atol=1.e-5)):
        return None
    keep = (colvar["s"][rows] >= site[0]) & (colvar["s"][rows] < site[1])
    return dict(frames=frames, frame_ps=interval, frame_index=index[post][keep].tolist()), rows[keep]


def saved_contact_wall(replica, structural_walkers):
    """Wall exposure for saved, contact-confirmed site frames using one replica's MBAR offsets.

    Restrict both numerator and denominator to the post-discard frames with the structural
    site's contacts. Different independent replicas have different offset normalizations.
    """
    entries = replica.get("dcd_site_weights")
    if entries is None:
        return None
    structures = {w["walker"]: w for w in structural_walkers}
    weights, wall, bound = [], [], []
    for w in entries:
        geometry = structures[w["walker"]]
        if geometry["frames"] != w["frames"] or not np.isclose(geometry["frame_ps"], w["frame_ps"]):
            raise ValueError("structural and energy reports use different DCD frames")
        bits = np.frombuffer(base64.b64decode(geometry["site_contact_bits_b64"], validate=True), dtype=np.uint8)
        if bits.size != (w["frames"] + 7) // 8:
            raise ValueError("structural contact mask length differs from the DCD frame count")
        contact = np.unpackbits(bits, bitorder="little")[np.asarray(w["frame_index"], dtype=int)].astype(bool)
        weights.extend(np.asarray(w["log_weight"])[contact])
        wall.extend(np.asarray(w["wall_kT"])[contact])
        bound.extend(np.asarray(w["bound_side"], dtype=bool)[contact])
    weights, wall, bound = np.asarray(weights), np.asarray(wall), np.asarray(bound, dtype=bool)
    out = {}
    for name, mask in (("site_contact", np.ones(len(weights), dtype=bool)), ("bound_side_contact", bound)):
        selected = weights[mask]
        above = wall[mask] > 1.
        out[name] = dict(**_sample_support(selected), above_1kT_count=int(above.sum()),
                         above_1kT_weight_fraction=float(np.exp(logsumexp(selected[above]) - logsumexp(selected)))
                         if selected.size else None)
    return out


def _bin_edges(grid, site, unbound):
    # Place edges directly at the locked bounds. No bin straddles the pocket and bulk.
    stops = np.unique([grid.s_min, *site, *unbound, grid.s_max])
    return np.concatenate([np.linspace(a, b, max(1, int(np.ceil((b - a) / 0.05))) + 1)[:-1]
                           for a, b in zip(stops, stops[1:])] + [stops[-1:]])


def _wall_support(s, rho, log_weights, funnel, kT, in_site, bound_below):
    """Fraction of the reweighted site distribution exposed to the funnel restraint."""
    axial = funnel.restraint_energy(s, np.zeros_like(rho))
    total = funnel.restraint_energy(s, rho)
    radial = total - axial
    site_log = logsumexp(log_weights[in_site])
    out = {}
    for name, mask in (("site", in_site), ("bound_side", in_site & (s <= bound_below))):
        weights = log_weights[mask]
        denominator = logsumexp(weights)
        levels = {}
        for threshold in (0.5, 1.0, 2.0):
            kinds = {}
            for component, energy in (("radial", radial), ("axial", axial), ("total", total)):
                above = energy[mask] > threshold * kT
                kinds[component] = dict(
                    count=int(above.sum()),
                    weight_fraction=float(np.exp(logsumexp(weights[above]) - denominator)) if weights.size else None)
            levels[str(threshold)] = kinds
        out[name] = dict(count=int(weights.size),
                         site_weight_fraction=float(np.exp(denominator - site_log)) if in_site.any() else None,
                         above_kT=levels)
    return out


def sample_binding(s, rho, log_weights, funnel, kT, site, unbound, core_radius, bound_below=None):
    """Estimate both bulk densities without binning or independently normalizing B and U."""
    in_site = (s >= site[0]) & (s < site[1])
    in_bulk = (s >= unbound[0]) & (s < unbound[1])
    # The logged CV is sqrt(physical radial distance squared + eps squared).
    core = in_bulk & (rho <= np.hypot(core_radius, funnel.eps))
    area = funnel.effective_area(s[in_bulk], kT)
    bulk_weights = log_weights[in_bulk] - np.log(area)
    support = dict(site=_sample_support(log_weights[in_site]), bulk=_sample_support(bulk_weights),
                   core=_sample_support(log_weights[core]))
    length = unbound[1] - unbound[0]
    site_log = logsumexp(log_weights[in_site])
    dG = dG_core = None
    if in_site.any() and in_bulk.any():
        log_density = logsumexp(bulk_weights) - np.log(length)
        dG = float(-kT * (site_log - log_density - np.log(STANDARD_VOLUME)))
    if in_site.any() and core.any():
        log_density = logsumexp(log_weights[core]) - np.log(np.pi * core_radius**2 * length)
        dG_core = float(-kT * (site_log - log_density - np.log(STANDARD_VOLUME)))
    wall = _wall_support(s, rho, log_weights, funnel, kT, in_site, site[1] if bound_below is None else bound_below)
    return dict(dG=dG, dG_core=dG_core, support=support, weighted_site_wall=wall)


def _roundtrips(s, bound_below, unbound):
    """Count only passages from the fixed site into the bulk window and back by this walker."""
    state = None
    out, back = 0, 0
    for value in s:
        if value <= bound_below:
            if state == "departed":
                back += 1
            state = "bound"
        elif unbound[0] <= value < unbound[1] and state == "bound":
            state = "departed"
            out += 1
    return dict(unbinding=out, rebinding=back)


def _read_last_time(path):
    """Read the last complete data row, including when the next row is still being written."""
    with Path(path).open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        # HILLS and COLVAR are short, fixed-column records. Include the full last two lines.
        start = max(0, end - 8192)
        handle.seek(start)
        rows = handle.read().splitlines(keepends=True)
    if start and rows:
        rows = rows[1:]
    for row in reversed(rows):
        if row.endswith(b"\n") and not row.startswith(b"#") and row.strip():
            return float(row.split()[0])
    raise ValueError(f"{path} contains no complete records")


def _sources(directory):
    if directory.name.startswith("w") and directory.name[1:].isdigit():
        raise ValueError(f"{directory} is a walker; pass the independent replica directory")
    return walker_directories(directory) or [directory]


def _validate_regions(grid, funnel, site, unbound, core_radius, bound_below):
    if not (len(site) == len(unbound) == 2 and
            np.isfinite([*site, *unbound, core_radius, bound_below]).all() and
            grid.s_min <= site[0] < site[1] <= unbound[0] < unbound[1] <= grid.s_max and
            funnel.lower <= unbound[0] < unbound[1] <= funnel.upper and
            core_radius > 0 and bound_below < unbound[0]):
        raise ValueError("binding analysis needs ordered finite site and bulk windows on the grid, "
                         "bulk inside both axial walls, a positive core radius and a fixed return threshold")
    if np.hypot(core_radius, funnel.eps) > min(funnel.wall_radius(np.array(unbound))):
        raise ValueError("the chosen disk extends beyond the wall-free bulk core")


def analyze_paper_replica(workdir, *, site, unbound, cut_ns, discard_ns=2.0,
                          core_radius=0.15, bound_below=None):
    """Read a common-time prefix of one replica without touching its outputs. Energies in kJ/mol.

    Retain later HILLS of peers when SYNC shows a historical sample actually felt them. Report
    how many kept-time samples depend on a peer beyond the cut, for prospective cost comparisons.
    """
    workdir = Path(workdir)
    sources = _sources(workdir)
    meta = [load_run_metadata(d) for d in sources]
    names = {d.name for d in sources}
    expected = {Path(d).name for m in meta for d in (m["settings"].get("walker_dirs") or [])}
    if expected - names:
        raise ValueError(f"{workdir} is missing walker directories: {sorted(expected - names)}")
    for m in meta[1:]:
        if (any(m.get(k) != meta[0].get(k) for k in ("bias", "grid", "funnel", "backend", "system", "timestep_ps"))
                or m["settings"].get("sync_hills") != meta[0]["settings"].get("sync_hills")):
            raise ValueError(f"{workdir}: walker production conditions differ")
    funnel, grid = Funnel.from_dict(meta[0]["funnel"]), BiasGrid(**meta[0]["grid"])
    bias = BiasSettings(**meta[0]["bias"])
    kT = bias.kT
    bound_below = site[1] if bound_below is None else bound_below
    _validate_regions(grid, funnel, site, unbound, core_radius, bound_below)
    if not (np.isfinite([cut_ns, discard_ns]).all() and 0 <= discard_ns < cut_ns):
        raise ValueError("discard_ns must be nonnegative and less than the measured cut_ns")
    stop, start = cut_ns * 1000.0, discard_ns * 1000.0
    parts = [read_hills(d / "HILLS") for d in sources]
    colvars = []
    for d, part in zip(sources, parts):
        data = read_colvar(d / "COLVAR")
        times = data["time"]
        if not times.size or not len(part) or min(times[-1], part.time[-1]) + 1.e-6 < stop:
            raise ValueError(f"{d} has not reached the requested {cut_ns:g} ns in both COLVAR and HILLS")
        mask = times <= stop + 1.e-6
        data = {k: v[mask] for k, v in data.items()}
        if (not np.isfinite([data["s"], data["rho"], data["time"]]).all() or
                not np.isfinite([part.time, part.s, part.rho, part.height]).all() or
                not np.all(np.diff(data["time"]) > 0) or not np.all(np.diff(part.time) > 0)):
            raise ValueError(f"{d}: nonfinite samples or nonmonotonic times")
        colvars.append(data)

    # The usual accumulated WT bias, projected and integrated over *these* site and bulk
    # limits. Include every deposited hill at each time, independent of its order within a tie.
    shared = SharedBias(grid, parts)
    cut_times = np.linspace(0.0, stop, 21)[1:]
    bias_dg = []
    for t in cut_times:
        counts = [np.searchsorted(h.time, t + 1.e-6, side="right") for h in parts]
        current = shared.at(counts)
        profile = projection_profile(current, bias.fes_scale, kT)
        bias_dg.append(profile_binding(grid.s, profile, funnel, kT, site, unbound))
    plateau = plateau_statistics(cut_times, np.array(bias_dg), 0.5)
    transitions = [_roundtrips(c["s"], bound_below, unbound) for c in colvars]
    site_s = _region_grid(grid.s, site)
    axial_wall = funnel.restraint_energy(site_s, np.zeros_like(site_s))
    empty_volume = np.trapezoid(funnel.effective_area(site_s, kT) * np.exp(-axial_wall / kT), site_s)
    result = dict(workdir=str(workdir), cut_ns=float(cut_ns), discard_ns=float(discard_ns),
                  n_walkers=len(parts), temperature=bias.temperature, status="unavailable",
                  bias=meta[0]["bias"], timestep_ps=meta[0].get("timestep_ps"),
                  backend=meta[0].get("backend"), sync_hills=meta[0]["settings"].get("sync_hills"),
                  diagnostics=[], dG=None, dG_core=None, dG_bias=plateau["mean"], dG_bias_final=bias_dg[-1],
                  bias_drift=plateau["drift"], bias_times_ns=(cut_times / 1000).tolist(), bias_dG_kj_mol=bias_dg,
                  support=None, profile=None, weighted_site_wall=None, dcd_site_weights=None,
                  unbound_roughness=None, peer_postcut_samples=None,
                  max_peer_time_ns=None, unknown_samples=None, walker_passages=transitions,
                  dG_empty_funnel=float(-kT * np.log(empty_volume / STANDARD_VOLUME)), no_binding=None,
                  unbinding_events=sum(t["unbinding"] for t in transitions),
                  rebinding_events=sum(t["rebinding"] for t in transitions))
    tables = hills_on_grid(sources, parts)
    if tables is None:
        result["diagnostics"].append("shared-walker SYNC history is missing")
        return result
    limits = np.array([np.searchsorted(h.time, stop + 1.e-6, side="right") for h in parts])
    postcut = unknown = 0
    maxima = np.zeros(len(parts), dtype=int)
    for colvar, own, table in zip(colvars, parts, tables):
        felt = table[_cycles(colvar, own)][colvar["time"] > start]
        unknown += int(np.any(felt < 0, axis=1).sum())
        postcut += int(np.any(felt > limits, axis=1).sum())
        if felt.size:
            maxima = np.maximum(maxima, felt.max(axis=0))
    result.update(unknown_samples=unknown, peer_postcut_samples=postcut,
                  max_peer_time_ns=max((float(h.time[n-1]) / 1000 for h, n in zip(parts, maxima) if n), default=0.0))
    result["required_walker_ns"] = [
        max(float(cut_ns), float(h.time[n - 1]) / 1000) if n else float(cut_ns)
        for h, n in zip(parts, maxima)
    ]
    if unknown:
        result["diagnostics"].append(f"the bias history of {unknown} post-discard samples is unavailable")
        return result
    selected = _slice_samples(colvars, parts, tables, start)
    if selected is None:
        result["diagnostics"].append("fewer than 100 post-discard samples with reconstructible bias")
        return result
    s, rho, biases, counts = selected
    # DCD saves and the deterministic MBAR training stride can miss each other entirely.
    # Evaluate the same fitted denominator at the saved frame coordinates as additional
    # targets; the primary estimate still uses only the original analysis samples.
    saved = [_dcd_site_rows(c, m, start, stop, site) for c, m in zip(colvars, meta)]
    dcd = []
    fit_count = s.size
    if all(r is not None for r in saved):
        extra_s, extra_rho = [], []
        for path, c, (record, indices) in zip(sources, colvars, saved):
            x, y = c["s"][indices], c["rho"][indices]
            extra_s.append(x)
            extra_rho.append(y)
            record.update(walker=str(path.relative_to(workdir.parent)), bound_side=(x <= bound_below).tolist(),
                          wall_kT=(funnel.restraint_energy(x, y) / kT).tolist())
            dcd.append(record)
        evaluated_s, evaluated_rho = np.concatenate([s, *extra_s]), np.concatenate([rho, *extra_rho])
    else:
        evaluated_s, evaluated_rho = s, rho
    try:
        all_weights = _weights_on_slices(evaluated_s, evaluated_rho, biases, counts, parts, grid, kT, fit_count)
    except ArithmeticError as exc:
        result["diagnostics"].append(str(exc))
        return result
    log_weights = all_weights[:fit_count]
    if dcd:
        at = fit_count
        for record in dcd:
            end = at + len(record["frame_index"])
            record["log_weight"] = all_weights[at:end].tolist()
            at = end
        result["dcd_site_weights"] = dcd
    result.update(sample_binding(s, rho, log_weights, funnel, kT, site, unbound, core_radius, bound_below))
    if result["dG"] is not None:
        result["no_binding"] = bool(result["dG_empty_funnel"] - result["dG"] < MIN_BINDING_EXCESS * kT)
    edges = _bin_edges(grid, site, unbound)
    centres = (edges[1:] + edges[:-1]) / 2
    pmf = reweighted_profile(s, log_weights, kT, edges)
    bulk = (centres >= unbound[0]) & (centres <= unbound[1])
    finite_bulk = np.isfinite(pmf[bulk])
    result["profile"] = dict(edges_nm=edges.tolist(), centres_nm=centres.tolist(),
                             pmf_kj_mol=[float(w) if np.isfinite(w) else None for w in pmf])
    result.update(n_mbar_samples=int(s.size), n_mbar_states=int(np.count_nonzero(counts)),
                  bulk_bins=int(bulk.sum()), sampled_bulk_bins=int(finite_bulk.sum()))
    site_bins = (centres >= site[0]) & (centres <= site[1])
    result.update(site_bins=int(site_bins.sum()), sampled_site_bins=int(np.isfinite(pmf[site_bins]).sum()))
    if finite_bulk.all():
        level = pmf[bulk] + kT * np.log(funnel.effective_area(centres[bulk], kT))
        result["unbound_roughness"] = float(np.std(level))
    if result["dG"] is None or result["dG_core"] is None or not finite_bulk.all():
        result["diagnostics"].append("weighted samples do not cover the site, bulk bins and bulk core")
    else:
        result["status"] = "available"
    if postcut:
        result["diagnostics"].append("some post-discard samples felt a peer's hills beyond the nominal cut")
    return result


def analyze_paper_replicas(workdirs, *, site, unbound, cut_ns=None, discard_ns=2.0,
                           core_radius=0.15, bound_below=None):
    """Measure all independent replicas at the same cut. Missing support never drops a replica."""
    directories = [Path(d) for d in workdirs]
    if not directories or len({d.resolve() for d in directories}) != len(directories):
        raise ValueError("provide distinct, independent replica directories")
    if cut_ns is None:
        cut_ns = min(_read_last_time(d / name) for r in directories for d in _sources(r)
                     for name in ("HILLS", "COLVAR")) / 1000
    runs = [analyze_paper_replica(d, site=site, unbound=unbound, cut_ns=cut_ns, discard_ns=discard_ns,
                                  core_radius=core_radius, bound_below=bound_below) for d in directories]
    if len({r["temperature"] for r in runs}) != 1:
        raise ValueError("replicas have different temperatures")
    available = all(r["status"] == "available" for r in runs)
    out = dict(status="available" if available else "unavailable", n_replicas=len(runs),
               cut_ns=float(cut_ns), discard_ns=float(discard_ns), site=list(site), unbound=list(unbound),
               core_radius=float(core_radius), bound_below=float(site[1] if bound_below is None else bound_below),
               replicas=runs, dG=None, sem=None, ci95=None, dG_kcal=None, ci95_kcal=None, dG_core=None,
               dG_bias=float(np.mean([r["dG_bias"] for r in runs])))
    if available:
        main = summarize_replicas([r["dG"] for r in runs])
        core = summarize_replicas([r["dG_core"] for r in runs])
        out.update(dG=main.mean, dG_kcal=main.mean / KCAL, sem=main.sem,
                   ci95=list(main.ci95) if main.ci95 else None,
                   ci95_kcal=[v / KCAL for v in main.ci95] if main.ci95 else None,
                   dG_core=core.mean, ci95_core=list(core.ci95) if core.ci95 else None)
    bias_summary = summarize_replicas([r["dG_bias"] for r in runs])
    out["ci95_bias"] = list(bias_summary.ci95) if bias_summary.ci95 else None
    out.setdefault("ci95_core", None)
    empty = float(np.mean([r["dG_empty_funnel"] for r in runs]))
    kT = KB * runs[0]["temperature"]
    out.update(dG_empty_funnel=empty, no_binding=bool(empty - out["dG"] < MIN_BINDING_EXCESS * kT)
               if available else None)
    return out
