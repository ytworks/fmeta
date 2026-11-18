"""Summarize independent-replica checkpoints on the same trajectories and frozen regions."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median, stdev

from export_binding_results import HASH_FIELDS, statistics_kcal_mol

from fmeta.analysis import KCAL


def replica_rows(energy):
    values = {}
    for row in energy["replicas"]:
        match = re.fullmatch(r"rep(\d+)", Path(row["workdir"]).name)
        if not match or row["n_walkers"] != 1:
            raise ValueError("each bias history must identify a distinct single-walker replica")
        index = int(match.group(1))
        if index in values or row["unknown_samples"] != 0 or row["peer_postcut_samples"] != 0:
            raise ValueError("duplicate replica or unresolved bias provenance")
        values[index] = row
    if set(values) != set(range(energy["protocol"]["replicas"])):
        raise ValueError("missing independent-replica histories")
    return values


def finite_primary(rows):
    return {i: r["dG"] / KCAL for i, r in rows.items() if r["dG"] is not None and math.isfinite(r["dG"])}


def paired_drift(energy, current, earlier):
    if (earlier["cut_ns"] >= energy["cut_ns"] or any(earlier[k] != energy[k] for k in
            ("protocol", "base_seed", "regions", "discard_ns", "reweighting_bias_interpolation"))):
        raise ValueError("drift requires two cutoffs on identical trajectories, settings and estimators")
    previous = finite_primary(replica_rows(earlier))
    ids = sorted(current.keys() & previous.keys())
    delta = [current[i] - previous[i] for i in ids]
    return dict(previous_cut_ns=earlier["cut_ns"], replica_ids=ids, n=len(ids),
                mean_signed_change_kcal_mol=mean(delta) if delta else None,
                median_absolute_change_kcal_mol=median(map(abs, delta)) if delta else None,
                largest_absolute_change_kcal_mol=max(map(abs, delta), default=None),
                per_replica_change_kcal_mol=delta)


def summarize(energy, geometry, coverage, baseline, previous):
    cut, seed = energy["cut_ns"], energy["base_seed"]
    protocol = energy["protocol"]
    if (protocol["walkers"] != 1 or protocol["replicas"] != 32 or energy["discard_ns"] != 2 or
            energy["reweighting_bias_interpolation"] != "production_natural_bicubic" or
            any(x["cut_ns"] != cut or x["base_seed"] != seed for x in (geometry, coverage)) or
            coverage["discard_ns"] != energy["discard_ns"] or geometry["region"] != energy["regions"]):
        raise ValueError("checkpoint reports disagree on protocol, regions, cut or seed")
    hashes = {k: energy["regions"][k] for k in HASH_FIELDS}
    if coverage["prepared_hashes"] != hashes or coverage["common_observation_period_ps"] != 2.:
        raise ValueError("the spatial coverage must use the same system and the common two-ps grid")
    raw = replica_rows(energy)
    finite = finite_primary(raw)
    frames = {r["walker"]: r for r in geometry["walkers"]}
    if set(frames) != {f"rep{i}" for i in raw} or len(frames) != len(geometry["walkers"]):
        raise ValueError("structural reports do not cover the original replica identities")
    marked = [i for i in sorted(raw) if raw[i]["status"] == "available"]
    reps = []
    for i in sorted(raw):
        e, s = raw[i], frames[f"rep{i}"]
        reps.append(dict(replica=i, primary_kcal_mol=finite.get(i), strict_available=i in marked,
                         mbar_samples=e["n_mbar_samples"], bulk_records=e["support"]["bulk"]["count"],
                         site_effective_samples=e["support"]["site"]["ess"],
                         bulk_effective_samples=e["support"]["bulk"]["ess"],
                         core_effective_samples=e["support"]["core"]["ess"],
                         bulk_bins_sampled=e["sampled_bulk_bins"], bulk_bins=e["bulk_bins"],
                         core_records=e["support"]["core"]["count"], pocket_departures=s["departures"],
                         pocket_returns=s["returns"], verified_bulk_frames=s["bulk_clear_frames"]))
    stats = statistics_kcal_mol(energy)
    primary_ci = stats["primary_reweighted"]["ci95"]
    finite_summary = None
    if 0 < len(finite) < len(raw):
        finite_summary = dict(n=len(finite), n_planned=len(raw), mean_kcal_mol=mean(finite.values()),
                              sample_sd_kcal_mol=stdev(finite.values()) if len(finite) > 1 else None)
    zones = {name: {k: v[k] for k in ("records", "physical_walkers_any", "independent_histories_any",
                                      "union_cells", "union_area_nm2", "median_walker_area_nm2",
                                      "median_history_area_nm2")} for name, v in coverage["zones"].items()}
    drifts = {r["cut_ns"]: paired_drift(energy, finite, r) for r in (baseline, previous)}
    return dict(cut_ns=cut, discard_ns=energy["discard_ns"], base_seed=seed, frozen_inputs=hashes,
                protocol=protocol, energy_unit="kcal/mol", area_unit="nm²",
                recorded_backends=energy["recorded_backends"], interpolation=energy["reweighting_bias_interpolation"],
                independent_bias_histories=len(raw), primary_finite=len(finite), strict_available=len(marked),
                primary_missing_replica_ids=sorted(raw.keys() - finite.keys()), strict_available_replica_ids=marked,
                primary_ci95_halfwidth_kcal_mol=(primary_ci[1] - primary_ci[0]) / 2 if primary_ci else None,
                method_statistics_kcal_mol=stats, conditional_primary_finite_subset=finite_summary,
                paired_primary_changes=[drifts[k] for k in sorted(drifts)],
                grid_origin_nm=coverage["grid_origin_nm"], cell_size_nm=coverage["cell_size_nm"],
                coverage=zones, walkers_both_site_and_bulk=coverage["walkers_both_site_and_bulk"],
                pocket_departures=sum(r["pocket_departures"] for r in reps),
                pocket_returns=sum(r["pocket_returns"] for r in reps),
                returning_physical_walkers=sum(r["pocket_returns"] > 0 for r in reps), replicas=reps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("energy", "structure", "coverage", "baseline_energy", "previous_energy"):
        parser.add_argument("--" + arg.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    files = (args.energy, args.structure, args.coverage, args.baseline_energy, args.previous_energy)
    print(json.dumps(summarize(*(json.loads(f.read_text()) for f in files)), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
