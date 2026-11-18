"""Export measured scientific quantities from paired fmeta analysis and structural reports."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev, stdev

from score_binding import AFFINITY_M, EXAMPLES, TRIAL_ARMS, score_pair, score_system

from fmeta.analysis import KCAL
from fmeta.analysis.paper import method_statistics
from fmeta.funnel.geometry import KB

HASH_FIELDS = tuple(name + "_sha256" for name in ("system", "positions", "topology", "funnel", "equil_state"))
ESTIMATORS = dict(primary_reweighted="dG", core_reweighted="dG_core",
                  wt_bias_late="dG_bias", wt_bias_final="dG_bias_final")
QUALITY_FLAGS = ("reweighting_solved", "site_sampled", "bulk_sampled", "core_sampled", "all_bulk_bins_sampled")


def read(path):
    return json.loads(Path(path).read_text())


def production_source(path, endpoints):
    """Read checked Git/Docker provenance for each segment of the cumulative trajectories."""
    data = read(path)
    base, tree = data["tracked_source_base_commit"], data["tracked_src_fmeta_tree"]
    if not all(re.fullmatch("[0-9a-f]{40}", value) for value in (base, tree)):
        raise ValueError("invalid commit or tree ID in the production manifest")
    groups = {}
    for name in EXAMPLES:
        cuts = {e["length_ns"] for e in endpoints if name in e["systems"]}
        if not cuts:
            continue
        entries = []
        for item in data["wrapper_segments"][name]:
            n, commit, dirty = item["end_ns"], item["wrapper_commit"], item["wrapper_dirty"]
            if (not isinstance(n, int) or n <= 0 or type(dirty) is not bool
                    or not re.fullmatch("[0-9a-f]{40}", commit) or item["src_fmeta_tree"] != tree):
                raise ValueError(f"invalid segment identity or tracked source mismatch for {name}")
            if n <= max(cuts):
                entries.append(dict(end_ns=n, wrapper_commit=commit, wrapper_dirty=dirty))
        lengths = [entry["end_ns"] for entry in entries]
        if lengths != sorted(set(lengths)) or not cuts.issubset(lengths):
            raise ValueError(f"missing or duplicate observed production segments for {name}")
        groups[name] = entries
    source = dict(tracked_source_base_commit=base, tracked_src_fmeta_tree=tree, wrapper_segments=groups)
    if any(s["wrapper_dirty"] for segments in groups.values() for s in segments):
        source["dirty_state_note"] = ("The wrapper records commit and dirty status. The tracked src/fmeta tree "
                                      "matches the base in all listed commits. Uncommitted historical file "
                                      "contents cannot be recovered from these records.")
    return source


def statistics_kcal_mol(energy):
    """Independent-replica statistics by estimator, including when another estimator lacks support."""
    rows = energy["replicas"]
    expected = energy["protocol"]["replicas"]
    if expected < len(rows) or len(rows) != energy["n_replicas"]:
        raise ValueError("replica count differs from the run")
    result = {}
    for name, key in ESTIMATORS.items():
        if "method_statistics" in energy and len(rows) == expected:
            data = energy["method_statistics"][name]
        else:
            data = method_statistics([row.get(key) for row in rows] + [None] * (expected - len(rows)))
        result[name] = dict(n_total=data["n_total"], n_finite=data["n_finite"], complete=data["complete"],
                            **{k: data[k] / KCAL if data[k] is not None else None for k in ("mean", "sd", "sem")},
                            ci95=[v / KCAL for v in data["ci95"]] if data["ci95"] is not None else None)
    return result


def comparison_kcal_mol(name, energy, statistics=None):
    statistics = statistics if statistics is not None else statistics_kcal_mol(energy)
    kT = KB * energy["protocol"]["bias"]["temperature"] / KCAL
    reference = kT * math.log(AFFINITY_M[name])
    principal = statistics["primary_reweighted"]
    predicted, ci = principal["mean"], principal["ci95"]
    return dict(reference=reference, predicted_mean=predicted,
                signed_error_predicted_minus_reference=predicted - reference if predicted is not None else None,
                independent_replica_sd=principal["sd"], ci95_independent_replicas=ci,
                ci95_halfwidth=(ci[1] - ci[0]) / 2 if ci is not None else None,
                fivefold_limit=kT * math.log(5),
                core_mean=statistics["core_reweighted"]["mean"], accumulated_wt_mean=statistics["wt_bias_late"]["mean"])


def measurements(name, energy, score=None):
    rows = []
    times = energy["replicas"][0].get("bias_times_ns", [])
    stats = statistics_kcal_mol(energy)
    for i, replica in enumerate(energy["replicas"]):
        support = replica["support"]
        contact = score["replicas"][i]["weighted_contact_wall"] if score else None
        site_wall = replica["weighted_site_wall"]
        if replica.get("bias_times_ns", []) != times:
            raise ValueError("bias trajectories have different time grids")
        trace = [v / KCAL for v in replica.get("bias_dG_kj_mol", [])]
        if len(trace) != len(times):
            raise ValueError("bias trajectory and saved times differ in length")
        # Older analysis records contain the cumulative trace and late mean, but no temporal SD.
        late_sd = replica.get("bias_late_temporal_sd")
        late_sd_kcal = late_sd / KCAL if late_sd is not None else None
        if late_sd_kcal is None and trace:
            late_sd_kcal = pstdev(trace[-max(2, round(len(trace) / 2)):])
        rows.append(dict(
            replica=i, status=replica["status"], dG=replica["dG"], dG_core=replica["dG_core"],
            primary_calculable=replica["dG"] is not None and math.isfinite(replica["dG"]),
            sampling_quality={k: replica["sampling_quality"][k] for k in QUALITY_FLAGS}
            if "sampling_quality" in replica else None,
            dG_bias=replica["dG_bias"], bulk_pmf_sd=replica["unbound_roughness"],
            wt_bias_trace_kcal_mol=trace, wt_bias_final_kcal_mol=trace[-1] if trace else None,
            wt_bias_late_mean_kcal_mol=replica["dG_bias"] / KCAL if replica["dG_bias"] is not None else None,
            wt_bias_final_minus_late_kcal_mol=trace[-1] - replica["dG_bias"] / KCAL
            if trace and replica["dG_bias"] is not None else None,
            wt_bias_late_time_sd_kcal_mol=late_sd_kcal,
            bulk_bins=replica.get("bulk_bins"), sampled_bulk_bins=replica.get("sampled_bulk_bins"),
            mbar_samples=replica.get("n_mbar_samples"), mbar_states=replica.get("n_mbar_states"),
            support={k: {q: support[k][q] for q in ("count", "ess")} for k in ("site", "bulk", "core")}
            if support else None,
            weighted_site_above_1kT=site_wall["site"]["above_kT"]["1.0"]["total"]["weight_fraction"]
            if site_wall else None,
            weighted_site_contact_above_1kT=contact["site_contact"]["above_1kT_weight_fraction"]
            if contact else None,
            weighted_bound_contact_above_1kT=contact["bound_side_contact"]["above_1kT_weight_fraction"]
            if contact else None,
        ))
        if energy["protocol"]["replicas"] == 1:
            rows[-1].update(unknown_samples=replica.get("unknown_samples"),
                            peer_postcut_samples=replica.get("peer_postcut_samples"),
                            max_peer_time_ns=replica.get("max_peer_time_ns"),
                            wt_bias_late_drift_kcal_mol=replica["bias_drift"] / KCAL)
    physical = None
    if score:
        physical = dict(
            returns_by_replica=score["returns_by_replica"], departures=score["departures"],
            counts={k: score["dcd_counts"][k] for k in ("site_frames", "site_contact_frames",
                    "bulk_frames", "bulk_clear_frames", "bulk_core_frames", "bulk_core_clear_frames",
                    "site_contact_wall_above_kT_frames", "bound_contact_wall_above_kT_frames")},
            reference_wall_kj_mol={k: score["reference_wall_kj_mol"][k] for k in ("prepared", "equil")},
        )
        if "structural_walkers" in score:
            physical["walkers"] = score["structural_walkers"]
    finite = [r["dG"] / KCAL for r in rows if r["primary_calculable"]]
    conditional = None
    if finite and not stats["primary_reweighted"]["complete"]:
        conditional = dict(n_included=len(finite), n_planned=energy["protocol"]["replicas"],
                           mean=mean(finite), sample_sd=stdev(finite) if len(finite) > 1 else None)
    return dict(comparison_kcal_mol=comparison_kcal_mol(name, energy, stats), method_statistics_kcal_mol=stats,
                n_primary_calculable=len(finite), n_sampling_quality_pass=sum(r["status"] == "available" for r in rows),
                primary_finite_subset_kcal_mol=conditional, wt_bias_times_ns=times,
                base_seed=energy["base_seed"], status=energy["status"], dG=energy["dG"], ci95=energy["ci95"],
                dG_core=energy["dG_core"], ci95_core=energy["ci95_core"], dG_bias=energy["dG_bias"],
                ci95_bias=energy["ci95_bias"], affinity_fold_error=score["fold_error"] if score else None,
                passes=score["passes"] if score else None,
                precision_passes=score["precision_passes"] if score else None, structure=physical, replicas=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=TRIAL_ARMS, required=True)
    parser.add_argument("--phase", choices=("screening", "validation"), default="screening")
    parser.add_argument("--production-source", type=Path, required=True,
                        help="JSON with tracked_source_base_commit, tracked_src_fmeta_tree and "
                             "per-system wrapper_segments (end_ns, wrapper_commit, wrapper_dirty, src_fmeta_tree)")
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--paired", action="append", nargs=4, default=[],
                        metavar=("TYK2_ENERGY", "TYK2_STRUCTURE", "A2A_ENERGY", "A2A_STRUCTURE"))
    parser.add_argument("--provisional", action="append", nargs=3, default=[],
                        metavar=("SYSTEM", "ENERGY", "STRUCTURE"),
                        help="one completed system at a pending paired endpoint; repeat for different lengths")
    parser.add_argument("--drift", nargs=2, metavar=("SYSTEM", "ENERGY"))
    args = parser.parse_args()
    single_bias = args.arm == "e"
    if single_bias and (args.paired or args.drift or args.phase != "screening"
                        or len({group[0] for group in args.provisional}) != 1):
        parser.error("E requires one example's measured energy and structure at each cut")
    if not re.fullmatch("[0-9a-f]{40}", args.analysis_commit):
        parser.error("analysis commit requires a full hexadecimal Git object ID")
    groups = []
    for filenames in args.paired:
        ty, ty_struct, aa, aa_struct = map(read, filenames)
        groups.append({"tyk2": (ty, ty_struct), "a2a": (aa, aa_struct)})
    for name, energy_file, structure_file in args.provisional:
        if name not in EXAMPLES:
            parser.error("unknown provisional system")
        groups.append({name: (read(energy_file), read(structure_file))})
    if not groups:
        parser.error("supply a paired or provisional endpoint")
    inputs, runs, results, interpolations = {}, {}, [], set()
    for group in groups:
        name, (energy, structure) = next(iter(group.items()))
        length = int(energy["cut_ns"])
        if any(p["length_ns"] == length for p in results):
            parser.error("duplicate endpoint")
        purpose = "single_shared_bias" if single_bias else args.phase
        endpoint = dict(length_ns=length, purpose=purpose, passes=None, precision_passes=None, systems={})
        if len(group) == len(EXAMPLES):
            pair_score = score_pair(args.arm, length, group)
            scores = pair_score["systems"]
            endpoint["passes"] = pair_score["passes"]
            endpoint["precision_passes"] = pair_score["precision_passes"]
        else:
            scores = {name: score_system(name, args.arm, length, energy, structure)}
            if not single_bias:
                endpoint["pending_systems"] = sorted(set(EXAMPLES) - group.keys())
        for name, (energy, _) in group.items():
            hashes = {key: energy["regions"][key] for key in HASH_FIELDS}
            run = energy["protocol"], energy["base_seed"]
            if inputs.setdefault(name, hashes) != hashes:
                parser.error(f"inconsistent {name} prepared inputs across endpoints")
            if runs.setdefault(name, run) != run:
                parser.error(f"inconsistent {name} protocol or seed across endpoints")
            interpolations.add(energy.get("reweighting_bias_interpolation", "legacy_bilinear"))
            endpoint["systems"][name] = measurements(name, energy, scores[name])
        results.append(endpoint)
    if args.drift:
        name, filename = args.drift
        if name not in EXAMPLES:
            parser.error("unknown drift diagnostic system")
        energy = read(filename)
        interpolations.add(energy.get("reweighting_bias_interpolation", "legacy_bilinear"))
        if (runs.get(name) != (energy["protocol"], energy["base_seed"])
                or {key: energy["regions"][key] for key in HASH_FIELDS} != inputs[name]):
            parser.error("drift diagnostic has different inputs, seed or protocol")
        results.append(dict(length_ns=energy["cut_ns"], purpose="drift_only", passes=None, precision_passes=None,
                            systems={name: measurements(name, energy)}))
    if len(interpolations) != 1:
        parser.error("endpoint reports use different reweighting bias interpolation")
    report = dict(arm=args.arm, energy_unit="kJ/mol", length_unit="ns/walker",
                  reweighting_bias_interpolation=interpolations.pop(),
                  production_source=production_source(args.production_source, results),
                  analysis_commit=args.analysis_commit,
                  frozen_inputs=inputs, endpoints=sorted(results, key=lambda r: r["length_ns"]))
    if single_bias:
        protocol, seed = next(iter(runs.values()))
        report["sampling"] = dict(independent_bias_histories=protocol["replicas"],
                                  walkers_per_bias=protocol["walkers"], base_seed=seed,
                                  hill_period_ps=protocol["pace_ps"], hill_height_kj_mol=protocol["bias"]["height"],
                                  sigma_s_nm=protocol["bias"]["sigma_s"], sigma_rho_nm=protocol["bias"]["sigma_rho"],
                                  bias_factor=protocol["bias"]["bias_factor"])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
