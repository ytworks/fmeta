"""Export measured scientific quantities from paired fmeta analysis and structural reports."""

import argparse
import json
import math
import re
from pathlib import Path

from score_binding import AFFINITY_M, ARMS, EXAMPLES, score_pair

from fmeta.analysis import KCAL
from fmeta.funnel.geometry import KB

HASH_FIELDS = tuple(name + "_sha256" for name in ("system", "positions", "topology", "funnel", "equil_state"))


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
    return dict(tracked_source_base_commit=base, tracked_src_fmeta_tree=tree, wrapper_segments=groups,
                dirty_state_note="The wrapper records commit and dirty status. The tracked src/fmeta tree "
                                 "matches the base in all listed commits. Uncommitted historical file "
                                 "contents cannot be recovered from these records.")


def comparison_kcal_mol(name, energy):
    kT = KB * energy["protocol"]["bias"]["temperature"] / KCAL
    reference = kT * math.log(AFFINITY_M[name])
    predicted = energy["dG"] / KCAL if energy["dG"] is not None else None
    ci = [v / KCAL for v in energy["ci95"]] if energy["ci95"] is not None else None
    return dict(reference=reference, predicted_mean=predicted,
                signed_error_predicted_minus_reference=predicted - reference if predicted is not None else None,
                ci95_independent_replicas=ci, ci95_halfwidth=(ci[1] - ci[0]) / 2 if ci is not None else None,
                fivefold_limit=kT * math.log(5),
                core_mean=energy["dG_core"] / KCAL if energy["dG_core"] is not None else None,
                accumulated_wt_mean=energy["dG_bias"] / KCAL if energy["dG_bias"] is not None else None)


def measurements(name, energy, score=None):
    rows = []
    for i, replica in enumerate(energy["replicas"]):
        support = replica["support"]
        contact = score["replicas"][i]["weighted_contact_wall"] if score else None
        site_wall = replica["weighted_site_wall"]
        rows.append(dict(
            replica=i, dG=replica["dG"], dG_core=replica["dG_core"], dG_bias=replica["dG_bias"],
            bulk_pmf_sd=replica["unbound_roughness"],
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
    physical = None
    if score:
        physical = dict(
            returns_by_replica=score["returns_by_replica"], departures=score["departures"],
            counts={k: score["dcd_counts"][k] for k in ("site_frames", "site_contact_frames",
                    "bulk_frames", "bulk_clear_frames", "bulk_core_frames", "bulk_core_clear_frames",
                    "site_contact_wall_above_kT_frames", "bound_contact_wall_above_kT_frames")},
            reference_wall_kj_mol={k: score["reference_wall_kj_mol"][k] for k in ("prepared", "equil")},
        )
    return dict(comparison_kcal_mol=comparison_kcal_mol(name, energy),
                base_seed=energy["base_seed"], status=energy["status"], dG=energy["dG"], ci95=energy["ci95"],
                dG_core=energy["dG_core"], ci95_core=energy["ci95_core"], dG_bias=energy["dG_bias"],
                ci95_bias=energy["ci95_bias"], affinity_fold_error=score["fold_error"] if score else None,
                passes=score["passes"] if score else None, structure=physical, replicas=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--phase", choices=("screening", "validation"), default="screening")
    parser.add_argument("--production-source", type=Path, required=True,
                        help="JSON with tracked_source_base_commit, tracked_src_fmeta_tree and "
                             "per-system wrapper_segments (end_ns, wrapper_commit, wrapper_dirty, src_fmeta_tree)")
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--paired", action="append", nargs=4, required=True,
                        metavar=("TYK2_ENERGY", "TYK2_STRUCTURE", "A2A_ENERGY", "A2A_STRUCTURE"))
    parser.add_argument("--drift", nargs=2, metavar=("SYSTEM", "ENERGY"))
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.analysis_commit):
        parser.error("analysis commit requires a full hexadecimal Git object ID")
    inputs, results = {}, []
    for filenames in args.paired:
        ty, ty_struct, aa, aa_struct = map(read, filenames)
        pair = {"tyk2": (ty, ty_struct), "a2a": (aa, aa_struct)}
        length = int(ty["cut_ns"])
        if any(p["length_ns"] == length for p in results):
            parser.error("duplicate paired endpoint")
        score = score_pair(args.arm, length, pair)
        for name, (energy, _) in pair.items():
            hashes = {key: energy["regions"][key] for key in HASH_FIELDS}
            if inputs.setdefault(name, hashes) != hashes:
                parser.error(f"inconsistent {name} prepared inputs across endpoints")
        results.append(dict(length_ns=length, purpose=args.phase, passes=score["passes"],
                            systems={name: measurements(name, pair[name][0], score["systems"][name])
                                     for name in EXAMPLES}))
    if args.drift:
        name, filename = args.drift
        if name not in EXAMPLES:
            parser.error("unknown drift diagnostic system")
        energy = read(filename)
        if (energy["protocol"] != ty["protocol"] or energy["base_seed"] != pair[name][0]["base_seed"]
                or {key: energy["regions"][key] for key in HASH_FIELDS} != inputs[name]):
            parser.error("drift diagnostic has different inputs, seed or protocol")
        results.append(dict(length_ns=energy["cut_ns"], purpose="drift_only", passes=None,
                            systems={name: measurements(name, energy)}))
    report = dict(arm=args.arm, energy_unit="kJ/mol", length_unit="ns/walker",
                  production_source=production_source(args.production_source, results),
                  analysis_commit=args.analysis_commit,
                  frozen_inputs=inputs, endpoints=sorted(results, key=lambda r: r["length_ns"]))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
