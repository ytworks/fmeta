"""Score a paired TYK2/A2A experiment at one predeclared endpoint (energies in kJ/mol)."""

import argparse
import json
import math
from pathlib import Path

from fmeta.analysis.paper import saved_contact_wall
from fmeta.funnel.geometry import KB

EXAMPLES = {"tyk2": "tyk2_ejm46", "a2a": "a2a_theophylline"}
AFFINITY_M = {"tyk2": 4.8e-9, "a2a": 1.64e-6}
# Independent bias histories, walkers per bias, axial/radial sigma (nm), hill period (ps).
ARMS = {"a": (4, 8, 0.05, 0.10, 1), "b": (8, 4, 0.05, 0.10, 1),
        "c": (8, 4, 0.025, 0.030, 2), "d": (32, 1, 0.025, 0.030, 2)}
ENDPOINTS = (20, 40, 80, 100)


def precision_passes(score):
    """Require complete estimates, binding, a pocket return and a sufficiently narrow replica interval."""
    return all(score["checks"][key] for key in (
        "all_replicas_available", "binding", "contact_confirmed_roundtrip", "ci_halfwidth_within_limit"))


def score_system(name, arm, length, energy, structure):
    n_rep, n_walk, sigma_s, sigma_rho, pace = ARMS[arm]
    protocol = energy["protocol"]
    bias = protocol["bias"]
    region_file = Path(__file__).resolve().parents[1] / "examples" / EXAMPLES[name] / "paper-regions.json"
    frozen = json.loads(region_file.read_text())
    if energy["regions"] != frozen or structure["region"] != frozen:
        raise ValueError(f"{name}: results differ from the frozen region definition")
    if (energy["cut_ns"] != length or structure["cut_ns"] != length or length not in ENDPOINTS
            or energy["discard_ns"] != 2 or energy["base_seed"] != structure["base_seed"]):
        raise ValueError(f"{name}: inconsistent cut, seed or discarded interval")
    if (protocol["replicas"], protocol["walkers"], bias["sigma_s"], bias["sigma_rho"], protocol["pace_ps"]
            ) != (n_rep, n_walk, sigma_s, sigma_rho, pace):
        raise ValueError(f"{name}: protocol does not match arm {arm.upper()}")
    if (bias["bias_factor"], bias["height"], bias["temperature"], energy["recorded_backends"]
            ) != (10, 1.5, 300, ["cupotential"]):
        raise ValueError(f"{name}: unexpected bias, temperature or backend")
    runs = energy["replicas"]
    if (len(runs) != n_rep or energy["n_replicas"] != n_rep or
            any(r["n_walkers"] != n_walk or r["bias"] != bias or r["backend"] != "cupotential"
                or r["sync_hills"] != protocol["sync_hills"] or
                not math.isclose(r["timestep_ps"] * 1000, protocol["timestep_fs"]) for r in runs)):
        raise ValueError(f"{name}: recorded run conditions differ from the specified protocol")
    expected = {f"rep{i}/w{j}" if n_walk > 1 else f"rep{i}" for i in range(n_rep) for j in range(n_walk)}
    walkers = structure["walkers"]
    if len(walkers) != len(expected) or {w["walker"] for w in walkers} != expected:
        raise ValueError(f"{name}: missing or duplicate walkers in the structural assessment")
    returns = [0] * n_rep
    for walker in walkers:
        replica = int(walker["walker"].split("/")[0][3:])
        returns[replica] += walker["returns"]
    kT = KB * bias["temperature"]
    reference = kT * math.log(AFFINITY_M[name])
    limit = kT * math.log(5)
    dg, ci = energy["dG"], energy["ci95"]
    valid = energy["status"] == "available" and dg is not None and ci is not None
    if valid and not (all(math.isfinite(v) for v in [dg, *ci]) and ci[0] <= dg <= ci[1]):
        raise ValueError(f"{name}: invalid replica interval")
    error = abs(dg - reference) if valid else None
    halfwidth = (ci[1] - ci[0]) / 2 if valid else None
    checks = dict(all_replicas_available=valid, binding=energy["no_binding"] is False,
                  contact_confirmed_roundtrip=sum(returns) > 0,
                  affinity_within_fivefold=valid and error <= limit,
                  ci_halfwidth_within_limit=valid and halfwidth <= limit)
    support = [{k: r.get(k) for k in ("status", "dG", "dG_core", "dG_bias", "support", "unbound_roughness",
                                       "site_bins", "sampled_site_bins", "bulk_bins", "sampled_bulk_bins",
                                       "rebinding_events", "unknown_samples", "peer_postcut_samples",
                                       "max_peer_time_ns", "weighted_site_wall", "diagnostics")} for r in runs]
    for row, run in zip(support, runs):
        row["weighted_contact_wall"] = saved_contact_wall(run, walkers)
    observed = [w["lipid_checkpoint"] for w in walkers
                if (w.get("lipid_checkpoint") or {}).get("status") == "observed"]
    result = dict(passes=all(checks.values()), checks=checks, reference_M=AFFINITY_M[name],
                reference_kj_mol=reference, tolerance_kj_mol=limit, dG=dg, ci95=ci,
                affinity_M=math.exp(dg / kT) if valid else None, error_kj_mol=error,
                fold_error=math.exp(error / kT) if valid else None, ci_halfwidth_kj_mol=halfwidth,
                dG_core=energy["dG_core"], dG_bias=energy["dG_bias"], base_seed=energy["base_seed"],
                returns_by_replica=returns, departures=sum(w["departures"] for w in walkers),
                dcd_counts={k: sum(w[k] for w in walkers) for k in
                            ("site_frames", "site_contact_frames", "bulk_frames", "bulk_clear_frames",
                             "bulk_core_frames", "bulk_core_clear_frames", "site_contact_wall_above_kT_frames",
                             "bound_contact_wall_above_kT_frames")},
                reference_wall_kj_mol=structure["reference_wall_kj_mol"],
                maximum_bound_contact_wall_kj_mol=max((w["maximum_bound_contact_wall_kj_mol"] for w in walkers
                                                       if w["maximum_bound_contact_wall_kj_mol"] is not None),
                                                      default=None),
                observed_bulk_lipid_checkpoints=len(observed),
                cutoff_clear_bulk_lipid_checkpoints=sum(v["cutoff_clear"] for v in observed),
                replicas=support)
    result["precision_passes"] = precision_passes(result)
    return result


def score_pair(arm, length, pairs):
    results = {name: score_system(name, arm, length, *pairs[name]) for name in EXAMPLES}
    if pairs["tyk2"][0]["protocol"] != pairs["a2a"][0]["protocol"]:
        raise ValueError("the two systems used different sampling protocols")
    return dict(arm=arm, length_ns=length, protocol=pairs["tyk2"][0]["protocol"],
                systems=results, passes=all(v["passes"] for v in results.values()),
                precision_passes=all(v["precision_passes"] for v in results.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("length_ns", type=int, choices=ENDPOINTS)
    for name in EXAMPLES:
        parser.add_argument("--" + name, required=True, nargs=2, type=Path, metavar=("ENERGY_JSON", "STRUCTURE_JSON"))
    args = parser.parse_args()
    pairs = {name: [json.loads(p.read_text()) for p in getattr(args, name)] for name in EXAMPLES}
    print(json.dumps(score_pair(args.arm, args.length_ns, pairs), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
