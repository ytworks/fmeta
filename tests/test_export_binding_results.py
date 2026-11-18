import json
import sys
from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(("values", "states", "status", "expected", "sd"), [
    ([0., 4.184, 8.368], ["available"] * 3, "available", 3, 1.),
    ([0., 4.184], ["available"] * 2, "available", 3, None),
    ([0., None, 8.368], ["available"] * 3, "available", 3, None),
    ([0., float("nan"), 8.368], ["available"] * 3, "available", 3, None),
    ([0., 4.184, 8.368], ["available", "unavailable", "available"], "available", 3, 1.),
    ([0., 4.184, 8.368], ["available"] * 3, "unavailable", 3, 1.),
    ([0.], ["available"], "available", 1, None),
])
def test_exported_sd_uses_every_planned_replica(monkeypatch, values, states, status, expected, sd):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    export = import_module("export_binding_results")
    energy = dict(protocol={"bias": {"temperature": 300}, "replicas": expected}, status=status,
                  n_replicas=len(values), replicas=[{"dG": v, "status": s} for v, s in zip(values, states)],
                  dG=4.184, dG_core=None, dG_bias=None, ci95=[0., 8.368])
    result = export.comparison_kcal_mol("tyk2", energy)["independent_replica_sd"]
    assert result == (pytest.approx(sd) if sd is not None else None)


def test_precision_gate_ignores_reference_affinity(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    precision_passes = import_module("score_binding").precision_passes
    checks = dict(all_replicas_available=True, binding=True, contact_confirmed_roundtrip=True,
                  ci_halfwidth_within_limit=True, affinity_within_fivefold=False)
    assert precision_passes({"checks": checks})
    for name in ("all_replicas_available", "binding", "contact_confirmed_roundtrip", "ci_halfwidth_within_limit"):
        assert not precision_passes({"checks": {**checks, name: False}})


def test_export_preserves_replica_status_and_observed_bulk_bins(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    export = import_module("export_binding_results")
    incomplete = dict(status="unavailable", dG=-10., dG_core=-11., dG_bias=-12., unbound_roughness=None,
                      support=None, weighted_site_wall=None)
    energy = dict(protocol={"bias": {"temperature": 300}, "replicas": 2}, status="unavailable", n_replicas=2,
                  replicas=[{**incomplete, "bulk_bins": 10, "sampled_bulk_bins": 8}, {**incomplete, "dG": None}],
                  dG=None, dG_core=None, dG_bias=-12., ci95=None, ci95_core=None, ci95_bias=None, base_seed=7)
    measurement = export.measurements("tyk2", energy)
    rows = measurement["replicas"]
    assert [(r["status"], r["dG"], r["bulk_bins"], r["sampled_bulk_bins"]) for r in rows] == [
        ("unavailable", -10., 10, 8), ("unavailable", None, None, None)]
    assert [r["primary_calculable"] for r in rows] == [True, False]
    assert measurement["n_primary_calculable"] == 1 and measurement["n_sampling_quality_pass"] == 0
    assert measurement["comparison_kcal_mol"]["predicted_mean"] is None
    assert measurement["primary_finite_subset_kcal_mol"] == dict(n_included=1, n_planned=2,
                                                                mean=-10./export.KCAL, sample_sd=None)
    assert measurement["method_statistics_kcal_mol"]["wt_bias_late"]["sd"] == 0.


def test_export_keeps_single_trajectory_time_sd_and_independent_replica_sd_distinct(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    export = import_module("export_binding_results")
    rows = []
    for offset in (0, -1):
        trace = [float(i + offset) for i in range(4)]
        rows.append(dict(status="available", support=None, weighted_site_wall=None, unbound_roughness=None,
                         dG=-1., dG_core=-1., dG_bias=export.KCAL*(2.5+offset), dG_bias_final=trace[-1]*export.KCAL,
                         bias_times_ns=[1, 2, 3, 4], bias_dG_kj_mol=[v * export.KCAL for v in trace]))
    energy = dict(replicas=rows, status="available", protocol={"bias":{"temperature":300}, "replicas":2},
                  n_replicas=2, dG=-1, dG_core=-1, dG_bias=2*export.KCAL, ci95=None, ci95_core=None,
                  ci95_bias=None, base_seed=7)
    measure = export.measurements("tyk2", energy)
    assert measure["wt_bias_times_ns"] == [1, 2, 3, 4]
    assert measure["replicas"][0]["wt_bias_trace_kcal_mol"] == [0., 1., 2., 3.]
    assert measure["replicas"][1]["wt_bias_final_minus_late_kcal_mol"] == pytest.approx(.5)
    assert measure["replicas"][1]["wt_bias_late_time_sd_kcal_mol"] == pytest.approx(.5)
    stats = measure["method_statistics_kcal_mol"]["wt_bias_final"]
    assert stats["mean"] == pytest.approx(2.5) and stats["sd"] == pytest.approx(2**-.5)
    assert stats["complete"] and stats["n_finite"] == stats["n_total"] == 2
    assert stats["ci95"][0] < 2.5 < stats["ci95"][1]


def test_repeated_provisional_cli_sorts_same_system_endpoints(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    export = import_module("export_binding_results")
    manifest = dict(tracked_source_base_commit="0"*40, tracked_src_fmeta_tree="1"*40,
                    wrapper_segments={"tyk2": [dict(end_ns=n, wrapper_commit="0"*40, wrapper_dirty=False,
                                                    src_fmeta_tree="1"*40) for n in (20, 40, 80)]})
    files = {"manifest": manifest}
    argv = ["export_binding_results", "--arm", "c", "--analysis-commit", "2"*40, "--production-source", "manifest"]
    for n in (80, 20, 40):
        files[f"energy{n}"] = dict(cut_ns=n, regions={k: "0"*64 for k in export.HASH_FIELDS},
                                   protocol={"replicas": 8}, base_seed=1)
        files[f"struct{n}"] = {}
        argv.extend(["--provisional", "tyk2", f"energy{n}", f"struct{n}"])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(export, "read", lambda path: files[str(path)])
    monkeypatch.setattr(export, "score_system",
                        lambda name, arm, n, e, s: {"passes": False, "precision_passes": n > 20})
    monkeypatch.setattr(export, "measurements", lambda name, e, score: score)
    export.main()
    endpoints = json.loads(capsys.readouterr().out)["endpoints"]
    assert [row["length_ns"] for row in endpoints] == [20, 40, 80]
    assert all(row["passes"] is None and row["precision_passes"] is None for row in endpoints)
    assert all(row["pending_systems"] == ["a2a"] and list(row["systems"]) == ["tyk2"] for row in endpoints)
    assert [row["systems"]["tyk2"]["precision_passes"] for row in endpoints] == [False, True, True]


@pytest.mark.parametrize(("name", "example"), [("tyk2", "tyk2_ejm46"), ("a2a", "a2a_theophylline")])
def test_one_shared_bias_exports_timepoints_without_between_replica_uncertainty(monkeypatch, capsys, name, example):
    repo = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(repo / "scripts"))
    export = import_module("export_binding_results")
    regions = json.loads((repo / "examples" / example / "paper-regions.json").read_text())
    bias = dict(bias_factor=10, height=1.5, temperature=300, sigma_s=.025, sigma_rho=.03)
    protocol = dict(replicas=1, walkers=32, timestep_fs=4, pace_ps=2, sync_hills=1, bias=bias)
    replica = dict(status="available", n_walkers=32, sync_hills=1, backend="cupotential", bias=bias,
                   timestep_ps=.004, dG=-41.84, dG_core=-42., dG_bias=-40., dG_bias_final=-38.,
                   bias_drift=4., support=None, unbound_roughness=None, weighted_site_wall=None,
                   unknown_samples=0, peer_postcut_samples=0)
    walkers = [dict(walker=f"rep0/w{i}", frames=50, site_frames=10, site_contact_frames=10,
                    bulk_frames=2, bulk_clear_frames=2, bulk_core_frames=1, bulk_core_clear_frames=1,
                    site_contact_wall_above_kT_frames=0, bound_contact_wall_above_kT_frames=0,
                    maximum_bound_contact_wall_kj_mol=None,
                    departures=1, returns=0, passages=[{"to":"bulk", "time_ns":4.}]) for i in range(32)]
    base = dict(base_seed=19711801, no_binding=False, status="available", discard_ns=2., dG=-41.84, dG_core=-42.,
                dG_bias=-40., ci95=None, ci95_core=None, ci95_bias=None, regions=regions, protocol=protocol,
                recorded_backends=["cupotential"], n_replicas=1,
                reweighting_bias_interpolation="production_natural_bicubic")
    structure = dict(region=regions, base_seed=19711801, reference_wall_kj_mol={"prepared":0., "equil":0.})
    files = {"manifest": dict(tracked_source_base_commit="0"*40, tracked_src_fmeta_tree="1"*40,
                              wrapper_segments={name:[dict(end_ns=n, wrapper_commit="0"*40, wrapper_dirty=False,
                                                            src_fmeta_tree="1"*40) for n in (5, 10)]})}
    argv = ["export_binding_results", "--arm", "e", "--analysis-commit", "2"*40, "--production-source", "manifest"]
    for n in (10, 5):
        run = dict(replica, max_peer_time_ns=n, bias_times_ns=[n/2, n], bias_dG_kj_mol=[-42., -38.])
        files[f"energy{n}"] = dict(base, cut_ns=n, replicas=[run])
        files[f"struct{n}"] = dict(structure, cut_ns=n, walkers=[dict(w, frames=n*10) for w in walkers])
        argv.extend(["--provisional", name, f"energy{n}", f"struct{n}"])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(export, "read", lambda path: files[str(path)])
    export.main()
    out = json.loads(capsys.readouterr().out)
    assert out["sampling"]["independent_bias_histories"] == 1 and out["sampling"]["walkers_per_bias"] == 32
    assert [p["length_ns"] for p in out["endpoints"]] == [5, 10]
    for p in out["endpoints"]:
        record = p["systems"][name]
        assert p["purpose"] == "single_shared_bias" and "pending_systems" not in p
        assert record["passes"] is None and record["precision_passes"] is None
        assert record["n_sampling_quality_pass"] == 1
        assert record["comparison_kcal_mol"]["predicted_mean"] == pytest.approx(-10.)
        assert record["affinity_fold_error"] is not None
        assert len(record["structure"]["walkers"]) == 32 and record["replicas"][0]["max_peer_time_ns"] == p["length_ns"]
        for result in record["method_statistics_kcal_mol"].values():
            assert result["n_total"] == result["n_finite"] == 1
            assert result["mean"] is not None and result["sd"] is result["sem"] is result["ci95"] is None
