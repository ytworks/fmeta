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
    ([0., 4.184, 8.368], ["available", "unavailable", "available"], "available", 3, None),
    ([0., 4.184, 8.368], ["available"] * 3, "unavailable", 3, None),
    ([0.], ["available"], "available", 1, None),
])
def test_exported_sd_uses_every_planned_replica(monkeypatch, values, states, status, expected, sd):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    export = import_module("export_binding_results")
    energy = dict(protocol={"bias": {"temperature": 300}, "replicas": expected}, status=status,
                  n_replicas=len(values), replicas=[{"dG": v, "status": s} for v, s in zip(values, states)],
                  dG=4.184, dG_core=None, dG_bias=None, ci95=[0., 8.368])
    result = export.comparison_kcal_mol("tyk2", energy)["independent_replica_sd"]
    assert result == sd


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
    rows = export.measurements("tyk2", energy)["replicas"]
    assert [(r["status"], r["dG"], r["bulk_bins"], r["sampled_bulk_bins"]) for r in rows] == [
        ("unavailable", -10., 10, 8), ("unavailable", None, None, None)]


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
