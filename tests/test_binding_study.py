"""Paired and TYK2-first study decisions on independent replicas and fresh validation seeds."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fmeta.config import Config
from fmeta.workflow import analyze_paper_stage


def test_paper_analysis_rejects_another_equilibration_and_incomplete_ensemble(tmp_path):
    regions = dict(site=[1, 2], unbound=[3, 4], core_radius=.15, bound_below=1.2)
    for key, filename in (("system", "prepared/system.xml"), ("positions", "prepared/positions.npy"),
                          ("topology", "prepared/topology.cif"), ("funnel", "prepared/funnel.yaml"),
                          ("equil_state", "equil/state.xml")):
        file = tmp_path / filename
        file.parent.mkdir(exist_ok=True)
        file.write_text(filename)
        regions[key + "_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    region_file = tmp_path / "regions.json"
    region_file.write_text(json.dumps(regions))
    Config().save(tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="replica directories"):
        analyze_paper_stage(tmp_path, region_file)
    (tmp_path / "equil/state.xml").write_text("different initial state")
    with pytest.raises(ValueError, match="equil/state.xml"):
        analyze_paper_stage(tmp_path, region_file)


def test_fresh_pair_preserves_preparation_and_uses_one_sampling_protocol(tmp_path, monkeypatch):
    repo = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(repo / "scripts"))
    prepare = importlib.import_module("prepare_binding_trial").prepare_trial
    protocols = []
    for seed, name in enumerate(("tyk2_ejm46", "a2a_theophylline"), start=99):
        source, target = tmp_path / name, tmp_path / "c" / name
        (source / "prepared").mkdir(parents=True)
        (source / "equil").mkdir()
        (source / "equil/state.xml").write_text(name)
        (source / "rep0").mkdir()
        (source / "rep0/HILLS").write_text("old hills")
        config = Config.load(repo / "examples" / name / "fmeta.yaml")
        config.save(source / "config.yaml")
        prepare(source, target, "c", 20, seed)
        trial = Config.load(target / "config.yaml")
        assert trial.system == config.system and trial.membrane == config.membrane
        assert (target / "equil/state.xml").read_text() == name and not (target / "rep0").exists()
        assert (trial.run.replicas, trial.run.walkers, trial.run.seed, trial.metad.pace) == (8, 4, seed, 500)
        protocol = trial.to_dict()
        del protocol["run"]["seed"]
        protocols.append((protocol["run"], protocol["metad"]))
        with pytest.raises(FileExistsError):
            prepare(source, target, "c", 40, seed)
    assert protocols[0] == protocols[1]

def test_selection_waits_for_both_systems_and_requires_independent_confirmation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    study = importlib.import_module("binding_study")
    args = SimpleNamespace(control=tmp_path / "control", output=tmp_path / "scores")
    (args.control / "requests").mkdir(parents=True)
    costs = args.control / "private-costs"
    costs.mkdir()
    measured = set()
    holdouts = set()

    def result(arm, length, confirmation, args):
        if (arm, length) not in (holdouts if confirmation else measured):
            return None
        seed = 100000 if confirmation else 10000
        return dict(arm=arm, length_ns=length, passes=arm in ("b", "c") and length == 40
                    and (arm == "c" or not confirmation), protocol={"replicas": 8, "walkers": 4},
                    systems={"tyk2": {"base_seed": seed}, "a2a": {"base_seed": seed + 20000}})

    monkeypatch.setattr(study, "get_score", result)
    assert study.step(args) is False
    assert not list((args.control / "requests").iterdir())
    measured.update((arm, length) for arm in study.ARMS for length in study.ENDPOINTS)
    for arm, seconds in (("b", 42), ("c", 53)):
        study.save_json(costs / f"{arm}-40.json", {"total_wall_seconds": seconds})
    assert study.step(args) is False
    requests = args.control / "requests"
    assert json.loads((requests / "3000.json").read_text()) == {"action": "validate", "arm": "b", "length_ns": 40}
    assert not (args.output / "decision.json").exists()
    holdouts.add(("b", 40))
    assert study.step(args) is False
    assert json.loads((requests / "3001.json").read_text())["arm"] == "c"
    holdouts.add(("c", 40))
    assert study.step(args) is True
    assert json.loads((args.output / "decision.json").read_text())["selection"]["arm"] == "c"
    assert Config.load(args.output / "common-production.yaml").run.length_ns == 40
    search = result("c", 40, False, args)
    with pytest.raises(ValueError, match="reused a seed"):
        study.confirm_seeds(search, search.copy())


def test_report_converts_prediction_reference_interval_and_signed_error_to_kcal(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    study = importlib.import_module("binding_study")
    trial = {"systems": {"tyk2": {"dG": -41.84, "ci95": [-46.024, -37.656], "reference_kj_mol": -50.208,
                                  "passes": False, "checks": {"affinity_within_fivefold": False},
                                  "returns_by_replica": [1, 2]}}}
    study.save_json(tmp_path / "b-20.json", trial)
    study.final_report(tmp_path, {"selection": None})
    report = (tmp_path / "report.md").read_text()
    assert "kcal/mol" in report
    assert "| -12.00 | -10.00 [-11.00, -9.00] | +2.00 | 1.00 | 3 |" in report
    score = trial["systems"]["tyk2"]
    score["checks"]["all_replicas_available"] = True
    score["replicas"] = [dict(status="available", dG=v) for v in (-50., -45., -40., -35.)]
    study.save_json(tmp_path / "partials/b-tyk2-20.json", trial)
    study.save_json(tmp_path / "strategy-tyk2-first.json", {"screening":{"b-tyk2-20":{"phase":"tyk2_pilot"}}})
    study.final_report(tmp_path, {"selection":None, "strategy":"tyk2-first",
                                  "pilot_ranking":[{"arm":"b", "replica_sd_kj_mol":study.replica_sd(score)}]})
    report = (tmp_path / "report.md").read_text()
    assert "| TYK2 予備計算 | B | 20 |" in report and "| 3 | 適格 | 1.543 |" in report


@pytest.fixture
def adaptive(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    study = importlib.import_module("binding_study")
    args = SimpleNamespace(control=tmp_path / "control", output=tmp_path / "scores", strategy="tyk2-first")
    (args.control / "requests").mkdir(parents=True)
    partials, pairs = {}, {}

    def system(system, arm, length, sd, precise, accurate=False):
        checks = dict(all_replicas_available=sd is not None, binding=True, contact_confirmed_roundtrip=True,
                      ci_halfwidth_within_limit=precise, affinity_within_fivefold=accurate)
        base = 10000 if system == "tyk2" else 200000
        score = dict(checks=checks, passes=all(checks.values()), base_seed=base)
        partials[system, arm, length] = dict(arm=arm, system=system, length_ns=length, replica_sd_kj_mol=sd,
                                              precision_passes=study.precision_passes(score), systems={system: score})

    def pair(arm, length, holdout=None):
        systems = {name: dict(partials[name, arm, length]["systems"][name]) for name in study.EXAMPLES}
        if holdout is not None:
            for score in systems.values():
                score["base_seed"] += 400000
                score["checks"] = dict(score["checks"], affinity_within_fivefold=holdout)
                score["passes"] = all(score["checks"].values())
        result = dict(arm=arm, length_ns=length, passes=all(r["passes"] for r in systems.values()), systems=systems,
                      protocol=dict(zip(("replicas", "walkers"), study.ARMS[arm][:2])))
        pairs[arm, length, holdout is not None] = result
        return result

    monkeypatch.setattr(study, "get_system_score", lambda n, a, l, args: partials.get((n, a, l)))
    monkeypatch.setattr(study, "get_score", lambda a, l, h, args: pairs.get((a, l, h)))

    def requests():
        return {p.stem: json.loads(p.read_text()) for p in (args.control / "requests").iterdir()}

    return SimpleNamespace(study=study, args=args, system=system, pair=pair, requests=requests)


def test_tyk2_first_ignores_screening_affinity_and_stops_at_first_heldout_pair(adaptive):
    a = adaptive
    for arm, sd in (("a", 5.), ("b", 1.), ("c", 1.)):
        # On the SD tie, B wins by arm ID despite its out-of-range screening estimate.
        a.system("tyk2", arm, 20, sd, precise=True, accurate=arm != "b")
    (a.args.control / "d-tyk2-20.done").touch()
    assert not a.study.step(a.args) and not a.requests()  # D's analysis is pending, no duplicate MD.
    a.system("tyk2", "d", 20, None, precise=False)
    assert not a.study.step(a.args)
    assert a.requests() == {"4101": {"action":"screen-system", "arm":"b", "system":"a2a", "length_ns":20}}
    progress = json.loads((a.args.output / "strategy-tyk2-first.json").read_text())
    assert [p["arm"] for p in progress["ranking"]] == ["b", "c", "a", "d"]
    assert progress["phase"] == "a2a_followup"
    assert not a.study.step(a.args) and len(a.requests()) == 1  # restart reuses the same request
    a.system("a2a", "b", 20, 2., precise=True, accurate=False)
    screening = a.pair("b", 20)
    assert not screening["passes"]
    assert not a.study.step(a.args)
    assert a.requests()["9100"] == {"action":"validate", "arm":"b", "length_ns":20}
    a.pair("b", 20, holdout=True)
    assert a.study.step(a.args)
    assert set(a.requests()) == {"4101", "9100", "9999"}
    decision = json.loads((a.args.output / "decision.json").read_text())
    assert decision["selection"]["arm"] == "b"
    assert decision["strategy"] == "tyk2-first"
    assert "標本標準偏差" in (a.args.output / "report.md").read_text()


def test_tyk2_first_catches_up_selected_a2a_then_extends_after_a_failed_holdout(adaptive):
    a = adaptive
    for arm, sd in (("a", None), ("b", 6.), ("c", 1.), ("d", 7.)):
        a.system("tyk2", arm, 20, sd, precise=False)
    assert not a.study.step(a.args) and set(a.requests()) == {"4210"}
    a.system("tyk2", "c", 40, 2., precise=True)
    assert not a.study.step(a.args) and "4201" in a.requests() and "4211" not in a.requests()
    a.system("a2a", "c", 20, 4., precise=False)
    assert not a.study.step(a.args) and "4211" in a.requests()
    a.system("a2a", "c", 40, 1., precise=True)
    a.pair("c", 40)
    assert not a.study.step(a.args) and "9210" in a.requests()
    a.pair("c", 40, holdout=False)
    assert not a.study.step(a.args) and "4220" in a.requests()
    a.system("tyk2", "c", 80, 5., precise=False)
    a.system("tyk2", "c", 100, 5., precise=False)
    assert not a.study.step(a.args) and "4221" in a.requests()  # collect A2A at the computable TYK2 cap
    a.system("a2a", "c", 80, 2., precise=False)
    assert not a.study.step(a.args) and "4231" in a.requests()
    a.system("a2a", "c", 100, 2., precise=True)
    a.pair("c", 100)
    assert not a.study.step(a.args) and "4110" in a.requests()  # next-ranked arm after exhausting C
    assert "9230" not in a.requests() and "9999" not in a.requests()
    assert not any(p.get("system") == "a2a" and p["arm"] != "c" for p in a.requests().values())


def test_pilot_sd_uses_every_independent_primary_estimate_with_n_minus_one(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    study = importlib.import_module("binding_study")
    args = SimpleNamespace(analysis=tmp_path/"analysis", geometry=tmp_path/"geometry", output=tmp_path/"output")
    sources = study.measurement_paths("tyk2", "a", 20, False, args)
    for p in sources:
        study.save_json(p, {})
    checks = dict(all_replicas_available=True, contact_confirmed_roundtrip=True, binding=True,
                  ci_halfwidth_within_limit=False, affinity_within_fivefold=False)
    rows = [dict(status="available", dG=x) for x in (0., 4., 8., 12.)]
    monkeypatch.setattr(study, "score_system", lambda *args: dict(checks=checks, replicas=rows))
    result = study.get_system_score("tyk2", "a", 20, args)
    assert result["replica_sd_kj_mol"] == pytest.approx((80 / 3)**.5)
    rows[-1]["status"] = "unavailable"
    checks["all_replicas_available"] = False
    cache = args.output / "partials/a-tyk2-20.json"
    cache.unlink()
    result = study.get_system_score("tyk2", "a", 20, args)
    assert result["replica_sd_kj_mol"] == pytest.approx((80 / 3)**.5) and not result["precision_passes"]
    study.save_json(cache, dict(result, replica_sd_kj_mol=None))
    assert study.get_system_score("tyk2", "a", 20, args)["replica_sd_kj_mol"] == pytest.approx((80 / 3)**.5)
    rows[-1]["dG"] = None
    cache.unlink()
    assert study.get_system_score("tyk2", "a", 20, args)["replica_sd_kj_mol"] is None
