"""The result of a sweep is the fastest eligible protocol that passes a fresh paired trial."""

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
