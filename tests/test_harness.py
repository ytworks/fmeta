"""The development harness: progress-log budget, and GPU evidence that the recorder writes
and the CI verifier accepts (and rejects once it is inconsistent)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_progress_size  # noqa: E402
import gpu_gates  # noqa: E402
import verify_bench_results  # noqa: E402

from fmeta.check import Item  # noqa: E402

COMMIT = "0123456789abcdef0123456789abcdef01234567"
WHERE = dict(image="sha256:" + "a" * 64, lock_sha256="b" * 64, gpu_uuids=["GPU-0", "GPU-1"])


def test_progress_log_budget():
    entry = "### 2026-01-01 — [progress] x\n- What: y\n"
    good = "# Progress\n\n## Product\n\n## Current Status\n\n## Change Log\n\n" + entry * 10
    assert check_progress_size.problems_in(good) == []
    assert any("11 Change Log entries" in p for p in check_progress_size.problems_in(good + entry))
    assert any("sections" in p for p in check_progress_size.problems_in(good.replace("## Product\n", "")))
    assert any("lines" in p for p in check_progress_size.problems_in(good + "x\n" * 200))


def record(items, backend="openmm"):
    criteria = {item.name: f"criterion of {item.name}" for item in items}
    tests = dict(name="gpu-tests", passed=True, metrics=dict(summary="2 passed", n_passed=2, seconds=5.0),
                 criteria="pytest -m gpu")
    return gpu_gates.make_record(items, [tests], COMMIT, False, "2026-11-15", 2, "Test GPU", "CUDA",
                                 "8.6.1", criteria, WHERE, backend=backend)


def test_recorded_evidence_is_accepted_and_tampering_is_caught(tmp_path):
    passed = [Item("forces", "PASS", "ok", 1.0), Item("toy_free_energy", "PASS", "ok", 60.0)]
    good = record(passed)
    assert good["all_passed"] and len(good["gates"]) == 3
    path = gpu_gates.write_record(good, tmp_path)
    assert path.name == "2026-11-15-012345678-gpu-gates.json"
    assert verify_bench_results.verify(tmp_path) == ([], 1)
    gpu_gates.write_record(good, tmp_path)                       # a rerun replaces its entry
    assert len(json.loads((tmp_path / "index.json").read_text())["entries"]) == 1

    # a skipped item (e.g. the ligand generators missing) is not a pass
    skipped = record(passed + [Item("pdb_sdf_openff", "SKIP", "not installed", 0.0)])
    assert not skipped["all_passed"]
    with pytest.raises(KeyError):                                # every gate needs a stated criterion
        gpu_gates.make_record(passed, [], COMMIT, False, "2026-11-15", 1, "g", "CUDA", "8", {}, WHERE)

    # evidence must name the image it was taken in
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "environment": {**WHERE, "image": None}}))
    errors, _ = verify_bench_results.verify(tmp_path)
    assert any("no environment" in e for e in errors)

    # the verifier notices a record edited after the fact, and files that are not indexed
    data["gates"][1]["passed"] = False
    path.write_text(json.dumps(data))
    errors, _ = verify_bench_results.verify(tmp_path)
    assert any("all_passed does not match" in e for e in errors)
    (tmp_path / "2026-11-16-fffffffff-gpu-gates.json").write_text(json.dumps(good))
    errors, _ = verify_bench_results.verify(tmp_path)
    assert any("not listed in index.json" in e for e in errors)
    path.write_text('{"all_passed": NaN}')
    errors, _ = verify_bench_results.verify(tmp_path)
    assert any("not strict JSON" in e for e in errors)
    assert verify_bench_results.verify(tmp_path / "empty") == ([], -1)   # no evidence yet is fine


def test_the_two_back_ends_of_a_commit_are_two_records_and_the_one_of_cupotential_holds_no_speed(tmp_path):
    """A speed measured with the cuPotential plugin goes to no tracked file (owner, 2026-11-17)."""
    items = [Item("forces", "PASS", "energy 7e-06", 1.0),
             Item("benchmark", "PASS", "24062 atoms", 9.0, "1410 ns/day plain")]
    stock, plugin = (gpu_gates.write_record(record(items, backend), tmp_path) for backend in ("openmm", "cupotential"))
    assert plugin.name == "2026-11-15-012345678-cupotential-gpu-gates.json" and stock.name != plugin.name
    assert verify_bench_results.verify(tmp_path) == ([], 2)
    kept = json.loads(stock.read_text())["gates"][2]["metrics"]
    assert kept["performance"] == "1410 ns/day plain" and kept["seconds"] == 9.0
    data = json.loads(plugin.read_text())
    assert [gate["metrics"] for gate in data["gates"]] == [
        dict(summary="2 passed", n_passed=2), dict(status="PASS", detail="energy 7e-06"),
        dict(status="PASS", detail="24062 atoms")]

    def rejected(change, message="holds no speed"):
        edited = json.loads(plugin.read_text())
        change(edited)
        plugin.write_text(json.dumps(edited))
        errors, _ = verify_bench_results.verify(tmp_path)
        plugin.write_text(json.dumps(data))
        return any(message in error for error in errors)

    assert rejected(lambda edited: edited["gates"][2]["metrics"].update(performance="as fast as it gets"))
    assert rejected(lambda edited: edited["gates"][1]["metrics"].update(seconds=1.0))
    assert rejected(lambda edited: edited["gates"][2]["metrics"].update(detail="24062 atoms, 1410 ns/day"))
    # ... and a record cannot be passed off as one of the other back end
    assert rejected(lambda edited: edited.update(backend="openmm"), "backend differs")


def test_only_the_image_built_from_the_repository_lock_is_official(tmp_path, monkeypatch):
    monkeypatch.delenv("FMETA_IMAGE_ID", raising=False)
    monkeypatch.setattr(gpu_gates, "nvidia_smi", dict(uuid=["GPU-0"], driver_version=["595.99.02"]).get)
    monkeypatch.setattr(gpu_gates, "loaded_libcuda", lambda: "libcuda.so.615.71.10")
    repo, installed = tmp_path / "repo", tmp_path / "image" / gpu_gates.LOCK
    repo.mkdir()
    installed.parent.mkdir()
    lock = ("@EXPLICIT\nhttps://c/noarch/cuda-version-13.4-h1_3.conda#0\n"
            "https://c/linux-64/openmm-8.4.0-py312h2_2.conda#1\n"
            "https://n/v0.3.5/nv-openmm-plugin-0.3.5-py312_0.conda#2\n")
    (repo / gpu_gates.LOCK).write_text(lock)
    assert not gpu_gates.environment(repo, installed, "sha256:1")[1]     # not inside an image
    installed.write_text(lock.replace("openmm-8.4.0", "openmm-8.6.1"))
    assert not gpu_gates.environment(repo, installed, "sha256:1")[1]     # an image of another lock
    installed.write_text(lock)
    assert not gpu_gates.environment(repo, installed)[1]                 # the image is not known
    where, official = gpu_gates.environment(repo, installed, "sha256:1")
    assert official
    # the record names the CUDA software too: a host driver older than the runtime of the lock
    # works through the compatibility libraries of the image, and the record has to show it
    assert where == dict(image="sha256:1", gpu_uuids=["GPU-0"],
                         lock_sha256=hashlib.sha256(installed.read_bytes()).hexdigest(),
                         driver="595.99.02", libcuda="libcuda.so.615.71.10",
                         cuda_runtime="13.4", plugin="0.3.5")
