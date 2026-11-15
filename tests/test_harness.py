"""The development harness: progress-log budget, and GPU evidence that the recorder writes
and the CI verifier accepts (and rejects once it is inconsistent)."""

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


def test_progress_log_budget():
    entry = "### 2026-01-01 — [progress] x\n- What: y\n"
    good = "# Progress\n\n## Product\n\n## Current Status\n\n## Change Log\n\n" + entry * 10
    assert check_progress_size.problems_in(good) == []
    assert any("11 Change Log entries" in p for p in check_progress_size.problems_in(good + entry))
    assert any("sections" in p for p in check_progress_size.problems_in(good.replace("## Product\n", "")))
    assert any("lines" in p for p in check_progress_size.problems_in(good + "x\n" * 200))


def record(items):
    criteria = {item.name: f"criterion of {item.name}" for item in items}
    tests = dict(name="gpu-tests", passed=True, metrics=dict(summary="2 passed", n_passed=2),
                 criteria="pytest -m gpu")
    return gpu_gates.make_record(items, [tests], COMMIT, False, "2026-11-15", 2, "Test GPU", "CUDA",
                                 "8.6.1", criteria)


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
        gpu_gates.make_record(passed, [], COMMIT, False, "2026-11-15", 1, "g", "CUDA", "8", {})

    # the verifier notices a record edited after the fact, and files that are not indexed
    data = json.loads(path.read_text())
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
