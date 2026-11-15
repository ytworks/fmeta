#!/usr/bin/env python3
"""Run the GPU gates and record the evidence.

The gates are the GPU tests (``pytest -m gpu``) and every item of ``fmeta check``
(fmeta/check.py: forces against the Reference platform, periodic images, the solvable toy
system, the route from a protein PDB and a ligand SDF, the full pipeline with restart and
walkers, a membrane build and run). On ALL PASSED the result is written to
``bench/results/<date>-<commit>-gpu-gates.json`` and added to ``bench/results/index.json``;
commit both. Procedure: docs/gpu-runbook.md.

``--quick`` and runs on a CPU platform exercise the steps but are never recorded.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
TRACKED = ["src", "tests", "scripts", "examples", "pyproject.toml", "environment.yml"]


def provenance(root: Path = ROOT) -> tuple[str, bool]:
    """Commit hash and whether the code differs from it (including untracked files)."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--", *TRACKED], cwd=root, text=True)
    return commit, bool(status.strip())


def run_gpu_tests(root: Path = ROOT) -> dict:
    """``pytest -m gpu`` as one gate. It passes only if tests actually ran."""
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "gpu", "tests"], cwd=root,
                            capture_output=True, text=True)
    tail = (result.stdout.strip().splitlines() or [""])[-1]
    passed = re.search(r"(\d+) passed", tail)
    n_passed = int(passed.group(1)) if passed else 0
    return dict(name="gpu-tests", passed=result.returncode == 0 and n_passed > 0,
                metrics=dict(summary=tail, n_passed=n_passed),
                criteria="pytest -m gpu: every GPU test passes and at least one runs")


def make_record(items, extra_gates, commit: str, dirty: bool, date: str, gpus: int, gpu_name: str,
                platform: str, openmm_version: str, criteria: dict) -> dict:
    """The evidence record: one gate per check item. A skipped item is not a pass."""
    gates = list(extra_gates)
    for item in items:
        gates.append(dict(name=item.name, passed=item.status == "PASS",
                          metrics=dict(status=item.status, detail=item.detail,
                                       seconds=round(item.seconds, 1)),
                          criteria=criteria[item.name]))   # every gate states its criterion
    return dict(date=date, commit=commit, dirty=dirty, gpus=gpus, gpu_name=gpu_name,
                platform=platform, openmm=openmm_version,
                all_passed=all(gate["passed"] for gate in gates), gates=gates)


def write_record(record: dict, results: Path = RESULTS) -> Path:
    """Write the record and update index.json (a rerun of the same commit replaces its entry)."""
    results.mkdir(parents=True, exist_ok=True)
    name = f"{record['date']}-{record['commit'][:9]}-gpu-gates.json"
    (results / name).write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    index_path = results / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"entries": []}
    entry = dict(file=name, **{key: record[key] for key in
                               ("date", "commit", "dirty", "gpus", "gpu_name", "all_passed")})
    index["entries"] = [e for e in index["entries"] if e["file"] != name] + [entry]
    index_path.write_text(json.dumps(index, indent=1, allow_nan=False) + "\n")
    return results / name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--out", default="gates_work", help="directory for the test runs")
    parser.add_argument("--platform", default="auto", help="OpenMM platform (default: the GPU found)")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--device", type=int, help="GPU device index")
    parser.add_argument("--gpus", help="comma-separated device indices (tests the multi-GPU launcher)")
    parser.add_argument("--quick", action="store_true", help="minimal run lengths; not recorded")
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest -m gpu; not recorded")
    args = parser.parse_args()

    import openmm

    from fmeta.check import CRITERIA, Checker
    from fmeta.engine.platform import GPU_PLATFORMS

    commit, dirty = provenance()
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    extra = []
    if not args.skip_tests:
        gate = run_gpu_tests()
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] gpu-tests            {gate['metrics']['summary']}",
              flush=True)
        extra.append(gate)
    checker = Checker(args.out, args.platform, args.precision, args.device, gpus, args.quick)
    checker.run_all()

    platform = checker.platform.getName() if checker.platform is not None else "none"
    record = make_record(checker.items, extra, commit, dirty,
                         datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                         len(gpus) if gpus else 1, checker.device_name or platform, platform,
                         openmm.__version__, CRITERIA)
    failed = [gate["name"] for gate in record["gates"] if not gate["passed"]]
    recordable = platform in GPU_PLATFORMS and not args.quick and not args.skip_tests
    if record["all_passed"] and recordable:
        path = write_record(record)
        note = " (DIRTY: the code differs from the commit)" if dirty else ""
        print(f"\nALL PASSED{note}. Recorded: {path.relative_to(ROOT)} and bench/results/index.json; "
              "commit both and update PROGRESS.md if the gate status changed.")
        return 0
    out = Path(args.out) / "gpu-gates-not-recorded.json"
    out.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    if failed:
        print(f"\nNOT PASSED: {', '.join(failed)}. Details: {out} and the traceback files beside it.")
        return 1
    print(f"\nAll items passed, but this run is not evidence (quick mode, skipped tests or no GPU "
          f"platform) and was not recorded: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
