#!/usr/bin/env python3
"""Run the GPU gates and record the evidence.

The gates are the GPU tests (``pytest -m gpu``) and every item of ``fmeta check``
(fmeta/check.py: forces against the Reference platform, the solvable toy system, the route
from a protein PDB and a ligand SDF, the CVs while the platform moves the complex between
periodic images, the full pipeline with restart and walkers, a membrane build and run). On
ALL PASSED the result is written to
``bench/results/<date>-<commit>-gpu-gates.json`` and added to ``bench/results/index.json``;
commit both. Procedure: docs/gpu-runbook.md.

A run whose test runs used the cuPotential plugin (``--backend``, FM-D18) is a record of its
own, ``<date>-<commit>-cupotential-gpu-gates.json``: it holds the verdicts and the accuracy, and
no speed and no duration (``PLUGIN_WITHHELD`` of scripts/verify_bench_results.py).

Evidence is recorded only inside the official image built from the repository's lock (run
this script through scripts/in_docker.sh). ``--quick`` and runs on a CPU platform exercise the
steps but are never recorded.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from verify_bench_results import PLUGIN_WITHHELD, record_name

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
LOCK = "conda-linux-64.lock"
IMAGE_LOCK = Path("/opt/fmeta") / LOCK          # the Dockerfile copies the lock it installs here
TRACKED = ["src", "tests", "scripts", "examples", "pyproject.toml", "environment.yml", LOCK,
           "Dockerfile", ".dockerignore", "docker"]


def provenance(root: Path = ROOT) -> tuple[str, bool]:
    """Commit hash and whether the code differs from it (including untracked files)."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--", *TRACKED], cwd=root, text=True)
    return commit, bool(status.strip())


def nvidia_smi(field: str) -> list[str]:
    """One value per GPU this process can see, in index order (empty without nvidia-smi)."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def loaded_libcuda() -> Optional[str]:
    """File name of the CUDA driver library that a process of this container loads.

    Its version is the driver of the host, or that of the compatibility libraries of the image
    when the container runtime has put them first (a host driver older than the CUDA runtime).
    """
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    names = [line.rsplit("/", 1)[-1] for line in Path("/proc/self/maps").read_text().splitlines()]
    return next((name for name in names if name.startswith("libcuda.so")), None)


def environment(root: Path = ROOT, image_lock: Path = IMAGE_LOCK,
                image: Optional[str] = None) -> tuple[dict, bool]:
    """Where the gates run, and whether that is the official image.

    Official means: the lock installed in the image is the lock of the repository, and the
    image is known (scripts/in_docker.sh passes its identifier as FMETA_IMAGE_ID). The record
    also names the CUDA software of the run: the driver of the host, the driver library the
    container loads, and the CUDA runtime and the cuPotential plugin of the lock.
    """
    def digest(path: Path) -> Optional[str]:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def locked(package: str) -> Optional[str]:
        found = re.search(rf"/{package}-([0-9][0-9.]*)-[^/]*$", lock, re.MULTILINE)
        return found.group(1) if found else None

    image = image or os.environ.get("FMETA_IMAGE_ID")
    installed = digest(image_lock)
    official = bool(image) and installed is not None and installed == digest(root / LOCK)
    lock = image_lock.read_text() if image_lock.exists() else ""
    return dict(image=image, lock_sha256=installed, gpu_uuids=nvidia_smi("uuid"),
                driver=next(iter(nvidia_smi("driver_version")), None), libcuda=loaded_libcuda(),
                cuda_runtime=locked("cuda-version"), plugin=locked("nv-openmm-plugin")), official


def run_gpu_tests(root: Path = ROOT) -> dict:
    """``pytest -m gpu`` as one gate. It passes only if tests actually ran."""
    start = time.time()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "gpu", "tests"], cwd=root,
                            capture_output=True, text=True)
    seconds = round(time.time() - start, 1)
    tail = re.sub(r" in [0-9.]+s\b.*$", "", (result.stdout.strip().splitlines() or [""])[-1])
    passed = re.search(r"(\d+) passed", tail)
    n_passed = int(passed.group(1)) if passed else 0
    return dict(name="gpu-tests", passed=result.returncode == 0 and n_passed > 0,
                metrics=dict(summary=tail, n_passed=n_passed, seconds=seconds),
                criteria="pytest -m gpu: every GPU test passes and at least one runs")


def make_record(items, extra_gates, commit: str, dirty: bool, date: str, gpus: int, gpu_name: str,
                platform: str, openmm_version: str, criteria: dict, environment: dict,
                backend: str = "openmm") -> dict:
    """The evidence record: one gate per check item. A skipped item is not a pass.

    ``backend`` names what the test runs were made with. The record of cuPotential leaves out the
    speeds and the durations (``PLUGIN_WITHHELD``); they stay in the directory of the test runs."""
    withheld = PLUGIN_WITHHELD if backend == "cupotential" else ()
    gates = [dict(gate) for gate in extra_gates]
    for item in items:
        metrics = dict(status=item.status, detail=item.detail, seconds=round(item.seconds, 1))
        if item.performance:
            metrics["performance"] = item.performance
        gates.append(dict(name=item.name, passed=item.status == "PASS", metrics=metrics,
                          criteria=criteria[item.name]))   # every gate states its criterion
    for gate in gates:
        gate["metrics"] = {key: value for key, value in gate["metrics"].items() if key not in withheld}
    return dict(date=date, commit=commit, dirty=dirty, gpus=gpus, gpu_name=gpu_name,
                platform=platform, openmm=openmm_version, backend=backend, environment=environment,
                all_passed=all(gate["passed"] for gate in gates), gates=gates)


def write_record(record: dict, results: Path = RESULTS) -> Path:
    """Write the record and update index.json. A rerun of the same commit with the same back end
    replaces its entry; the two back ends of one commit are two records."""
    results.mkdir(parents=True, exist_ok=True)
    name = record_name(record)
    (results / name).write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    index_path = results / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"entries": []}
    entry = dict(file=name, **{key: record[key] for key in
                               ("date", "commit", "dirty", "gpus", "gpu_name", "all_passed", "backend")})
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
    parser.add_argument("--backend", choices=("auto", "cupotential", "openmm"),
                        help="the back end of the test runs (default: that of a new configuration); "
                             "a run with cupotential and one with openmm are two records")
    args = parser.parse_args()

    import openmm

    from fmeta.check import CRITERIA, Checker
    from fmeta.engine.platform import GPU_PLATFORMS

    commit, dirty = provenance()
    where, official = environment()
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    extra = []
    if not args.skip_tests:
        gate = run_gpu_tests()
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] gpu-tests            {gate['metrics']['summary']}",
              flush=True)
        extra.append(gate)
    checker = Checker(args.out, args.platform, args.precision, args.device, gpus, args.quick,
                      backend=args.backend)
    checker.run_all()

    platform = checker.platform.getName() if checker.platform is not None else "none"
    record = make_record(checker.items, extra, commit, dirty,
                         datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                         len(gpus) if gpus else 1, checker.device_name or platform, platform,
                         openmm.__version__, CRITERIA, where, backend=checker.backend_used)
    failed = [gate["name"] for gate in record["gates"] if not gate["passed"]]
    recordable = platform in GPU_PLATFORMS and not args.quick and not args.skip_tests and official
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
    reason = ("quick mode, skipped tests or no GPU platform" if official else
              "not inside the official image built from this repository's lock: use scripts/in_docker.sh")
    print(f"\nAll items passed, but this run is not evidence ({reason}) and was not recorded: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
