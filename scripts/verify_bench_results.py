#!/usr/bin/env python3
"""Validate the indexes of bench/results and bench/reports against their files (run on CPU CI).

bench/results: every index entry must point to an existing gate record (scripts/gpu_gates.py)
whose commit and verdicts agree with the entry and that names the image and lock it was taken
in; every record must be indexed.

bench/reports: every index entry must point to a benchmark report ('fmeta benchmark report')
that carries the same set, date and commit, names the image and lock, lists its ligands and
has its text beside it; every report must be indexed. A report states what was measured: a
pass / fail field in it is an error.

Every file is strict JSON (NaN / Infinity are rejected: unmeasured values are null). This
keeps what is committed consistent and reviewable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"
REPORTS = Path(__file__).resolve().parent.parent / "bench" / "reports"
ENTRY_KEYS = {"file", "date", "commit", "dirty", "gpus", "gpu_name", "all_passed"}
GATE_KEYS = {"name", "passed", "metrics", "criteria"}
ENVIRONMENT_KEYS = ("image", "lock_sha256")      # evidence comes from the official image only
# A record of gates whose test runs used the cuPotential plugin holds verdicts and accuracy, and
# no speed: the licence of the plugin (2.12) asks for the permission of NVIDIA before one is made
# public (owner, 2026-11-17). These metrics are left out of such a record (scripts/gpu_gates.py)
# and rejected in one here, as is a speed written into a text. With the permission, empty the
# tuple; to keep the accuracy out as well, add "detail".
PLUGIN_WITHHELD = ("performance", "seconds")
SPEED_UNITS = ("ns/day",)
REPORT_KEYS = ("date", "commit", "dirty", "set")                 # the same in the index and in the report
VERDICT_KEYS = {"pass", "passed", "fail", "failed", "all_passed"}


def loads_strict(text: str):
    def reject(constant):
        raise ValueError(f"non-finite constant {constant}")
    return json.loads(text, parse_constant=reject)


def _load(path: Path, errors: list):
    try:
        return loads_strict(path.read_text())
    except ValueError as error:
        errors.append(f"{path.name}: not strict JSON ({error})")
        return None


def record_name(record: dict) -> str:
    """The file of a gate record: one per date, commit and back end (no entry: OpenMM)."""
    backend = record.get("backend", "openmm")
    label = "" if backend == "openmm" else f"{backend}-"
    return f"{record.get('date')}-{str(record.get('commit'))[:9]}-{label}gpu-gates.json"


def _keys(value):
    """Every key of a JSON value, at any depth."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _texts(value):
    """Every text of a JSON value, at any depth."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, (dict, list)):
        for item in (value.values() if isinstance(value, dict) else value):
            yield from _texts(item)


def verify_reports(reports: Path) -> tuple[list[str], int]:
    """Return (errors, number of indexed reports) for a directory of benchmark reports."""
    index_path = reports / "index.json"
    if not index_path.exists():
        return [], -1
    errors: list[str] = []
    index = _load(index_path, errors) or {}
    seen = set()
    for entry in index.get("entries", []):
        missing = {"file", *REPORT_KEYS} - set(entry)
        if missing:
            errors.append(f"index entry {entry.get('file')}: missing keys {sorted(missing)}")
            continue
        name = entry["file"]
        seen.add(name)
        if not (reports / name).exists():
            errors.append(f"index entry {name}: file not found")
            continue
        report = _load(reports / name, errors)
        if report is None:
            continue
        for key in REPORT_KEYS:
            if report.get(key) != entry[key]:
                errors.append(f"{name}: {key} differs between index ({entry[key]}) and file ({report.get(key)})")
        if name != f"{entry['date']}-{str(entry['commit'])[:9]}-{entry['set']}.json":
            errors.append(f"{name}: file name is not <date>-<commit>-<set>.json")
        if not (reports / name).with_suffix(".md").exists():
            errors.append(f"{name}: the text of the report (.md) is not beside it")
        where = report.get("environment") or {}
        if not all(where.get(key) for key in ENVIRONMENT_KEYS):
            errors.append(f"{name}: no environment (image and lock of the official image)")
        if not report.get("ligands"):
            errors.append(f"{name}: no ligands")
        verdicts = VERDICT_KEYS & set(_keys(report))
        if verdicts:
            errors.append(f"{name}: a report holds no verdict, but has {sorted(verdicts)}")
    for path in reports.glob("*.json"):
        if path.name != "index.json" and path.name not in seen:
            errors.append(f"{path.name}: not listed in index.json")
    return errors, len(seen)


def verify(results: Path) -> tuple[list[str], int]:
    """Return (errors, number of indexed files) for a results directory."""
    index_path = results / "index.json"
    if not index_path.exists():
        return [], -1
    errors: list[str] = []

    def load(path: Path):
        return _load(path, errors)

    index = load(index_path) or {}
    seen = set()
    for entry in index.get("entries", []):
        missing = ENTRY_KEYS - set(entry)
        if missing:
            errors.append(f"index entry {entry.get('file')}: missing keys {sorted(missing)}")
            continue
        seen.add(entry["file"])
        path = results / entry["file"]
        if not path.exists():
            errors.append(f"index entry {entry['file']}: file not found")
            continue
        record = load(path)
        if record is None:
            continue
        for key in ("date", "commit", "dirty", "gpus", "gpu_name", "all_passed"):
            if record.get(key) != entry[key]:
                errors.append(f"{entry['file']}: {key} differs between index ({entry[key]}) "
                              f"and file ({record.get(key)})")
        backend = record.get("backend", "openmm")
        if entry.get("backend", "openmm") != backend:
            errors.append(f"{entry['file']}: backend differs between index "
                          f"({entry.get('backend', 'openmm')}) and file ({backend})")
        if entry["file"] != record_name(record):
            errors.append(f"{entry['file']}: file name is not <date>-<commit>-gpu-gates.json, with the "
                          "back end before gpu-gates when it is not openmm")
        if backend == "cupotential":
            held = sorted(set(PLUGIN_WITHHELD) & set(_keys(record)))
            if held or any(unit in text for text in _texts(record) for unit in SPEED_UNITS):
                errors.append(f"{entry['file']}: a record of the cuPotential back end holds no speed and no "
                              f"duration, but has {held or 'a speed in a text'}")
        where = record.get("environment") or {}
        if not all(where.get(key) for key in ENVIRONMENT_KEYS):
            errors.append(f"{entry['file']}: no environment (image and lock of the official image)")
        gates = record.get("gates", [])
        if not gates:
            errors.append(f"{entry['file']}: no gates")
        for gate in gates:
            if GATE_KEYS - set(gate):
                errors.append(f"{entry['file']}: gate {gate.get('name')} missing keys "
                              f"{sorted(GATE_KEYS - set(gate))}")
        if record.get("all_passed") != all(gate.get("passed") for gate in gates):
            errors.append(f"{entry['file']}: all_passed does not match the gate verdicts")
    for path in results.glob("*-gpu-gates.json"):
        if path.name not in seen:
            errors.append(f"{path.name}: not listed in index.json")
    return errors, len(seen)


def main() -> int:
    failed = False
    for name, (errors, count), what, nothing in (
            ("bench/results", verify(RESULTS), "result file(s)", "no GPU evidence recorded yet"),
            ("bench/reports", verify_reports(REPORTS), "benchmark report(s)", "no benchmark report recorded yet")):
        if count < 0:
            print(f"no {name}/index.json ({nothing})")
            continue
        for message in errors:
            print(f"{name}: {message}", file=sys.stderr)
        if not errors:
            print(f"{name}: {count} indexed {what} consistent")
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
