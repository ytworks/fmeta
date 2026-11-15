#!/usr/bin/env python3
"""Validate bench/results/index.json against the result files (run on CPU CI).

Every index entry must point to an existing gate record (scripts/gpu_gates.py) whose commit
and verdicts agree with the entry and that names the image and lock it was taken in; every
record must be indexed; every file is strict JSON (NaN / Infinity are rejected: unmeasured
values are null). This keeps the committed GPU evidence consistent and reviewable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"
ENTRY_KEYS = {"file", "date", "commit", "dirty", "gpus", "gpu_name", "all_passed"}
GATE_KEYS = {"name", "passed", "metrics", "criteria"}
ENVIRONMENT_KEYS = ("image", "lock_sha256")      # evidence comes from the official image only


def loads_strict(text: str):
    def reject(constant):
        raise ValueError(f"non-finite constant {constant}")
    return json.loads(text, parse_constant=reject)


def verify(results: Path) -> tuple[list[str], int]:
    """Return (errors, number of indexed files) for a results directory."""
    index_path = results / "index.json"
    if not index_path.exists():
        return [], -1
    errors: list[str] = []

    def load(path: Path):
        try:
            return loads_strict(path.read_text())
        except ValueError as error:
            errors.append(f"{path.name}: not strict JSON ({error})")
            return None

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
        if not entry["file"].startswith(f"{entry['date']}-{entry['commit'][:9]}-"):
            errors.append(f"{entry['file']}: file name does not carry the date and commit")
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
    errors, count = verify(RESULTS)
    if count < 0:
        print("no bench/results/index.json (no GPU evidence recorded yet)")
        return 0
    for message in errors:
        print(message, file=sys.stderr)
    if not errors:
        print(f"bench/results: {count} indexed result file(s) consistent")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
