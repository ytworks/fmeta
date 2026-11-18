"""Advance the paired eight-GPU search from complete energy and trajectory measurements.

Run in the official CPU container with the production dispatcher's control directory
mounted writable. Private performance records are read only for ranking. The emitted
scientific results contain no timings. Rerunning after interruption reuses the same requests.
"""

import argparse
import fcntl
import json
import math
import os
import tempfile
import time
from pathlib import Path

import yaml
from score_binding import ARMS, ENDPOINTS, EXAMPLES, score_pair

from fmeta.analysis import KCAL


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def submit(control, order, action, arm, length):
    payload = {"action": action, "arm": arm, "length_ns": length}
    directory = control / "requests"
    path = directory / f"{order:04d}.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(payload, f)
            f.close()
            try:
                os.link(temp, path)
            except FileExistsError:
                if json.loads(path.read_text()) != payload:
                    raise ValueError(f"request {order} conflicts with an existing decision") from None
        finally:
            temp.unlink(missing_ok=True)


def get_score(arm, length, confirmation, args):
    label = "confirm-" + arm if confirmation else arm
    score_file = args.output / f"{label}-{length}.json"
    archived = arm == "a" and length < 100 and not confirmation
    paths = {}
    for name in EXAMPLES:
        energy_path = (args.analysis / "baseline" / f"{name}-{length}.json" if archived else
                       args.analysis / "endpoints" / f"{label}-{name}-{length}.json")
        structure_path = args.geometry / f"{name}-{'r2' if archived else label}-{length}.json"
        if not energy_path.exists() or not structure_path.exists():
            return None
        paths[name] = (energy_path, structure_path)
    if score_file.exists() and all(p.stat().st_mtime_ns <= score_file.stat().st_mtime_ns
                                   for files in paths.values() for p in files):
        return json.loads(score_file.read_text())
    pairs = {name: [json.loads(p.read_text()) for p in files] for name, files in paths.items()}
    result = score_pair(arm, length, pairs)
    save_json(score_file, result)
    return result


def confirm_seeds(search, confirmation):
    if search["protocol"] != confirmation["protocol"] or search["length_ns"] != confirmation["length_ns"]:
        raise ValueError("validation conditions differ from screening")
    protocol = search["protocol"]
    used = set()
    for trial in (search, confirmation):
        for system in EXAMPLES:
            base = trial["systems"][system]["base_seed"]
            if base == 0 and trial is search:
                continue  # archived A used fresh OpenMM velocities without an explicit seed
            seeds = {base + 1000*i + j for i in range(protocol["replicas"]) for j in range(protocol["walkers"])}
            if base == 0 or seeds.intersection(used):
                raise ValueError("independent validation reused a seed")
            used.update(seeds)


def final_report(output, decision):
    """Write the measured scientific results and the common example settings only if validated."""
    output.mkdir(parents=True, exist_ok=True)
    winner = decision["selection"]
    title = "TYK2・A2A 共通設定の実験結果"
    if winner:
        arm, length = winner["arm"], winner["length_ns"]
        nr, nw, ss, sr, pace = ARMS[arm]
        common = dict(metad=dict(height=1.5, bias_factor=10., sigma_s=ss, sigma_rho=sr),
                      run=dict(length_ns=length, replicas=nr, walkers=nw, pace_ps=pace, timestep_fs=4.,
                               walker_sync_hills=1, platform="CUDA", backend="cupotential",
                               mode="parallel", gpus="all", per_gpu=4, precision="mixed",
                               checkpoint_ps=250., trajectory_ps=100., trajectory_solute_only=True))
        (output / "common-production.yaml").write_text(yaml.safe_dump(common, sort_keys=False))
        conclusion = f"独立検証で候補 {arm.upper()}、{length} ns/walker が両系の精度・区間幅の条件を満たした。"
    else:
        conclusion = "100 ns/walker までに両系の採用条件を満たした共通設定は得られなかった。"
    table = [f"# {title}", "", conclusion, "",
             "エネルギーは kcal/mol。実験との差は予測 ΔG° から実験参照を引いた値。"
             "誤差の絶対値と独立レプリカの95%区間の半幅は、ともに RT ln 5 以下（300 K で約 0.96 kcal/mol）が条件。", "",
             "| 段階 | 候補 | ns/walker | 系 | 実験参照 | ΔG° [95% 区間] | 実験との差 | 半幅 | 復帰数 | 判定 |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for phase in ("", "confirm-"):
        for length in ENDPOINTS:
            for arm in ARMS:
                path = output / f"{phase}{arm}-{length}.json"
                if not path.is_file():
                    continue
                score = json.loads(path.read_text())
                for system, r in score["systems"].items():
                    reference = r["reference_kj_mol"] / KCAL
                    estimate, gap, halfwidth = "全群の推定不可", "—", "—"
                    if r["ci95"] is not None:
                        dg = r["dG"] / KCAL
                        lo, hi = (v / KCAL for v in r["ci95"])
                        estimate = f"{dg:.2f} [{lo:.2f}, {hi:.2f}]"
                        gap, halfwidth = f"{dg - reference:+.2f}", f"{(hi - lo) / 2:.2f}"
                    checks = "適格" if r["passes"] else ", ".join(k for k, v in r["checks"].items() if not v)
                    table.append(f"| {'独立検証' if phase else '探索'} | {arm.upper()} | {length} | {system} | "
                                 f"{reference:.2f} | {estimate} | {gap} | {halfwidth} | "
                                 f"{sum(r['returns_by_replica'])} | {checks} |")
    table.extend(["", "判定には各系の全独立レプリカを使用。探索と独立検証を別々に報告した。",
                  "参照は TYK2 の Ki 4.8 nM と A2A の Kd 約 1.64 µM。A2A の実験は 10 °C、本計算は 300 K。"])
    (output / "report.md").write_text("\n".join(table) + "\n")


def step(args):
    # Advance all four candidates through the same prespecified checkpoints. A's first
    # three checkpoints already exist. Costs and independent seeds never influence the
    # integration windows, estimator, or numerical acceptance conditions.
    candidates = []
    for idx, length in enumerate(ENDPOINTS):
        scores = {}
        for n, arm in enumerate(ARMS):
            if idx and (arm != "a" or length == 100):
                submit(args.control, 1000 + 10*idx + n, "extend", arm, length)
            scores[arm] = get_score(arm, length, False, args)
        if any(v is None for v in scores.values()):
            return False
        candidates.extend(s for s in scores.values() if s["passes"])

    ranked = []
    for trial in candidates:
        arm, length = trial["arm"], trial["length_ns"]
        label = arm
        if arm == "a":
            # R2 restarted every 10 ns. Obtain a comparable standalone two-system
            # measurement for A before it can enter the ranking of successful protocols.
            submit(args.control, 2000 + ENDPOINTS.index(length), "time", arm, length)
            label = "timed-a"
        path = args.control / "private-costs" / f"{label}-{length}.json"
        if not path.exists():
            return False
        seconds = json.loads(path.read_text())["total_wall_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"invalid measured cost for {arm}/{length}")
        ranked.append((seconds, length, arm, trial))
    winner = None
    for order, (_, length, arm, trial) in enumerate(sorted(ranked)):
        submit(args.control, 3000 + order, "validate", arm, length)
        confirmation = get_score(arm, length, True, args)
        if confirmation is None:
            return False
        confirm_seeds(trial, confirmation)
        if confirmation["passes"]:
            winner = confirmation
            break
    decision = dict(status="validated" if winner else "no_validated_protocol",
                    selection=winner, screened_candidates=[f"{s['arm']}-{s['length_ns']}" for s in candidates])
    final_report(args.output, decision)
    save_json(args.output / "decision.json", decision)
    submit(args.control, 9999, "finish", "none", 0)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "geometry", "analysis", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not step(args) and not args.once:
            time.sleep(30)


if __name__ == "__main__":
    main()
