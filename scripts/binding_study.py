"""Advance the paired eight-GPU search from complete energy and trajectory measurements.

Run in the official CPU container with the production dispatcher's control directory
mounted writable. The paired sweep uses private performance records. TYK2-first ranks
the four TYK2 pilots by independent-replica scatter and stops at the first validated pair.
Rerunning after interruption reuses the same requests.
"""

import argparse
import fcntl
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

import yaml
from score_binding import ARMS, ENDPOINTS, EXAMPLES, precision_passes, score_pair, score_system

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


def submit(control, order, action, arm, length, system=None):
    payload = {"action": action, "arm": arm, "length_ns": length}
    if system is not None:
        payload["system"] = system
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


def measurement_paths(name, arm, length, confirmation, args):
    archived = arm == "a" and length < 100 and not confirmation
    label = "confirm-" + arm if confirmation else arm
    energy = (args.analysis / "baseline" / f"{name}-{length}.json" if archived else
              args.analysis / "endpoints" / f"{label}-{name}-{length}.json")
    structure = args.geometry / f"{name}-{'r2' if archived else label}-{length}.json"
    return energy, structure


def get_score(arm, length, confirmation, args):
    label = "confirm-" + arm if confirmation else arm
    score_file = args.output / f"{label}-{length}.json"
    paths = {name: measurement_paths(name, arm, length, confirmation, args) for name in EXAMPLES}
    if not all(p.is_file() for files in paths.values() for p in files):
        return None
    if score_file.exists() and all(p.stat().st_mtime_ns <= score_file.stat().st_mtime_ns
                                   for files in paths.values() for p in files):
        return json.loads(score_file.read_text())
    pairs = {name: [json.loads(p.read_text()) for p in files] for name, files in paths.items()}
    result = score_pair(arm, length, pairs)
    save_json(score_file, result)
    return result


def get_system_score(name, arm, length, args):
    """A scored system at one cutoff, with sample SD of all independent primary estimates."""
    paths = measurement_paths(name, arm, length, False, args)
    if not all(p.is_file() for p in paths):
        return None
    cache = args.output / "partials" / f"{arm}-{name}-{length}.json"
    if cache.is_file() and all(p.stat().st_mtime_ns <= cache.stat().st_mtime_ns for p in paths):
        return json.loads(cache.read_text())
    energy, structure = (json.loads(p.read_text()) for p in paths)
    score = score_system(name, arm, length, energy, structure)
    result = dict(arm=arm, length_ns=length, system=name, phase="screening", replica_sd_kj_mol=replica_sd(score),
                  precision_passes=precision_passes(score), systems={name: score})
    save_json(cache, result)
    return result


def replica_sd(score):
    """Sample SD in kJ/mol from every independently measured primary ΔG."""
    if not score["checks"]["all_replicas_available"] or len(score["replicas"]) < 2:
        return None
    if any(r["status"] != "available" or r["dG"] is None or not math.isfinite(r["dG"]) for r in score["replicas"]):
        return None
    return statistics.stdev(r["dG"] for r in score["replicas"])


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
    adaptive = decision.get("strategy") == "tyk2-first"
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
    elif adaptive:
        conclusion = "TYK2 を先行した候補探索では、両系で独立検証の採用条件を満たした共通設定は得られなかった。"
    else:
        conclusion = "100 ns/walker までに両系の採用条件を満たした共通設定は得られなかった。"
    table = [f"# {title}", "", conclusion, ""]
    if adaptive:
        table.extend(["20 ns の TYK2 全独立レプリカの標本標準偏差で優先順を決めた。同値は候補名順。"
                      "探索ではサンプルの支持、結合、ポケット復帰、95% 区間の半幅を評価した。"
                      "実験参照との一致は新しい seed による独立検証で判定した。", "",
                      "| TYK2 の優先順 | 候補 | 標本標準偏差 (kcal/mol) |", "|---:|---|---:|"])
        for index, pilot in enumerate(decision["pilot_ranking"], 1):
            sd = pilot["replica_sd_kj_mol"]
            displayed = f"{sd / KCAL:.3f}" if sd is not None else "全群の推定不可"
            table.append(f"| {index} | {pilot['arm'].upper()} | {displayed} |")
        table.append("")
    table.extend(["エネルギーは kcal/mol。実験との差は予測 ΔG° から実験参照を引いた値。"
                  "独立検証では誤差の絶対値と独立レプリカの95%区間の半幅がともに RT ln 5 以下"
                  "（300 K で約 0.96 kcal/mol）であることを求めた。", "",
                  "| 段階 | 候補 | ns/walker | 系 | 実験参照 | ΔG° [95% 区間] | 実験との差 | 半幅 | 復帰数 | 判定 |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|---|"])
    phases = {}
    if adaptive:
        phases = json.loads((output / "strategy-tyk2-first.json").read_text())["screening"]
        table[-2] += " レプリカ SD |"
        table[-1] += "---:|"
    for phase in ("", "confirm-"):
        for length in ENDPOINTS:
            for arm in ARMS:
                path = output / f"{phase}{arm}-{length}.json"
                systems = json.loads(path.read_text())["systems"] if path.is_file() else {}
                if adaptive and not phase:
                    for system in EXAMPLES:
                        partial = output / "partials" / f"{arm}-{system}-{length}.json"
                        if partial.is_file():
                            systems.update(json.loads(partial.read_text())["systems"])
                for system, r in systems.items():
                    reference = r["reference_kj_mol"] / KCAL
                    estimate, gap, halfwidth = "全群の推定不可", "—", "—"
                    if r["ci95"] is not None:
                        dg = r["dG"] / KCAL
                        lo, hi = (v / KCAL for v in r["ci95"])
                        estimate = f"{dg:.2f} [{lo:.2f}, {hi:.2f}]"
                        gap, halfwidth = f"{dg - reference:+.2f}", f"{(hi - lo) / 2:.2f}"
                    failures = [k for k, v in r["checks"].items() if not v and
                                (phase or not adaptive or k != "affinity_within_fivefold")]
                    checks = ", ".join(failures) if failures else "適格"
                    stage = "独立検証" if phase else "探索"
                    sd_cell = ""
                    if adaptive:
                        if not phase:
                            measured = phases.get(f"{arm}-{system}-{length}", {})
                            stage = {"tyk2_pilot": "TYK2 予備計算", "tyk2_screening": "TYK2 延長",
                                     "a2a_followup": "A2A 評価"}.get(measured.get("phase"), "移行前の探索")
                        sd = replica_sd(r)
                        sd_cell = f" {sd / KCAL:.3f} |" if sd is not None else " — |"
                    table.append(f"| {stage} | {arm.upper()} | {length} | {system} | "
                                 f"{reference:.2f} | {estimate} | {gap} | {halfwidth} | "
                                 f"{sum(r['returns_by_replica'])} | {checks} |{sd_cell}")
    table.extend(["", "判定には各系の全独立レプリカを使用。探索と独立検証を別々に報告した。",
                  "参照は TYK2 の Ki 4.8 nM と A2A の Kd 約 1.64 µM。A2A の実験は 10 °C、本計算は 300 K。"])
    (output / "report.md").write_text("\n".join(table) + "\n")


def finish(args, decision):
    final_report(args.output, decision)
    save_json(args.output / "decision.json", decision)
    submit(args.control, 9999, "finish", "none", 0)
    return True


def paired_sweep(args):
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
    return finish(args, decision)


def screen_system(args, state, phase, system, arm, length):
    """Reuse completed runs and enqueue a single system, including shorter checkpoint prerequisites."""
    score = get_system_score(system, arm, length, args)
    if score is not None:
        key = f"{arm}-{system}-{length}"
        state["screening"].setdefault(key, dict(phase=phase, system=system, arm=arm, length_ns=length,
                                               precision_passes=score["precision_passes"],
                                               replica_sd_kj_mol=score["replica_sd_kj_mol"]))
        return score
    archived = arm == "a" and length < 100
    completed = (args.control / f"{arm}-{system}-{length}.done").exists()
    if not archived and not completed:
        arm_index = list(ARMS).index(arm)
        order = 4000 + 100*arm_index + 10*ENDPOINTS.index(length) + list(EXAMPLES).index(system)
        submit(args.control, order, "screen-system", arm, length, system)
    state["pending"].append(dict(phase=phase, system=system, arm=arm, length_ns=length,
                                 awaiting="analysis" if archived or completed else "production_and_analysis"))
    return None


def tyk2_first(args):
    progress = args.output / "strategy-tyk2-first.json"
    previous = json.loads(progress.read_text()) if progress.is_file() else {}
    state = dict(strategy="tyk2-first", phase="tyk2_pilots", ranking=previous.get("ranking", []),
                 screening={}, pending=[], paired_screening=[], validation=[])

    def wait():
        if previous != state:
            save_json(progress, state)
        return False

    pilots = [screen_system(args, state, "tyk2_pilot", "tyk2", arm, ENDPOINTS[0]) for arm in ARMS]
    if any(result is None for result in pilots):
        return wait()
    if not state["ranking"]:
        def pilot_order(pilot):
            sd = pilot["replica_sd_kj_mol"]
            return sd if sd is not None else math.inf, pilot["arm"]

        state["ranking"] = [dict(arm=p["arm"], replica_sd_kj_mol=p["replica_sd_kj_mol"])
                            for p in sorted(pilots, key=pilot_order)]
        # Pin the pilot ranking before issuing a request on its basis, including over a restart.
        save_json(progress, state)

    eligible = []
    winner = None
    for pilot in state["ranking"]:
        arm = pilot["arm"]
        for index, length in enumerate(ENDPOINTS):
            state["phase"] = "tyk2_screening"
            tyk2 = screen_system(args, state, "tyk2_screening", "tyk2", arm, length)
            if tyk2 is None:
                return wait()
            at_cap_with_estimate = length == ENDPOINTS[-1] and tyk2["replica_sd_kj_mol"] is not None
            if not tyk2["precision_passes"] and not at_cap_with_estimate:
                continue

            state["phase"] = "a2a_followup"
            # The A2A trajectory advances through the same checkpoints. Its acceptance
            # test is paired with TYK2 only at the length TYK2 has actually passed.
            for earlier in ENDPOINTS[:index + 1]:
                if screen_system(args, state, "a2a_followup", "a2a", arm, earlier) is None:
                    return wait()
            pair = get_score(arm, length, False, args)
            if pair is None:
                state["pending"].append(dict(phase="a2a_followup", arm=arm, length_ns=length, awaiting="paired_score"))
                return wait()
            ready = all(precision_passes(r) for r in pair["systems"].values())
            state["paired_screening"].append(dict(arm=arm, length_ns=length, precision_passes=ready))
            if not ready:
                continue
            eligible.append(f"{arm}-{length}")
            state["phase"] = "independent_validation"
            holdout = get_score(arm, length, True, args)
            if holdout is None:
                order = 9000 + 100*list(ARMS).index(arm) + 10*index
                submit(args.control, order, "validate", arm, length)
                state["pending"].append(dict(phase="independent_validation", arm=arm, length_ns=length,
                                             awaiting="independent_pair"))
                return wait()
            confirm_seeds(pair, holdout)
            state["validation"].append(dict(arm=arm, length_ns=length, passes=holdout["passes"]))
            if holdout["passes"]:
                winner = holdout
                break
        if winner is not None:
            break
    state["phase"] = "complete"
    save_json(progress, state)
    decision = dict(strategy=state["strategy"], status="validated" if winner else "no_validated_protocol",
                    selection=winner, screened_candidates=eligible, pilot_ranking=state["ranking"])
    return finish(args, decision)


def step(args):
    return tyk2_first(args) if getattr(args, "strategy", "paired-sweep") == "tyk2-first" else paired_sweep(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "geometry", "analysis", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--strategy", choices=("paired-sweep", "tyk2-first"), default="paired-sweep")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not step(args) and not args.once:
            time.sleep(30)


if __name__ == "__main__":
    main()
