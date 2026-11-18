"""The stages of a calculation and what each leaves on disk.

    workdir/config.yaml
    workdir/prepared/   system.xml  topology.cif  funnel.yaml  funnel.pdb  report.json  meta.json
    workdir/equil/      state.xml  equil.json
    workdir/rep0/ ...   run.yaml  HILLS  COLVAR  checkpoint.zip  traj.dcd  device.json  result.json
    workdir/schedule.jsonl  rep0.log ...
    workdir/summary.json

Every stage reads what the previous one wrote, so a calculation can be resumed
or re-analysed from any point.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from .analysis import analyze_replicas, walker_directories
from .config import Config
from .engine.backend import build, resolve, settle
from .engine.bias import FunnelBias
from .engine.equilibration import equilibrate
from .engine.platform import GPU_PLATFORMS, device_record, resolve_devices, select_platform, visible_gpus
from .engine.runner import (
    MetadynamicsRun,
    RunSettings,
    check_same_run,
    discard_run_without_checkpoint,
    rollback_to_checkpoint,
    system_conditions,
)
from .funnel.geometry import KB, Funnel
from .funnel.validate import validate_funnel
from .funnel.visualize import write_funnel_pdb
from .scheduler import DONE, FAILED, Job, Task, directory_lock, is_locked, run_jobs
from .system import inputs
from .system.builder import Prepared, add_barostat, prepare
from .system.selection import classify

log = logging.getLogger(__name__)


# ---- prepare ----------------------------------------------------------------

def prepare_stage(config: Config, workdir, device: Optional[int] = None) -> Prepared:
    workdir = Path(workdir)
    out = workdir / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    config.save(workdir / "config.yaml")
    source = config.input
    platform, properties = select_platform(config.run.platform, config.run.precision, device)

    if source.complex:
        if not source.ligand_resname:
            raise ValueError("input.complex needs input.ligand_resname")
        source.protein, source.ligand = inputs.split_complex(
            source.complex, source.ligand_resname, workdir / "inputs", source.ligand_smiles,
            source.ligand_chain, source.ligand_resid)
        config.system.ligand_resname = source.ligand_resname
        config.save(workdir / "config.yaml")
    if not (source.protein and source.ligand):
        raise ValueError("config.input needs protein + ligand (or complex + ligand_resname)")
    prepared = prepare(source.protein, source.ligand, config.system, config.membrane,
                       config.funnel, config.funnel_overrides, platform=platform,
                       scratch_dir=workdir / "inputs", platform_properties=properties)
    save_prepared(prepared, out)
    log.info("prepared system: %d atoms\n%s", prepared.topology.getNumAtoms(), prepared.report)
    return prepared


def save_prepared(prepared: Prepared, out: Path) -> None:
    with open(out / "system.xml", "w") as handle:
        handle.write(mm.XmlSerializer.serialize(prepared.system))
    with open(out / "topology.cif", "w") as handle:
        app.PDBxFile.writeFile(prepared.topology, prepared.positions * unit.nanometer, handle,
                               keepIds=True)
    np.save(out / "positions.npy", prepared.positions)     # full precision, unlike the CIF
    prepared.funnel.save(out / "funnel.yaml")
    write_funnel_pdb(out / "funnel.pdb", prepared.funnel, prepared.positions)
    with open(out / "report.json", "w") as handle:
        json.dump(prepared.report.to_dict(), handle, indent=2, default=float)
    meta = dict(restraint_group=prepared.restraint_group, membrane=prepared.membrane,
                ligand_atoms=[int(i) for i in prepared.selection.ligand],
                kept_atoms=[int(i) for i in prepared.selection.kept],
                n_atoms=prepared.topology.getNumAtoms())
    with open(out / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)


def load_prepared(workdir, config: Optional[Config] = None) -> Prepared:
    workdir = Path(workdir)
    out = workdir / "prepared"
    config = config or Config.load(workdir / "config.yaml")
    with open(out / "meta.json") as handle:
        meta = json.load(handle)
    with open(out / "system.xml") as handle:
        system = mm.XmlSerializer.deserialize(handle.read())
    topology = app.PDBxFile(str(out / "topology.cif")).topology
    positions = np.load(out / "positions.npy")
    funnel = Funnel.load(out / "funnel.yaml")
    # a directory prepared before FM-D15 has no kept atoms recorded
    selection = classify(topology, meta["ligand_atoms"], meta.get("kept_atoms", ()))
    from .funnel.validate import FunnelReport
    return Prepared(topology, system, positions, funnel, selection, FunnelReport(),
                    meta["restraint_group"], meta["membrane"], config.system)


# ---- equilibrate ------------------------------------------------------------

def equilibrate_stage(config: Config, workdir, device: Optional[int] = None) -> dict:
    workdir = Path(workdir)
    out = workdir / "equil"
    out.mkdir(parents=True, exist_ok=True)
    prepared = load_prepared(workdir, config)
    platform, properties = select_platform(config.run.platform, config.run.precision, device)
    result = equilibrate(prepared, platform, properties, config.equilibration)

    positions = np.array(result.state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box = np.array(result.state.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    # The structure has been through dynamics: its ligand is judged by the walls (FM-D7).
    report = validate_funnel(prepared.funnel, positions, prepared.selection, box=box,
                             cutoff=config.system.cutoff, tumbling=not prepared.membrane,
                             sample_kT=KB * config.system.temperature)
    wall_ok = result.wall_ok(config.system.temperature)
    summary = dict(wall_energy_mean=result.wall_energy_mean, wall_energy_max=result.wall_energy_max,
                   wall_ok=wall_ok, box_lengths=result.box_lengths.tolist(),
                   area_per_lipid=result.area_per_lipid, area_drift=result.area_drift,
                   free_blocks=result.free_blocks,
                   funnel_checks=report.to_dict())
    with open(out / "state.xml", "w") as handle:
        handle.write(mm.XmlSerializer.serialize(result.state))
    with open(out / "equil.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=float)
    with open(out / "equilibrated.cif", "w") as handle:
        app.PDBxFile.writeFile(prepared.topology, result.state.getPositions(), handle, keepIds=True)
    if not wall_ok:
        log.warning("the funnel walls act on the bound ligand (mean %.2f kJ/mol); widen the "
                    "funnel (funnel_overrides: {width: ...}) before production",
                    result.wall_energy_mean)
    log.info("after equilibration:\n%s", report)
    report.raise_for_errors()
    return summary


# ---- production -------------------------------------------------------------

def replica_directory(workdir, replica: int, walker: Optional[int] = None) -> Path:
    base = Path(workdir) / f"rep{replica}"
    return base if walker is None else base / f"w{walker}"


def run_stage(config: Config, workdir, replica: int = 0, device: Optional[int] = None,
              length_ns: Optional[float] = None, walker: Optional[int] = None) -> Path:
    """Run (or continue) one replica, or one walker of it. Returns its directory.

    With ``config.run.walkers > 1`` every walker of a replica must be running
    at the same time for the bias to be shared; :func:`run_replicas` does that.
    The directory is held while the run lasts: a second run of it is refused.
    """
    workdir = Path(workdir)
    if (walker is None) != (config.run.walkers <= 1):
        raise ValueError("pass walker=<index> exactly when config.run.walkers > 1")
    check_walker_set(replica_directory(workdir, replica), config.run.walkers)
    directory = replica_directory(workdir, replica, walker)
    with directory_lock(directory):
        _run(config, workdir, directory, replica, device, length_ns, walker)
    return directory


def _run(config: Config, workdir: Path, directory: Path, replica: int, device: Optional[int],
         length_ns: Optional[float], walker: Optional[int]) -> None:
    run_cfg, temperature = config.run, config.system.temperature
    requested, parts = settle(run_cfg.backend, directory, workdir)    # refuses before a device or a file is touched
    prepared = load_prepared(workdir, config)
    with open(workdir / "equil" / "state.xml") as handle:
        state = mm.XmlSerializer.deserialize(handle.read())
    dt = run_cfg.timestep_fs / 1000.0

    system = prepared.system
    add_barostat(system, temperature, config.system.pressure, prepared.membrane)
    # asked before the first context of this process, for the memory that the run adds
    gpus_before = visible_gpus() if run_cfg.platform == "auto" or run_cfg.platform in GPU_PLATFORMS else None
    platform, properties = select_platform(run_cfg.platform, run_cfg.precision, device)
    bias = FunnelBias(system, prepared.funnel, config.metad, platform=platform.getName())
    box = state.getPeriodicBoxVectors()
    backend = resolve(requested, platform.getName(), run_cfg.precision, system, box, parts)
    if backend.reason:
        log.warning("run.backend auto: this run is made with OpenMM itself, because %s", backend.reason)
    conditions = dict(backend.record(), system=system_conditions(system))     # before forces are replaced
    seed = run_cfg.seed + 1000 * replica + (walker or 0) if run_cfg.seed else 0
    integrator = build(backend, system, temperature, run_cfg.friction, dt, seed, box=box, bias=bias)
    simulation = app.Simulation(prepared.topology, system, integrator, platform, properties)
    simulation.context.setState(state)
    # fresh velocities decorrelate replicas and walkers (a restart overwrites them again)
    if seed:
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
    else:
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    hill_ps = config.metad.pace * dt
    selection = prepared.selection
    solute = np.sort(np.concatenate([selection.protein, selection.ligand, selection.other]))
    walker_dirs = None
    if walker is not None:
        walker_dirs = [replica_directory(workdir, replica, k) for k in range(run_cfg.walkers)]
    settings = RunSettings(
        checkpoint_hills=max(1, round(run_cfg.checkpoint_ps / hill_ps)),
        trajectory_hills=round(run_cfg.trajectory_ps / hill_ps) if run_cfg.trajectory_ps else 0,
        trajectory_atoms=[int(i) for i in solute] if run_cfg.trajectory_solute_only else None,
        walker_dirs=walker_dirs, sync_hills=run_cfg.walker_sync_hills)
    run = MetadynamicsRun(simulation, bias, directory, prepared.restraint_group, settings,
                          conditions=conditions)
    with open(directory / "device.json", "w") as handle:    # after a restart was accepted or refused
        json.dump(dict(device_record(simulation.context, device, gpus_before), **backend.record(),
                       restored_from=run.restored), handle, indent=2)
    length = run_cfg.length_ns if length_ns is None else length_ns
    run.run(round(length * 1000.0 / hill_ps))
    run.close()


def check_walker_set(replica: Path, n_walkers: int) -> None:
    """Refuse a run of ``n_walkers`` walkers that would leave hills of its replica directory
    out of its bias, before a file is touched. The analysis adds every HILLS file of the
    replica, and a walker reads only the walkers it is run with: with 3 walkers cut to 2, the
    bias of the analysis was off by 2.7 kJ/mol from the bias of the run (review of PR #1).
    More walkers than before are fine: the new one reads the hills that are there."""
    n_walkers = n_walkers if n_walkers > 1 else 0
    left_out = [d.name for d in walker_directories(replica) if int(d.name[1:]) >= n_walkers]
    if n_walkers and (replica / "HILLS").exists():
        left_out.append("HILLS (a run without walkers)")
    if left_out:
        raise RuntimeError(f"{replica} holds hills that a run with {n_walkers or 'no'} walkers leaves out of its "
                           f"bias ({', '.join(left_out)}), and the analysis would add them. Continue with the "
                           "walkers it was run with (more are allowed), or use a new directory.")


def _rollback_walkers(directories: Sequence[Path], conditions: dict) -> None:
    """Roll every walker of a replica back to its checkpoint, before any of them starts and
    reads the hills of the others. A walker without a checkpoint goes back to the start: the
    hills it wrote before it was stopped belong to no run. Nothing is touched when a walker was
    run under other ``conditions`` (entries of its run.yaml) than the ones it would continue with,
    or when the replica holds hills of a walker that is not among ``directories``."""
    check_walker_set(Path(directories[0]).parent, len(directories))
    for directory in directories:
        if is_locked(directory):
            raise RuntimeError(f"{directory} is in use by another fmeta process")
    for directory in directories:
        if (directory / "checkpoint.zip").exists():
            check_same_run(directory, conditions)
    for directory in directories:
        if (directory / "checkpoint.zip").exists():
            rollback_to_checkpoint(directory)
        else:
            discard_run_without_checkpoint(directory)


def replica_jobs(config: Config, workdir, length_ns: Optional[float] = None) -> list[Job]:
    """The replicas of ``workdir`` as jobs for :func:`fmeta.scheduler.run_jobs`.

    A replica is one child process (``fmeta run WORKDIR --replica I``). With walkers it is one
    child process per walker: they start together, because they share their bias, and the
    replica takes as many devices as it has walkers (or all of them, if there are fewer)."""
    workdir = Path(workdir)
    n_walkers = max(1, config.run.walkers)
    jobs = []
    for replica in range(config.run.replicas):
        walkers = list(range(n_walkers)) if n_walkers > 1 else [None]
        tasks = []
        for walker in walkers:
            name = f"rep{replica}" if walker is None else f"rep{replica}_w{walker}"
            command = [sys.executable, "-m", "fmeta.cli", "run", str(workdir), "--replica", str(replica),
                       "--backend", config.run.backend]
            if walker is not None:
                command += ["--walker", str(walker)]
            if length_ns is not None:
                command += ["--length-ns", str(length_ns)]
            tasks.append(Task(name, command, workdir / f"{name}.log"))
        before_start = None
        if n_walkers > 1:
            conditions = dict(bias=config.metad.to_dict(), timestep_ps=config.run.timestep_fs / 1000.0)
            if config.run.backend != "auto":        # auto follows what each walker was run with
                conditions["backend"] = config.run.backend
            before_start = functools.partial(
                _rollback_walkers, [replica_directory(workdir, replica, walker) for walker in walkers],
                conditions)
        jobs.append(Job(f"rep{replica}", tasks, devices=n_walkers, before_start=before_start))
    return jobs


def _last_line(path: Path) -> str:
    try:
        lines = path.read_text(errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[-1].strip().removeprefix("fmeta: error: ") if lines else ""


def unfinished_jobs(status: dict, record: Sequence[dict], workdir) -> Optional[str]:
    """What kept the jobs of a :func:`fmeta.scheduler.run_jobs` call from finishing, as one
    message for the user (None when all of them finished). ``record`` holds the events of the
    call; the log of a task is ``workdir/<task>.log``."""
    workdir = Path(workdir)
    problems = []
    for event in record:
        if event["event"] == "end" and event["status"] == FAILED:
            # in the order of their end: the first task that failed is the one that failed by itself
            name = next(name for name, code in event["returncodes"].items() if code != 0)
            logfile = workdir / f"{name}.log"
            problems.append(f"{name} failed: {_last_line(logfile) or 'no output'} (see {logfile})")
        elif event["event"] == "skip" and status.get(event["job"]) == FAILED:
            problems.append(f"{event['job']} {event['reason']}")
    if problems:
        return "; ".join(problems)
    unfinished = [key for key, value in status.items() if value != DONE]
    if unfinished:
        reason = next((event["reason"] for event in record if event["event"] == "stop"), "stop requested")
        return (f"stopped ({reason}) before {', '.join(unfinished)} finished; "
                "run the same command again to continue")
    return None


def run_replicas(config: Config, workdir, gpus=None, length_ns: Optional[float] = None,
                 mode: Optional[str] = None, launcher=None, per_gpu: Optional[int] = None) -> list[Path]:
    """Run all replicas (and their walkers) as child processes on the devices of the run.

    ``gpus``, ``mode`` and ``per_gpu`` stand in for ``run.gpus``, ``run.mode`` and
    ``run.per_gpu`` in this call (see :func:`fmeta.engine.platform.resolve_devices`). On one
    device the replicas run one after another; on several they run side by side, ``per_gpu``
    of them on each device, and the next replica starts on the place that
    becomes free. Every decision is appended to ``workdir/schedule.jsonl`` (see
    :mod:`fmeta.scheduler`), the output of a child to ``workdir/rep0.log`` and so on.

    SIGINT and SIGTERM end the children before the call returns. A replica that failed or was
    stopped is a ``RuntimeError``; the same call again continues from the checkpoints.
    ``launcher`` replaces the start of a child process (for tests).
    """
    workdir = Path(workdir)
    devices = resolve_devices(mode or config.run.mode, gpus if gpus is not None else config.run.gpus,
                              config.run.platform, per_gpu=per_gpu or config.run.per_gpu)
    record: list = []
    with directory_lock(workdir):
        status = run_jobs(replica_jobs(config, workdir, length_ns), devices, launcher,
                          log=workdir / "schedule.jsonl", events=record.append)
    problem = unfinished_jobs(status, record, workdir)
    if problem:
        raise RuntimeError(problem)
    return [replica_directory(workdir, i) for i in range(config.run.replicas)]


_PROGRESS = re.compile(r"hills (\d+)/(\d+)\s+time ([\d.]+) ns\s+speed ([\d.]+) ns/day")


def status_table(workdir) -> list[dict]:
    """Where the production of ``workdir`` stands: one row per replica, or per walker.

    ``state`` is ``running`` (a process holds the directory), ``done`` (the target of its last
    run is reached), ``stopped`` (it ran and is neither) or ``not started``. Hills, time and
    speed are those of its last checkpoint; the target is that of the last run, or the
    configured length when nothing ran yet."""
    workdir = Path(workdir)
    config = Config.load(workdir / "config.yaml")
    hill_ps = config.metad.pace * config.run.timestep_fs / 1000.0
    n_walkers = max(1, config.run.walkers)
    rows = []
    for replica in range(config.run.replicas):
        for walker in (range(n_walkers) if n_walkers > 1 else [None]):
            directory = replica_directory(workdir, replica, walker)
            row = dict(name=f"rep{replica}" if walker is None else f"rep{replica}/w{walker}",
                       hills=0, target_hills=round(config.run.length_ns * 1000.0 / hill_ps),
                       time_ns=0.0, speed_ns_day=None, device=None, backend=None)
            progress = None
            if (directory / "run.log").exists():
                progress = _PROGRESS.search(_last_line(directory / "run.log"))
            if progress:
                row.update(hills=int(progress[1]), target_hills=int(progress[2]),
                           time_ns=float(progress[3]), speed_ns_day=float(progress[4]))
            row["target_ns"] = row["target_hills"] * hill_ps / 1000.0
            if (directory / "device.json").exists():
                with open(directory / "device.json") as handle:
                    device = json.load(handle)
                row["device"] = " ".join(filter(None, [device["platform"], device.get("device_index")]))
                row["backend"] = device.get("backend")
            if is_locked(directory):
                row["state"] = "running"
            elif progress and row["hills"] >= row["target_hills"]:
                row["state"] = "done"
            elif (directory / "run.log").exists() or (directory / "HILLS").exists():
                row["state"] = "stopped"
            else:
                row["state"] = "not started"
            rows.append(row)
    return rows


def format_status(rows: Sequence[dict]) -> str:
    lines = [f"{'replica':10s} {'state':12s} {'time / target (ns)':>22s} {'ns/day':>9s}  device (back end)"]
    for row in rows:
        speed = "" if row["speed_ns_day"] is None else f"{row['speed_ns_day']:.1f}"
        progress = f"{row['time_ns']:.3f} / {row['target_ns']:.3f}"
        device = " ".join(filter(None, [row["device"], row.get("backend") and f"({row['backend']})"]))
        lines.append(f"{row['name']:10s} {row['state']:12s} {progress:>22s} {speed:>9s}  {device}")
    return "\n".join(lines)


# ---- analysis ---------------------------------------------------------------

def analyze_paper_stage(workdir, regions_path, length_ns: Optional[float] = None,
                        discard_ns: float = 2.0) -> dict:
    """Measure all configured independent replicas at a common, per-walker endpoint.

    The region definition belongs to the prepared system and funnel. This entry point
    only reads simulation files, including when used on an archived run.
    """
    from .analysis.paper import analyze_paper_replicas

    workdir = Path(workdir)
    regions = json.loads(Path(regions_path).read_text())
    required = {"system_sha256", "positions_sha256", "funnel_sha256", "topology_sha256", "equil_state_sha256",
                "site", "unbound", "bound_below", "core_radius"}
    if not isinstance(regions, dict) or not required.issubset(regions):
        raise ValueError(f"a paper region file requires {', '.join(sorted(required))}")
    for field, filename in (("system_sha256", "prepared/system.xml"), ("positions_sha256", "prepared/positions.npy"),
                            ("funnel_sha256", "prepared/funnel.yaml"), ("topology_sha256", "prepared/topology.cif"),
                            ("equil_state_sha256", "equil/state.xml")):
        digest = hashlib.sha256()
        with (workdir / filename).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != regions[field]:
            raise ValueError(f"the region file belongs to a different {filename}")

    config = Config.load(workdir / "config.yaml")
    directories = [replica_directory(workdir, i) for i in range(config.run.replicas)]
    if set(directories) != {d for d in workdir.glob("rep*") if d.is_dir() and d.name[3:].isdigit()}:
        raise ValueError("the replica directories differ from the independent replicas in config.yaml")
    backends = set()
    for directory in directories:
        check_walker_set(directory, config.run.walkers)
        walkers = [directory / f"w{i}" for i in range(config.run.walkers)] if config.run.walkers > 1 else []
        present = {d for d in directory.glob("w*") if d.is_dir() and d.name[1:].isdigit()}
        if set(walkers) != present:
            raise ValueError(f"{directory.name}: walker directories differ from config.yaml")
        for source in walkers or [directory]:
            for filename in ("run.yaml", "HILLS", "COLVAR"):
                if not (source / filename).is_file():
                    raise FileNotFoundError(source / filename)
            device = source / "device.json"
            backends.add(json.loads(device.read_text())["backend"] if device.exists() else None)
    result = analyze_paper_replicas(
        directories, site=regions["site"], unbound=regions["unbound"],
        bound_below=regions["bound_below"], core_radius=regions["core_radius"],
        cut_ns=config.run.length_ns if length_ns is None else length_ns, discard_ns=discard_ns)
    result["regions"] = regions
    result["base_seed"] = config.run.seed
    result["recorded_backends"] = sorted(backends, key=str)
    result["protocol"] = dict(replicas=config.run.replicas, walkers=config.run.walkers,
                              timestep_fs=config.run.timestep_fs, pace_ps=config.run.pace_ps,
                              friction=config.run.friction, sync_hills=config.run.walker_sync_hills,
                              precision=config.run.precision, bias=config.metad.to_dict())
    return result


def analyze_stage(workdir, plot: bool = False, **kwargs) -> dict:
    """Analyse every replica under ``workdir`` and combine them. With ``plot``,
    also write ``analysis.png`` per replica and ``summary.png`` (needs matplotlib)."""
    workdir = Path(workdir)
    directories = sorted((d for d in workdir.glob("rep*") if d.is_dir() and d.name[3:].isdigit()
                          and ((d / "HILLS").exists() or walker_directories(d))),
                         key=lambda d: int(d.name[3:]))
    if not directories:
        raise FileNotFoundError(f"no replica directories with a HILLS file under {workdir}")
    summary = analyze_replicas(directories, write_to=workdir / "summary.json", **kwargs)
    if plot:
        from .analysis.plots import plot_run, plot_summary
        for directory in directories:
            plot_run(directory)
        plot_summary(workdir)
    return summary


def format_summary(summary: dict) -> str:
    lines = [f"dG0 = {summary['dG']:.2f} kJ/mol  ({summary['dG_kcal']:.2f} kcal/mol)"
             f"  from {summary['n_replicas']} replica(s)"]
    if summary["ci95"] is not None:
        low, high = summary["ci95"]
        lines.append(f"95% interval: [{low:.2f}, {high:.2f}] kJ/mol   (standard error {summary['sem']:.2f})")
    else:
        lines.append("one replica: no interval. The scatter between replicas is the uncertainty of the value")
    if summary["no_binding"]:
        lines.append(f"NO BINDING: the value is within 1 kT of what the empty funnel gives "
                     f"({summary['dG_empty_funnel']:.2f} kJ/mol); it is not a binding free energy")
    for replica in summary["replicas"]:
        samples = replica["dG_reweighted"]
        lines.append(f"  {Path(replica['workdir']).name}: {replica['dG']:8.2f} +/- {replica['dG_error']:.2f} kJ/mol "
                     f"(from the samples {'none' if samples is None else format(samples, '.2f')}) "
                     f"over {replica['simulated_ns']:.1f} ns, {replica['rebinding_events']} rebinding event(s)")
    if summary["diagnostics"]:
        # a replica of the default length does not converge on its own: notes, not a verdict
        lines.append("Diagnostics of the replicas:")
        for directory, notes in summary["diagnostics"].items():
            lines += [f"  {Path(directory).name}: {note}" for note in notes]
    return "\n".join(lines)


def run_all(config: Config, workdir) -> dict:
    # the membrane build and the equilibration use the first device of the run
    device = resolve_devices(config.run.mode, config.run.gpus, config.run.platform)[0]
    prepare_stage(config, workdir, device)
    equilibrate_stage(config, workdir, device)
    run_replicas(config, workdir)
    return analyze_stage(workdir)
