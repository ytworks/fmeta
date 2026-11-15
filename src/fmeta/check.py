"""``fmeta check``: verify an installation on the hardware it will run on.

One command exercises everything that depends on the platform or on optional
packages, on a small self-contained system (the 35-residue protein shipped with
OpenMM plus a benzene molecule), and prints PASS / FAIL / SKIP per item:

* the funnel and bias forces on the selected platform against the Reference platform;
* periodic-image handling of the collective variables on a real solvated system;
* funnel metadynamics on a solvable toy system against its exact binding free energy;
* the whole pipeline (build, equilibrate, replicas, restart, walkers, analysis);
* a membrane build with ``Modeller.addMembrane`` and a short membrane run;
* OpenFF and GAFF ligand parameterisation, when those packages are installed;
* throughput with and without the bias.

On a machine with a GPU this is the acceptance test. ``--platform CPU --quick``
runs the same steps as a dry run (it says nothing about a GPU).
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from .analysis import analyze_run
from .config import Config
from .engine import BiasSettings, FunnelBias, MetadynamicsRun, RunSettings
from .engine.hills import read_colvar, read_hills
from .engine.platform import GPU_PLATFORMS, available_platforms, select_platform
from .funnel.geometry import compute_cvs
from .system.builder import SystemSettings, add_barostat, build_complex, ligand_net_charge
from .system.ligand import register_ligand
from .testing.systems import BENZENE_XML, write_complex_inputs, write_membrane_inputs
from .testing.toy import make_simulation, make_toy
from .workflow import (analyze_stage, equilibrate_stage, load_prepared, prepare_stage,
                       run_replicas)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

PROFILES = {
    # lengths in ps unless noted
    "full": dict(equil=dict(minimize_iterations=2000, heating_ps=20.0, release_ps=20.0, free_ps=50.0),
                 run_ps=100.0, pace_ps=1.0, toy_hills=5000, toy_tolerance=2.0, bench_steps=10000),
    "quick": dict(equil=dict(minimize_iterations=200, heating_ps=0.2, release_ps=0.2, free_ps=0.4),
                  run_ps=1.0, pace_ps=0.1, toy_hills=1500, toy_tolerance=None, bench_steps=200),
}


@dataclass
class Item:
    name: str
    status: str
    detail: str
    seconds: float


class Skip(Exception):
    """Raised by a check that cannot run here (missing optional package, failed prerequisite)."""


class Checker:
    def __init__(self, workdir, platform: str = "auto", precision: str = "mixed",
                 device: Optional[int] = None, gpus: Optional[list] = None, quick: bool = False):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.requested = platform
        self.precision = precision
        self.device = device if device is not None else (gpus[0] if gpus else None)
        # with an explicit device, replicas go through the subprocess launcher on that device
        self.gpus = gpus if gpus else ([device] if device is not None else None)
        self.profile = PROFILES["quick" if quick else "full"]
        self.items: list[Item] = []
        self.done: set[str] = set()
        self.platform: Optional[mm.Platform] = None
        self.properties: dict = {}

    # ---- plumbing ---------------------------------------------------------
    def run(self, name: str, function: Callable[[], str], needs: tuple = ()) -> None:
        start = time.time()
        missing = [n for n in needs if n not in self.done]
        try:
            if missing:
                raise Skip(f"needs {', '.join(missing)}")
            detail, status = function(), PASS
            self.done.add(name)
        except Skip as skip:
            detail, status = str(skip), SKIP
        except Exception as error:   # a failed check must not stop the others
            detail, status = f"{type(error).__name__}: {error}", FAIL
            (self.workdir / f"{name}.traceback.txt").write_text(traceback.format_exc())
        item = Item(name, status, detail, time.time() - start)
        self.items.append(item)
        print(f"[{status}] {name:<20s} {detail}  ({item.seconds:.0f} s)", flush=True)

    def config(self, **overrides) -> Config:
        p = self.profile
        data = {
            "system": {"ligand_method": "xml", "ligand_xml": [BENZENE_XML], "ligand_resname": "BNZ"},
            "equilibration": dict(p["equil"], seed=3),
            "run": {"length_ns": p["run_ps"] / 1000.0, "replicas": 2, "pace_ps": p["pace_ps"],
                    "checkpoint_ps": p["run_ps"] / 2.0, "trajectory_ps": p["run_ps"] / 2.0,
                    "seed": 11, "platform": self.platform.getName(), "precision": self.precision},
        }
        for section, values in overrides.items():
            if section == "input" or section == "membrane":
                data[section] = values
            else:
                data[section].update(values)
        return Config.from_dict(data)

    @property
    def n_hills(self) -> int:
        return round(self.profile["run_ps"] / self.profile["pace_ps"])

    # ---- the checks -------------------------------------------------------
    def check_platform(self) -> str:
        self.platform, self.properties = select_platform(self.requested, self.precision, self.device)
        name = self.platform.getName()
        system = mm.System()
        system.addParticle(1.0)
        context = mm.Context(system, mm.VerletIntegrator(0.001), self.platform, self.properties)
        device = ""
        for key in ("DeviceName", "DeviceIndex", "Precision"):
            try:
                device += f" {key}={self.platform.getPropertyValue(context, key)}"
            except Exception:
                pass
        text = f"OpenMM {mm.__version__}, platform {name}{device}; available: {', '.join(available_platforms())}"
        if name not in GPU_PLATFORMS:
            if self.requested == "auto":
                raise RuntimeError("no usable GPU platform (CUDA, HIP or OpenCL) was found. " + text)
            text += ". DRY RUN on a CPU platform: this does not verify a GPU"
        return text

    def check_forces(self) -> str:
        reference = mm.Platform.getPlatformByName("Reference")
        toy = make_toy()
        hills = [(1.0, 0.05, 6.0), (1.6, 0.2, 4.0), (2.6, 0.1, 3.0), (3.4, 0.05, 2.0)]
        box = 12.0
        offsets = [[0.0, 0.0, 0.0], [0.15, -0.1, 0.5], [0.6, 0.3, 0.3], [0.02, 0.0, 2.6],
                   [1e-4, 0.0, 1.2], [0.1, 0.05, -0.7], [0.15 + box, -0.1, 0.5 - box]]

        def evaluate(platform, properties, positions):
            system_toy = make_toy()
            bias = FunnelBias(system_toy.system, system_toy.funnel, BiasSettings(), periodic=True)
            for hill in hills:
                bias.grid.add_hill(*hill)
            context = mm.Context(system_toy.system, mm.VerletIntegrator(0.001), platform, properties)
            context.setPositions(positions)
            bias.push(context)
            out = []
            for group in (bias.force_group, system_toy.restraint_group):
                state = context.getState(getEnergy=True, getForces=True, groups={group})
                out.append((state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                            state.getForces(asNumpy=True).value_in_unit(
                                unit.kilojoule_per_mole / unit.nanometer)))
            return out, bias.collective_variables(context)

        worst_energy = worst_force = worst_cv = 0.0
        for offset in offsets:
            positions = toy.positions.copy()
            positions[0] += np.array(offset)
            (mine, my_cv), (ref, ref_cv) = (evaluate(p, props, positions) for p, props in
                                            ((self.platform, self.properties), (reference, {})))
            worst_cv = max(worst_cv, float(np.abs(np.array(my_cv) - np.array(ref_cv)).max()))
            for (e_mine, f_mine), (e_ref, f_ref) in zip(mine, ref):
                worst_energy = max(worst_energy, abs(e_mine - e_ref) / max(1.0, abs(e_ref)))
                scale = max(1.0, float(np.abs(f_ref).max()))
                worst_force = max(worst_force, float(np.abs(f_mine - f_ref).max()) / scale)
        if worst_cv > 2e-5 or worst_energy > 1e-4 or worst_force > 2e-3:
            raise AssertionError(f"differs from Reference: CV {worst_cv:.1e} nm, energy "
                                 f"{worst_energy:.1e} (relative), force {worst_force:.1e} (relative)")
        return (f"bias and wall forces match Reference at {len(offsets)} positions incl. a periodic "
                f"image: CV {worst_cv:.1e} nm, energy {worst_energy:.1e}, force {worst_force:.1e} (relative)")

    def _toy_run(self, directory, n_hills: int, seed: int):
        toy = make_toy()
        settings = BiasSettings(temperature=300.0, bias_factor=10.0, height=1.2, sigma_s=0.05,
                                sigma_rho=0.05, pace=100, points_per_sigma=2.0)
        bias = FunnelBias(toy.system, toy.funnel, settings, periodic=True)
        simulation = make_simulation(toy, seed=seed, platform=self.platform.getName(),
                                     properties=self.properties)
        run = MetadynamicsRun(simulation, bias, directory, toy.restraint_group,
                              RunSettings(checkpoint_hills=max(20, n_hills // 4)))
        run.run(n_hills)
        run.close()
        return toy

    def check_toy_restart(self) -> str:
        directory = self.workdir / "toy_restart"
        shutil.rmtree(directory, ignore_errors=True)
        for target in (40, 80):
            toy = self._toy_run(directory, target, seed=2)
        hills = read_hills(directory / "HILLS")
        colvar = read_colvar(directory / "COLVAR")
        if len(hills) != 80 or not np.all(np.diff(hills.time) > 0):
            raise AssertionError(f"expected 80 time-ordered hills after a restart, found {len(hills)}")
        if not (np.isfinite(colvar["s"]).all() and hills.s.min() > toy.funnel.lower - 0.3
                and hills.s.max() < toy.funnel.upper + 0.3):
            raise AssertionError("the ligand left the funnel or produced non-finite CVs")
        return "toy run of 80 hills, restarted from its checkpoint at 40; CVs stay inside the walls"

    def check_toy_free_energy(self) -> str:
        directory = self.workdir / "toy_free_energy"
        shutil.rmtree(directory, ignore_errors=True)
        n_hills = self.profile["toy_hills"]
        toy = self._toy_run(directory, n_hills, seed=12)
        result = analyze_run(directory)
        exact = toy.exact_binding_free_energy(300.0, result["site_max"])
        text = (f"{result['simulated_ns']:.0f} ns: dG0 reweighted {result['dG_reweighted']:.2f}, "
                f"plateau {result['dG_plateau']:.2f}, exact {exact:.2f} kJ/mol; "
                f"{result['rebinding_events']} rebinding events")
        tolerance = self.profile["toy_tolerance"]
        if tolerance is None:
            return text + " (quick mode: too short to assert agreement)"
        if abs(result["dG_reweighted"] - exact) > tolerance or abs(result["dG_plateau"] - exact) > 1.5 * tolerance:
            raise AssertionError(text + f"; outside the tolerance of {tolerance} kJ/mol")
        return text

    def _check_ligand(self, method: str) -> str:
        try:
            import openmmforcefields  # noqa: F401
            from openff.toolkit import Molecule  # noqa: F401
        except ImportError:
            raise Skip("openmmforcefields / openff-toolkit not installed; automatic ligand "
                       "parameters are unavailable (ligand ffxml or pre-parameterised input still work)")
        directory = self.workdir / "inputs"
        directory.mkdir(exist_ok=True)
        protein, ligand = write_complex_inputs(directory)
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        register_ligand(forcefield, ligand, method)
        settings = SystemSettings(ligand_method=method, ligand_resname="BNZ")
        modeller, ligand_atoms, _ = build_complex(protein, ligand, forcefield, settings)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
        context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
        context.setPositions(modeller.positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        charge = ligand_net_charge(system, ligand_atoms)
        if not np.isfinite(energy) or abs(charge) > 0.01:
            raise AssertionError(f"energy {energy}, benzene net charge {charge:+.3f} e")
        return (f"benzene parameterised with {method}: {len(ligand_atoms)} atoms, net charge "
                f"{charge:+.3f} e, complex energy {energy:.0f} kJ/mol")

    def check_prepare(self) -> str:
        directory = self.workdir / "inputs"
        directory.mkdir(exist_ok=True)
        protein, ligand = write_complex_inputs(directory)
        work = self.workdir / "soluble"
        shutil.rmtree(work, ignore_errors=True)
        prepared = prepare_stage(self.config(input={"protein": protein, "ligand": ligand}), work)
        problems = [c.name for c in prepared.report.errors]
        if problems:
            raise AssertionError(f"funnel checks failed: {problems}")
        funnel = prepared.funnel
        return (f"{prepared.topology.getNumAtoms()} atoms; funnel from s = {funnel.lower:.2f} to "
                f"{funnel.upper:.2f} nm, placed and validated automatically")

    def check_pbc(self) -> str:
        work = self.workdir / "soluble"
        prepared = load_prepared(work)
        config = Config.load(work / "config.yaml")
        system = prepared.system
        bias = FunnelBias(system, prepared.funnel, config.metad, periodic=True)
        context = mm.Context(system, mm.VerletIntegrator(0.001), self.platform, self.properties)
        box = prepared.box_vectors
        context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in box))
        reference = np.array(compute_cvs(prepared.funnel, prepared.positions))
        cases = {"as built": prepared.positions}
        shifted = prepared.positions + 0.37 * box[0] + 0.6 * box[1] - 0.45 * box[2]
        cases["whole system shifted across the box"] = shifted
        moved = prepared.positions.copy()
        moved[prepared.selection.ligand] += box[2] - box[0]
        cases["ligand placed in another periodic image"] = moved
        worst = 0.0
        for label, positions in cases.items():
            context.setPositions(positions)
            values = np.array(bias.collective_variables(context))
            wall = context.getState(getEnergy=True, groups={prepared.restraint_group})
            wall = wall.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            error = float(np.abs(values - reference).max())
            worst = max(worst, error)
            if error > 2e-3 or abs(wall) > 1e-2:
                raise AssertionError(f"{label}: CVs {values} vs {reference}, wall energy {wall:.3f}")
        return (f"CVs of the solvated complex are unchanged under {len(cases) - 1} periodic "
                f"re-imagings (max deviation {worst:.1e} nm)")

    def check_equilibrate(self) -> str:
        work = self.workdir / "soluble"
        summary = equilibrate_stage(Config.load(work / "config.yaml"), work, self.device)
        if not summary["funnel_checks"]["ok"]:
            raise AssertionError("funnel checks failed after equilibration")
        return (f"funnel wall energy on the bound ligand {summary['wall_energy_mean']:.3f} kJ/mol "
                f"({'ok' if summary['wall_ok'] else 'above 0.1 kT'}); box "
                + " x ".join(f"{v:.2f}" for v in summary["box_lengths"]) + " nm")

    def _verify_run(self, directory: Path, expected: int, funnel) -> str:
        hills = read_hills(directory / "HILLS")
        colvar = read_colvar(directory / "COLVAR")
        if len(hills) != expected:
            raise AssertionError(f"{directory.name}: {len(hills)} hills, expected {expected}")
        if not np.isfinite(colvar["s"]).all() or not np.isfinite(colvar["bias"]).all():
            raise AssertionError(f"{directory.name}: non-finite values in COLVAR")
        jump = float(np.abs(np.diff(colvar["s"])).max()) if colvar["s"].size > 1 else 0.0
        if jump > 1.0 or colvar["s"].min() < funnel.lower - 0.5 or colvar["s"].max() > funnel.upper + 0.5:
            raise AssertionError(f"{directory.name}: CV left the funnel or jumped by {jump:.2f} nm "
                                 "(a periodic-image problem would look like this)")
        speed = (directory / "run.log").read_text().strip().splitlines()[-1].split("speed")[-1].strip()
        return speed

    def check_production(self) -> str:
        work = self.workdir / "soluble"
        config = Config.load(work / "config.yaml")
        funnel = load_prepared(work).funnel
        run_replicas(config, work, self.gpus)
        speeds = [self._verify_run(work / f"rep{i}", self.n_hills, funnel) for i in range(2)]
        first = (work / "rep0" / "HILLS").read_text()
        run_replicas(config, work, self.gpus, length_ns=1.5 * config.run.length_ns)
        longer = round(1.5 * self.n_hills)
        for i in range(2):
            self._verify_run(work / f"rep{i}", longer, funnel)
        if not (work / "rep0" / "HILLS").read_text().startswith(first):
            raise AssertionError("continuing the run changed hills that were already written")
        return (f"2 replicas x {self.n_hills} hills, then continued to {longer} from their "
                f"checkpoints; {speeds[0]}")

    def check_walkers(self) -> str:
        source, work = self.workdir / "soluble", self.workdir / "walkers"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        for stage in ("prepared", "equil"):
            shutil.copytree(source / stage, work / stage)
        config = Config.load(source / "config.yaml")
        config.run.walkers, config.run.replicas = 2, 1
        config.save(work / "config.yaml")
        run_replicas(config, work, self.gpus)
        funnel = load_prepared(work).funnel
        for walker in ("w0", "w1"):
            self._verify_run(work / "rep0" / walker, self.n_hills, funnel)
        # each walker must feel hills the other one deposited
        own = read_hills(work / "rep0" / "w0" / "HILLS")
        seen = read_colvar(work / "rep0" / "w0" / "COLVAR")["bias"][-1]
        result = analyze_run(work / "rep0", write=False)
        if result["n_walkers"] != 2 or result["n_hills"] != 2 * self.n_hills:
            raise AssertionError("the analysis did not merge the two walkers")
        return (f"2 walkers sharing one bias ({result['n_hills']} hills in total; walker 0 "
                f"deposited {len(own)} and feels a bias of {seen:.1f} kJ/mol)")

    def check_analysis(self) -> str:
        work = self.workdir / "soluble"
        try:
            import matplotlib  # noqa: F401
            plot = True
        except ImportError:
            plot = False
        summary = analyze_stage(work, plot=plot)
        if not np.isfinite(summary["dG"]) or not (work / "summary.json").exists():
            raise AssertionError("analysis did not produce a finite result")
        figures = "; figures written" if plot else "; matplotlib not installed, no figures"
        return (f"analysis of {summary['n_replicas']} replicas ran (far too short to converge, as "
                f"expected: converged = {summary['converged']}){figures}")

    def check_benchmark(self) -> str:
        work = self.workdir / "soluble"
        config = Config.load(work / "config.yaml")
        with open(work / "equil" / "state.xml") as handle:
            state = mm.XmlSerializer.deserialize(handle.read())
        steps = self.profile["bench_steps"]
        dt = config.run.timestep_fs / 1000.0
        pace = max(1, round(1.0 / dt))                 # production setting: one hill per ps
        speeds = {}
        for label in ("plain", "biased"):
            prepared = load_prepared(work, config)
            system = prepared.system
            add_barostat(system, config.system.temperature, config.system.pressure, False)
            metad = BiasSettings(temperature=config.system.temperature, pace=pace)
            bias = FunnelBias(system, prepared.funnel, metad, periodic=True) if label == "biased" else None
            integrator = mm.LangevinMiddleIntegrator(config.system.temperature * unit.kelvin,
                                                     1.0 / unit.picosecond, dt * unit.picosecond)
            simulation = app.Simulation(prepared.topology, system, integrator, self.platform,
                                        self.properties)
            simulation.context.setState(state)
            simulation.step(min(100, steps))          # warm up (kernel compilation, neighbour list)
            start = time.time()
            if bias is None:
                simulation.step(steps)
            else:
                directory = self.workdir / "benchmark"
                shutil.rmtree(directory, ignore_errors=True)
                run = MetadynamicsRun(simulation, bias, directory, prepared.restraint_group,
                                      RunSettings(checkpoint_hills=10**9))
                start = time.time()
                run.run(max(1, steps // pace))
                run.close()
                steps = max(1, steps // pace) * pace
            simulation.context.getState(getEnergy=True)     # wait for the device to finish
            speeds[label] = steps * dt / 1000.0 / (time.time() - start) * 86400.0
        overhead = 100.0 * (speeds["plain"] / speeds["biased"] - 1.0)
        days = 500.0 / speeds["biased"]
        grid = bias.grid.values.size
        return (f"{prepared.topology.getNumAtoms()} atoms at {config.run.timestep_fs:.0f} fs: "
                f"{speeds['plain']:.0f} ns/day plain, {speeds['biased']:.0f} ns/day with funnel "
                f"metadynamics (overhead {overhead:.0f}%, {grid} grid points, 1 hill/ps); "
                f"a 500 ns replica of this system takes about {days:.1f} days")

    def check_membrane_build(self) -> str:
        directory = self.workdir / "inputs"
        directory.mkdir(exist_ok=True)
        protein, ligand = write_membrane_inputs(directory)
        work = self.workdir / "membrane"
        shutil.rmtree(work, ignore_errors=True)
        config = self.config(input={"protein": protein, "ligand": ligand},
                             membrane={"lipid": "POPC"}, run={"replicas": 1})
        prepared = prepare_stage(config, work)
        checks = {c.name: c.status for c in prepared.report.checks}
        if prepared.report.errors or checks.get("membrane") != "ok":
            raise AssertionError(f"funnel checks: {checks}")
        atoms = list(prepared.topology.atoms())
        lipids = sum(atoms[i].element.symbol == "P" for i in prepared.selection.lipid)
        box = np.diag(prepared.box_vectors)
        return (f"POPC bilayer built with Modeller.addMembrane: {lipids} lipids, "
                f"{prepared.topology.getNumAtoms()} atoms, box " + " x ".join(f"{v:.2f}" for v in box)
                + " nm; funnel points into water and clears the membrane")

    def check_membrane_run(self) -> str:
        work = self.workdir / "membrane"
        config = Config.load(work / "config.yaml")
        summary = equilibrate_stage(config, work, self.device)
        if not summary["funnel_checks"]["ok"]:
            raise AssertionError("funnel checks failed after membrane equilibration")
        run_replicas(config, work, self.gpus)
        speed = self._verify_run(work / "rep0", self.n_hills, load_prepared(work).funnel)
        return (f"equilibrated with the semi-isotropic barostat (area per lipid "
                f"{summary['area_per_lipid']:.2f} nm2 incl. the protein footprint), "
                f"{self.n_hills} hills of production; {speed}")

    # ---- driver -----------------------------------------------------------
    def run_all(self) -> bool:
        self.run("platform", self.check_platform)
        self.run("forces", self.check_forces, needs=("platform",))
        self.run("toy_restart", self.check_toy_restart, needs=("platform",))
        self.run("toy_free_energy", self.check_toy_free_energy, needs=("platform",))
        self.run("ligand_openff", lambda: self._check_ligand("openff"))
        self.run("ligand_gaff", lambda: self._check_ligand("gaff"))
        self.run("prepare", self.check_prepare, needs=("platform",))
        self.run("periodic_images", self.check_pbc, needs=("prepare",))
        self.run("equilibrate", self.check_equilibrate, needs=("prepare",))
        self.run("production", self.check_production, needs=("equilibrate",))
        self.run("walkers", self.check_walkers, needs=("equilibrate",))
        self.run("analysis", self.check_analysis, needs=("production",))
        self.run("benchmark", self.check_benchmark, needs=("equilibrate",))
        self.run("membrane_build", self.check_membrane_build, needs=("platform",))
        self.run("membrane_run", self.check_membrane_run, needs=("membrane_build",))

        counts = {status: sum(item.status == status for item in self.items)
                  for status in (PASS, FAIL, SKIP)}
        gpu = self.platform is not None and self.platform.getName() in GPU_PLATFORMS
        report = dict(platform=None if self.platform is None else self.platform.getName(),
                      gpu=gpu, counts=counts, items=[asdict(item) for item in self.items])
        with open(self.workdir / "check_report.json", "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped"
              f"   (report: {self.workdir / 'check_report.json'})")
        if not gpu:
            print("This was a CPU dry run; run 'fmeta check' on the GPU machine for the real test.")
        return counts[FAIL] == 0
