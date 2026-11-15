"""``fmeta check``: verify an installation on the hardware it will run on.

One command exercises everything that depends on the platform or on optional
packages, on a small self-contained system (the 35-residue protein shipped with
OpenMM plus a benzene molecule), and prints PASS / FAIL / SKIP per item:

* the funnel and bias forces on the selected platform against the Reference platform;
* the collective variables of a real solvated system while the platform moves the complex
  between periodic images;
* funnel metadynamics on a solvable toy system against its exact binding free energy;
* the whole pipeline (build, equilibrate, replicas, restart, walkers, analysis);
* the two run modes: replicas one after another on one device, and side by side on two with
  the next replica on the device that becomes free;
* a membrane build with ``Modeller.addMembrane`` and a short membrane run;
* the default route from a protein PDB and a ligand SDF alone, with OpenFF and with GAFF
  ligand parameters, when those packages are installed;
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
from .funnel.geometry import compute_cvs, funnel_frame
from .scheduler import read_schedule, verify_schedule
from .system.builder import SystemSettings, add_barostat, ligand_net_charge, prepare
from .testing.systems import BENZENE_XML, write_complex_inputs, write_membrane_inputs
from .testing.toy import make_simulation, make_toy
from .workflow import analyze_stage, equilibrate_stage, load_prepared, prepare_stage, run_replicas

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

PROFILES = {
    # lengths in ps unless noted
    "full": dict(equil=dict(minimize_iterations=2000, heating_ps=20.0, release_ps=20.0, free_ps=50.0),
                 run_ps=100.0, pace_ps=1.0, toy_hills=5000, toy_tolerance=2.0, bench_hills=200),
    "quick": dict(equil=dict(minimize_iterations=200, heating_ps=0.2, release_ps=0.2, free_ps=0.4),
                  run_ps=1.0, pace_ps=0.1, toy_hills=1500, toy_tolerance=None, bench_hills=4),
}


# What each item has to show. scripts/gpu_gates.py stores these with the recorded evidence.
CRITERIA = {
    "platform": "a GPU platform (CUDA, HIP or OpenCL) is selected and usable",
    "forces": "funnel-wall and bias energies, forces and CVs match the Reference platform at 6 "
              "positions: CV <= 2e-5 nm, energy <= 1e-4, force <= 2e-3 (relative)",
    "toy_restart": "80 time-ordered hills after a restart at 40; CVs finite and inside the walls",
    "toy_free_energy": "solvable toy system, 10 ns: reweighted dG0 within 2.0 kJ/mol and plateau "
                       "mean within 3.0 kJ/mol of the exact value",
    "pdb_sdf_openff": "system built from the protein PDB and the ligand SDF alone with OpenFF ligand "
                      "parameters: finite energy, neutral benzene, funnel checks pass",
    "pdb_sdf_gaff": "the same with GAFF ligand parameters",
    "prepare": "solvated system built, funnel placed automatically, no funnel check fails",
    "equilibrate": "staged equilibration completes; funnel checks pass on the equilibrated structure",
    "periodic_images": "the ligand and the funnel anchors are in one molecule of the context; during "
                       "1000 steps each with the whole system shifted across the box and with a corner "
                       "of the periodic cell inside the complex, and with the ligand at the upper wall "
                       "while the funnel axis lies along the narrowest direction of the box, the CVs on "
                       "the run platform are within 2e-3 nm of the numpy CVs of the reported "
                       "coordinates (as reported and wrapped into the cell) and the wall energy within "
                       "0.1 kJ/mol of the walls at those CVs",
    "production": "2 replicas reach the requested number of hills and continue from their "
                  "checkpoints without changing earlier hills; CVs finite, inside the funnel, "
                  "no jump above 1 nm between samples",
    "walkers": "2 walkers deposit the requested hills each and the analysis merges them",
    "replicas_sequential": "mode sequential: 2 replicas run one after another on the last device given; "
                           "schedule.jsonl has no violation of the scheduler's rules; each replica "
                           "reaches the requested number of hills; its device.json names the assigned "
                           "device, and the memory in use on that GPU (nvidia-smi) grew with the run",
    "replicas_parallel": "mode parallel: 3 replicas on 2 GPUs; the first two start side by side on "
                         "different GPUs and the third on the GPU that the first ended replica "
                         "released; schedule.jsonl, hills, device.json and GPU memory as for "
                         "replicas_sequential",
    "analysis": "analysis of the replicas produces a finite result and summary.json",
    "benchmark": "throughput with and without the bias is measured (diagnostic: decides nothing)",
    "membrane_build": "POPC bilayer built with Modeller.addMembrane; funnel and membrane checks pass",
    "membrane_run": "membrane system equilibrates with the semi-isotropic barostat and reaches the "
                    "requested number of hills",
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
        # the devices of the replicas: those given, or the one device given, or the default one
        self.gpus = gpus if gpus else ([device] if device is not None else None)
        self.profile = PROFILES["quick" if quick else "full"]
        self.items: list[Item] = []
        self.done: set[str] = set()
        self.platform: Optional[mm.Platform] = None
        self.properties: dict = {}
        self.device_name: Optional[str] = None

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
                value = self.platform.getPropertyValue(context, key)
            except Exception:
                continue
            device += f" {key}={value}"
            if key == "DeviceName":
                self.device_name = value
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
        offsets = [[0.0, 0.0, 0.0], [0.15, -0.1, 0.5], [0.6, 0.3, 0.3], [0.02, 0.0, 2.6],
                   [1e-4, 0.0, 1.2], [0.1, 0.05, -0.7]]

        def evaluate(platform, properties, positions):
            system_toy = make_toy()
            bias = FunnelBias(system_toy.system, system_toy.funnel, BiasSettings(), platform=platform.getName())
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
        return (f"bias and wall forces match Reference at {len(offsets)} positions: CV {worst_cv:.1e} nm, "
                f"energy {worst_energy:.1e}, force {worst_force:.1e} (relative)")

    def _toy_run(self, directory, n_hills: int, seed: int):
        toy = make_toy()
        settings = BiasSettings(temperature=300.0, bias_factor=10.0, height=1.2, sigma_s=0.05,
                                sigma_rho=0.05, pace=100, points_per_sigma=2.0)
        bias = FunnelBias(toy.system, toy.funnel, settings, platform=self.platform.getName())
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
        """The default user path with nothing but a protein PDB and a ligand SDF:
        the ligand force field is generated on the fly (OpenFF or GAFF)."""
        try:
            import openmmforcefields  # noqa: F401
            from openff.toolkit import Molecule  # noqa: F401
        except ImportError:
            raise Skip("openmmforcefields / openff-toolkit not installed: without them a ligand "
                       "needs its own ffxml (system.ligand_xml). Install the conda environment "
                       "to run from a PDB and an SDF alone") from None
        directory = self.workdir / "inputs"
        directory.mkdir(exist_ok=True)
        protein, ligand = write_complex_inputs(directory)
        settings = SystemSettings(ligand_method=method)
        prepared = prepare(protein, ligand, settings, scratch_dir=self.workdir / f"ligand_{method}")
        charge = ligand_net_charge(prepared.system, prepared.selection.ligand)
        context = mm.Context(prepared.system, mm.VerletIntegrator(0.001), self.platform,
                             self.properties)
        context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in prepared.box_vectors))
        context.setPositions(prepared.positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        if not np.isfinite(energy) or abs(charge) > 0.01 or not prepared.report.ok:
            raise AssertionError(f"energy {energy}, benzene net charge {charge:+.3f} e, "
                                 f"funnel checks ok = {prepared.report.ok}")
        return (f"protein PDB + ligand SDF only, ligand parameters from {method}: "
                f"{prepared.topology.getNumAtoms()} atoms built, ligand net charge {charge:+.3f} e, "
                f"energy {energy:.0f} kJ/mol")

    def check_prepare(self) -> str:
        directory = self.workdir / "inputs"
        directory.mkdir(exist_ok=True)
        protein, ligand = write_complex_inputs(directory)
        work = self.workdir / "soluble"
        shutil.rmtree(work, ignore_errors=True)
        prepared = prepare_stage(self.config(input={"protein": protein, "ligand": ligand}), work, self.device)
        problems = [c.name for c in prepared.report.errors]
        if problems:
            raise AssertionError(f"funnel checks failed: {problems}")
        funnel = prepared.funnel
        return (f"{prepared.topology.getNumAtoms()} atoms; funnel from s = {funnel.lower:.2f} to "
                f"{funnel.upper:.2f} nm, placed and validated automatically")

    def check_equilibrate(self) -> str:
        work = self.workdir / "soluble"
        summary = equilibrate_stage(Config.load(work / "config.yaml"), work, self.device)
        if not summary["funnel_checks"]["ok"]:
            raise AssertionError("funnel checks failed after equilibration")
        return (f"funnel wall energy on the bound ligand {summary['wall_energy_mean']:.3f} kJ/mol "
                f"({'ok' if summary['wall_ok'] else 'above 0.1 kT'}); box "
                + " x ".join(f"{v:.2f}" for v in summary["box_lengths"]) + " nm")

    def check_pbc(self) -> str:
        work = self.workdir / "soluble"
        config = Config.load(work / "config.yaml")
        prepared = load_prepared(work, config)
        funnel = prepared.funnel
        with open(work / "equil" / "state.xml") as handle:
            state = mm.XmlSerializer.deserialize(handle.read())
        system = prepared.system
        add_barostat(system, config.system.temperature, config.system.pressure, False)
        bias = FunnelBias(system, funnel, config.metad, platform=self.platform.getName())
        integrator = mm.LangevinMiddleIntegrator(config.system.temperature * unit.kelvin,
                                                 1.0 / unit.picosecond,
                                                 config.run.timestep_fs * unit.femtosecond)
        integrator.setRandomNumberSeed(1)
        context = mm.Context(system, integrator, self.platform, self.properties)
        bias.exclude_probe(integrator)

        # A platform moves whole molecules between periodic images. The forces of the CVs join
        # the ligand and the anchors into one, and the CVs rely on that.
        anchors = set(funnel.ligand_atoms + funnel.p0_atoms + funnel.p1_atoms)
        molecules = [molecule for molecule in context.getMolecules() if anchors.intersection(molecule)]
        if len(molecules) != 1:
            raise AssertionError(f"the ligand and the funnel anchors are spread over {len(molecules)} "
                                 "molecules of the context: the platform may move them to different "
                                 "periodic images")
        solute = np.array(molecules[0])
        worst = {"cv": 0.0, "wall": 0.0}
        cells = set()

        def compare(label):
            values = np.array(bias.collective_variables(context))
            wall = context.getState(getEnergy=True, groups={prepared.restraint_group})
            wall = wall.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            images = []
            for wrapped in (False, True):
                snapshot = context.getState(getPositions=True, enforcePeriodicBox=wrapped)
                images.append(snapshot.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
                reference = np.array(compute_cvs(funnel, images[-1]))
                error = float(np.abs(values - reference).max())
                wall_error = abs(wall - float(funnel.restraint_energy(*reference)))
                worst["cv"], worst["wall"] = max(worst["cv"], error), max(worst["wall"], wall_error)
                if error > 2e-3 or wall_error > 0.1:
                    raise AssertionError(
                        f"{label}: CVs {values} and wall energy {wall:.3f} kJ/mol on the platform, CVs "
                        f"{reference} from the coordinates "
                        f"{'wrapped into the cell' if wrapped else 'as reported'}")
            lattice = snapshot.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            moved = images[1][solute[0]] - images[0][solute[0]]
            cells.add(tuple(np.round(np.linalg.solve(lattice.T, moved)).astype(int)))

        start = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        centre = start[solute].mean(axis=0)
        chunks, steps = 4, 250
        placements = {"whole system shifted across the box": 0.37 * box[0] + 0.6 * box[1] - 0.45 * box[2],
                      "corner of the periodic cell inside the complex": -centre}
        for label, shift in placements.items():
            context.setState(state)
            context.setPositions(start + shift)
            compare(label)
            for _ in range(chunks):
                integrator.step(steps)
                compare(label)

        # The far end of the funnel in the worst orientation of the tumbling complex: the axis
        # along the narrowest direction of the box, where a vector as long as the funnel is more
        # than half a box width and a periodic force would take another image of the ligand.
        p0, axis, ligand = funnel_frame(funnel, start)
        across = np.cross(np.eye(3)[np.argmin(np.abs(axis))], axis)
        across /= np.linalg.norm(across)
        narrow = int(np.argmin(np.diag(box)))
        rotation = np.roll([across, np.cross(axis, across), axis], narrow + 1, axis=0)
        far = start.copy()
        far[prepared.selection.ligand] += (funnel.upper - np.dot(ligand - p0, axis)) * axis
        context.setState(state)
        context.setPositions(centre + (far - centre) @ rotation.T)
        compare("ligand at the upper wall, funnel axis along the narrowest direction of the box")
        return (f"ligand and anchors in one molecule of {len(solute)} atoms; shifted across the box and "
                f"with a cell corner inside the complex ({chunks * steps} steps each, the complex in "
                f"{len(cells)} periodic cells), and at the upper wall along the narrowest direction of "
                f"the box: CVs within {worst['cv']:.1e} nm, wall energy within {worst['wall']:.1e} kJ/mol "
                "of the values from the coordinates")

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
                                 "(a ligand in another periodic image than the protein looks like this)")
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

    def _fresh_copy(self, name: str, **run) -> tuple[Config, Path]:
        """A new work directory ``name`` with the prepared and equilibrated system of ``soluble``."""
        source, work = self.workdir / "soluble", self.workdir / name
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        for stage in ("prepared", "equil"):
            shutil.copytree(source / stage, work / stage)
        config = Config.load(source / "config.yaml")
        for key, value in run.items():
            setattr(config.run, key, value)
        config.save(work / "config.yaml")
        return config, work

    def check_walkers(self) -> str:
        config, work = self._fresh_copy("walkers", walkers=2, replicas=1)
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

    def _verify_device(self, directory: Path, device) -> int:
        """Compare the ``device.json`` of a run with the device the scheduler gave it. Returns by
        how many MiB the memory in use on that GPU grew."""
        with open(directory / "device.json") as handle:
            record = json.load(handle)
        name = self.platform.getName()
        if record["platform"] != name or record["requested_device"] != device:
            raise AssertionError(f"{directory.name} was given {name} and device {device}; its device.json "
                                 f"names {record['platform']} and device {record['requested_device']}")
        if device is not None and record.get("device_index") != str(device):
            raise AssertionError(f"{directory.name} was given device {device}; the platform reports "
                                 f"device {record.get('device_index')}")
        before, after = record.get("memory_before_mib"), record.get("memory_after_mib")
        if before is None or after is None:
            raise Skip("the device of a replica cannot be confirmed: nvidia-smi does not report the "
                       f"memory in use on device {record.get('device_index')}")
        if after <= before:
            raise AssertionError(
                f"{directory.name}: the memory in use on GPU {record['device_index']} of nvidia-smi went "
                f"from {before} to {after} MiB when the run started there. The run is on another GPU, "
                "or another process freed memory on this one (the check needs GPUs that are idle)")
        return after - before

    def _run_mode(self, name: str, mode: str, devices: Optional[list], replicas: int) -> tuple[list, str]:
        """Run ``replicas`` replicas in ``mode`` and verify the record, the hills and the devices.
        Returns the start and end events as ``(event, job, device)`` and a text on the devices."""
        if self.platform.getName() not in GPU_PLATFORMS:
            raise Skip("needs a GPU platform: the item checks the device of each replica")
        config, work = self._fresh_copy(name, replicas=replicas)
        run_replicas(config, work, gpus=devices, mode=mode)
        events = read_schedule(work / "schedule.jsonl")[-1]
        problems = verify_schedule(events)
        if problems:
            raise AssertionError("schedule.jsonl: " + "; ".join(problems))
        funnel = load_prepared(work).funnel
        steps = [(event["event"], event["job"], event["devices"][0]) for event in events
                 if event["event"] in ("start", "end")]
        growth = []
        for kind, job, device in steps:
            if kind == "start":
                self._verify_run(work / job, self.n_hills, funnel)
                growth.append(self._verify_device(work / job, device))
        return steps, ("device.json agrees with the assignment; the memory in use on the assigned GPU "
                       "grew by " + ", ".join(str(value) for value in growth) + " MiB")

    def check_replicas_sequential(self) -> str:
        # the last device, not the default one: a run that ignores its device shows here
        device = self.gpus[-1] if self.gpus else None
        steps, devices = self._run_mode("sequential", "sequential", None if device is None else [device], 2)
        if steps != [(kind, f"rep{i}", device) for i in range(2) for kind in ("start", "end")]:
            raise AssertionError(f"the replicas did not run one after another on device {device}: {steps}")
        where = "the default device" if device is None else f"device {device}"
        return f"2 replicas x {self.n_hills} hills one after another on {where}; {devices}"

    def check_replicas_parallel(self) -> str:
        pair = (self.gpus or [])[:2]
        if len(pair) < 2 or self.platform.getName() not in GPU_PLATFORMS:
            raise Skip("needs two GPUs: give them with --gpus, such as --gpus 0,1")
        steps, devices = self._run_mode("parallel", "parallel", pair, 3)
        freed = steps[2][2]
        if (steps[:2] != [("start", "rep0", pair[0]), ("start", "rep1", pair[1])] or steps[2][0] != "end"
                or steps[3] != ("start", "rep2", freed) or [step[0] for step in steps[4:]] != ["end", "end"]):
            raise AssertionError("the replicas did not run two at a time, with the third on the device "
                                 f"that became free: {steps}")
        return (f"3 replicas x {self.n_hills} hills on devices {pair[0]} and {pair[1]}: rep0 and rep1 side "
                f"by side, rep2 on device {freed} after {steps[2][1]} ended there; {devices}")

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
        hills = self.profile["bench_hills"]
        dt = config.run.timestep_fs / 1000.0
        pace = max(1, round(1.0 / dt))                 # production setting: one hill per ps
        steps = hills * pace                           # the same stretch with and without the bias
        speeds = {}
        for label in ("plain", "biased"):
            prepared = load_prepared(work, config)
            system = prepared.system
            add_barostat(system, config.system.temperature, config.system.pressure, False)
            metad = BiasSettings(temperature=config.system.temperature, pace=pace)
            bias = (FunnelBias(system, prepared.funnel, metad, platform=self.platform.getName())
                    if label == "biased" else None)
            integrator = mm.LangevinMiddleIntegrator(config.system.temperature * unit.kelvin,
                                                     1.0 / unit.picosecond, dt * unit.picosecond)
            simulation = app.Simulation(prepared.topology, system, integrator, self.platform,
                                        self.properties)
            simulation.context.setState(state)
            # Warm-up of one hill period before the clock starts: kernel compilation, the
            # neighbour list and, with the bias, the first hill and the first checkpoint.
            if bias is None:
                simulation.step(pace)
                simulation.context.getState(getEnergy=True)
                start = time.time()
                simulation.step(steps)
                simulation.context.getState(getEnergy=True)     # wait for the device to finish
                elapsed = time.time() - start
            else:
                directory = self.workdir / "benchmark"
                shutil.rmtree(directory, ignore_errors=True)
                run = MetadynamicsRun(simulation, bias, directory, prepared.restraint_group,
                                      RunSettings(checkpoint_hills=10**9))
                run.run(1)
                start = time.time()
                run.run(1 + hills)                    # ends with a checkpoint ...
                elapsed = time.time() - start
                start = time.time()
                run.checkpoint()                      # ... whose cost is reported on its own
                checkpoint = time.time() - start
                elapsed -= checkpoint
                run.close()
            speeds[label] = steps * dt / 1000.0 / elapsed * 86400.0
        overhead = 100.0 * (speeds["plain"] / speeds["biased"] - 1.0)
        days = 500.0 / speeds["biased"]
        grid = bias.grid.values.size
        return (f"{prepared.topology.getNumAtoms()} atoms at {config.run.timestep_fs:.0f} fs, "
                f"{hills} ps each: {speeds['plain']:.0f} ns/day plain, {speeds['biased']:.0f} ns/day "
                f"with funnel metadynamics (overhead {overhead:.1f}%, {grid} grid points, 1 hill/ps, "
                f"checkpoints excluded: {1e3 * checkpoint:.0f} ms each); "
                f"a 500 ns replica of this system takes about {days:.1f} days")

    def check_membrane_build(self) -> str:
        directory = self.workdir / "inputs"
        directory.mkdir(exist_ok=True)
        protein, ligand = write_membrane_inputs(directory)
        work = self.workdir / "membrane"
        shutil.rmtree(work, ignore_errors=True)
        config = self.config(input={"protein": protein, "ligand": ligand},
                             membrane={"lipid": "POPC"}, run={"replicas": 1})
        prepared = prepare_stage(config, work, self.device)
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
        self.run("pdb_sdf_openff", lambda: self._check_ligand("openff"), needs=("platform",))
        self.run("pdb_sdf_gaff", lambda: self._check_ligand("gaff"), needs=("platform",))
        self.run("prepare", self.check_prepare, needs=("platform",))
        self.run("equilibrate", self.check_equilibrate, needs=("prepare",))
        self.run("periodic_images", self.check_pbc, needs=("equilibrate",))
        self.run("production", self.check_production, needs=("equilibrate",))
        self.run("walkers", self.check_walkers, needs=("equilibrate",))
        self.run("replicas_sequential", self.check_replicas_sequential, needs=("equilibrate",))
        self.run("replicas_parallel", self.check_replicas_parallel, needs=("equilibrate",))
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
