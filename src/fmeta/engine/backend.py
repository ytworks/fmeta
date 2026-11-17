"""What computes the nonbonded forces and moves the atoms of a production run (FM-D18).

    openmm        OpenMM's own ``NonbondedForce`` and ``LangevinMiddleIntegrator``
    cupotential   NVIDIA's cuPotential plugin of OpenMM's CUDA platform (``nv_openmm_plugin``)
                  in their place
    auto          cuPotential where its conditions hold, OpenMM otherwise, with the reason
                  written down

The engine is OpenMM in every case: the plugin replaces objects inside an OpenMM System, in
memory, just before the Context of a run is made. ``prepared/system.xml`` stays as OpenMM wrote
it, and the forces of the funnel (CVs, walls, bias) are OpenMM's own under every back end.

One run keeps the back end it began with (:func:`settle`): its hills and samples are then those
of one set of forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import openmm as mm
import openmm.unit as unit
import yaml

BACKENDS = ("auto", "cupotential", "openmm")

# What the plugin provides in a run, by the kind of system (measured choice, FM-D18). There is
# no replacement for the membrane barostat; "barostat" stands for the isotropic one.
CONFIGURATION = {"soluble": ("force", "integrator"), "membrane": ("force", "integrator")}

EWALD_TOLERANCE = 5e-4      # the only tolerance the plugin sizes its mesh for
BOX_CUTOFFS = 3.0           # the plugin needs a box of at least this many cutoffs


@dataclass(frozen=True)
class Backend:
    name: str                       # "cupotential" or "openmm"
    parts: tuple = ()               # what the plugin provides: of "force", "integrator", "barostat"
    reason: Optional[str] = None    # why ``auto`` gave OpenMM

    def record(self) -> dict:
        """The entries of ``run.yaml`` and ``device.json``."""
        data: dict = dict(backend=self.name)
        if self.parts:
            data["backend_parts"] = list(self.parts)
        if self.reason:
            data["backend_reason"] = self.reason
        return data


def plugin():
    """NVIDIA's plugin module with its libraries loaded. ``ImportError`` says why not."""
    try:
        import nv_openmm_plugin as nv
    except ImportError as error:
        raise ImportError(f"the cuPotential plugin (nv_openmm_plugin) is not installed ({error})") from None
    try:
        _ = nv.NonbondedForce   # the import alone loads nothing: the libraries come with the first class
    except (ImportError, RuntimeError, OSError) as error:
        first_line = (str(error).strip().splitlines() or [type(error).__name__])[0]
        raise ImportError(f"the cuPotential plugin does not load ({first_line})") from error
    return nv


def _kind(system: mm.System) -> str:
    membrane = any(isinstance(force, mm.MonteCarloMembraneBarostat) for force in system.getForces())
    return "membrane" if membrane else "soluble"


def unmet(platform: str, precision: str, system: mm.System, box=None) -> Optional[str]:
    """The first condition of the plugin that a run does not meet, as text (None: all hold).

    ``box`` holds the periodic box vectors the run starts from (default: those of ``system``).
    These are the settings the plugin refuses by name; the box rule is from its documentation."""
    if platform != "CUDA":
        return f"the platform is {platform}, and cuPotential is a plugin of the CUDA platform"
    if precision not in ("single", "mixed"):
        return f"the precision is {precision}, and cuPotential supports single and mixed"
    try:
        plugin()
    except ImportError as error:
        return str(error)
    forces = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
    if len(forces) != 1:
        return f"the system has {len(forces)} NonbondedForce objects, and cuPotential replaces exactly one"
    force = forces[0]
    if force.getNonbondedMethod() != mm.NonbondedForce.PME:
        return "the NonbondedForce does not use PME"
    tolerance = force.getEwaldErrorTolerance()
    if abs(tolerance - EWALD_TOLERANCE) > 1e-9:
        return f"the Ewald error tolerance is {tolerance:g}, and cuPotential supports {EWALD_TOLERANCE:g} only"
    alpha, *mesh = force.getPMEParameters()
    if any(mesh) or (alpha.value_in_unit(unit.nanometer**-1) if unit.is_quantity(alpha) else alpha):
        return "the PME parameters are set by hand, and cuPotential derives its own mesh"
    cutoff = force.getCutoffDistance().value_in_unit(unit.nanometer)
    a, b, c = (box if box is not None else system.getDefaultPeriodicBoxVectors())
    widths = [vector[i].value_in_unit(unit.nanometer) for i, vector in enumerate((a, b, c))]
    if min(widths) < BOX_CUTOFFS * cutoff:
        return (f"the box ({' x '.join(f'{w:.2f}' for w in widths)} nm along x, y, z) is narrower than "
                f"{BOX_CUTOFFS:g} cutoffs of {cutoff:g} nm")
    return None


def resolve(requested: str, platform: str, precision: str, system: mm.System, box=None,
            parts: Optional[tuple] = None) -> Backend:
    """The back end of a run that asks for ``requested``; ``parts`` holds what the plugin provided
    in the run it continues (both as :func:`settle` returned them).

    ``cupotential`` under conditions the plugin does not support is a ``ValueError``; ``auto``
    gives OpenMM there, with the condition as the reason. An error of the plugin that this
    module has no name for is never turned into a run with OpenMM: it stops the run. Call this
    when the barostat is in ``system``: it tells a membrane system from a soluble one."""
    if requested not in BACKENDS:
        raise ValueError(f"run.backend must be one of {BACKENDS}, not {requested!r}")
    if requested == "openmm":
        return Backend("openmm")
    reason = unmet(platform, precision, system, box)
    if reason is None:
        return Backend("cupotential", tuple(parts) if parts else CONFIGURATION[_kind(system)])
    if requested == "cupotential":
        raise ValueError(f"the cupotential back end cannot run this: {reason}. "
                         "Use run.backend: openmm (or auto, for a new run).")
    return Backend("openmm", reason=reason)


def _recorded(directory: Path) -> dict:
    with open(directory / "run.yaml") as handle:
        record = yaml.safe_load(handle) or {}
    return dict(backend=record.get("backend", "openmm"),           # no entry: a run from before FM-D18
                parts=tuple(record.get("backend_parts", ())) or None)


def settle(requested: str, directory, workdir) -> tuple:
    """What a run of ``directory`` (a replica or a walker under ``workdir``) asks for, given the
    runs that are on disk: the name, and the parts the plugin provided in the run that
    ``directory`` continues (None: a new run, or one made with OpenMM). Reads only.

    A run that continues keeps its back end: ``auto`` follows its ``run.yaml``, and another back
    end by name is refused (``RuntimeError``) before a device or a file is touched. A new run
    under ``auto`` follows the other runs of ``workdir``, so that the replicas of one calculation
    are made with one back end; a back end by name is taken as asked."""
    directory, workdir = Path(directory), Path(workdir)
    if (directory / "checkpoint.zip").exists() and (directory / "run.yaml").exists():
        own = _recorded(directory)
        if requested not in ("auto", own["backend"]):
            raise RuntimeError(f"{directory} holds a run made with the {own['backend']} back end and is not "
                               f"continued with {requested}. Use a new directory, or run.backend: "
                               f"{own['backend']} (or auto).")
        return own["backend"], own["parts"]
    if requested != "auto":
        return requested, None
    others = {_recorded(path.parent)["backend"] for pattern in ("rep*/run.yaml", "rep*/w*/run.yaml")
              for path in workdir.glob(pattern) if path.parent.resolve() != directory.resolve()}
    return (others.pop() if len(others) == 1 else "auto"), None


def build(backend: Backend, system: mm.System, temperature: float, friction: float, timestep: float,
          seed: int = 0, box=None, bias=None) -> mm.Integrator:
    """The integrator of a run with ``backend``; for cuPotential, ``system`` is converted in place.

    ``temperature`` in K, ``friction`` in 1/ps, ``timestep`` in ps; ``seed`` 0 draws one.
    ``box`` holds the periodic box vectors the run starts from, and for cuPotential it becomes
    the default box of ``system``: the plugin sizes its mesh and its cell list when the Context
    is made, from that box, with room for a barostat to change it (about 15% along each axis).
    A run passes the box of its equilibrated state, also when it continues: every Context of the
    run then has the same mesh. The probe of ``bias`` is taken out of the integration here,
    before the Context exists.

    Call this after the barostat and the funnel forces are in ``system`` and before the Context
    is made; what a run records about its system (``run.yaml``) is read before this call."""
    kelvin, ps = unit.kelvin, unit.picosecond
    if backend.name == "openmm":
        integrator = mm.LangevinMiddleIntegrator(temperature * kelvin, friction / ps, timestep * ps)
        if seed:
            integrator.setRandomNumberSeed(seed)
    else:
        nv = plugin()
        if box is not None:
            system.setDefaultPeriodicBoxVectors(*box)
        if "force" in backend.parts:
            _replace(system, mm.NonbondedForce, nv.NonbondedForce, keep_group=True)
        if "barostat" in backend.parts:
            _replace(system, mm.MonteCarloBarostat, lambda stock: nv.MonteCarloBarostat(
                stock.getDefaultPressure(), stock.getDefaultTemperature(), stock.getFrequency(),
                stock.getRandomNumberSeed()))
        if "integrator" in backend.parts:
            # the plugin's integrator removes the motion of the centre of mass itself
            for index in reversed(range(system.getNumForces())):
                if isinstance(system.getForce(index), mm.CMMotionRemover):
                    system.removeForce(index)
            integrator = nv.LangevinMiddleIntegrator(temperature * kelvin, friction / ps, timestep * ps,
                                                     seed=seed)
        else:
            integrator = mm.LangevinMiddleIntegrator(temperature * kelvin, friction / ps, timestep * ps)
            if seed:
                integrator.setRandomNumberSeed(seed)
    if bias is not None:
        bias.exclude_probe(integrator)
    return integrator


def _replace(system: mm.System, stock_type, convert, keep_group: bool = False) -> None:
    """Replace the force of exactly ``stock_type`` in ``system`` by ``convert(force)``, if there is one."""
    found = [index for index in range(system.getNumForces()) if type(system.getForce(index)) is stock_type]
    for index in reversed(found):
        stock = system.getForce(index)
        converted = convert(stock)
        if keep_group:
            converted.setForceGroup(stock.getForceGroup())
        system.addForce(converted)
        system.removeForce(index)
