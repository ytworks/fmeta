"""The back end of a run (FM-D18): which one a run gets, the conditions of the plugin by name,
the conversion of a System, and a run that keeps the back end it began with.

NVIDIA's plugin is not needed here: a stand-in with its three entry points takes its place. What
the plugin itself computes is checked on a GPU (``fmeta check``, item ``backend``). The one test
that asks the plugin itself (``gpu``) holds the rule of the constraints to its integrator."""

import sys
import types

import openmm as mm
import openmm.unit as unit
import pytest
import yaml

from fmeta.config import Config, RunConfig
from fmeta.engine import BiasSettings, FunnelBias, MetadynamicsRun, RunSettings, backend
from fmeta.engine.backend import (
    CONFIGURATION,
    Backend,
    build,
    integrator_only,
    resolve,
    settle,
    unavailable,
    unmet,
    unsolved_constraints,
)
from fmeta.engine.hills import read_hills
from fmeta.engine.platform import available_platforms
from fmeta.testing.toy import make_simulation, make_toy
from fmeta.workflow import run_stage

SETTINGS = dict(temperature=300.0, bias_factor=8.0, height=1.2, sigma_s=0.05, sigma_rho=0.05,
                pace=20, points_per_sigma=3.0)
DODECAHEDRON = (mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(2, 2, 4 / 2**0.5))     # 2.83 nm along z


@pytest.fixture
def stand_in(monkeypatch):
    """The plugin's entry points, made of OpenMM objects that can be told from the stock ones."""
    barostats, com_intervals = [], []

    def nonbonded(stock):
        force = mm.CustomExternalForce("0")
        force.setName("PluginNonbonded")
        return force

    def barostat(pressure, temperature, frequency, seed):
        barostats.append((pressure, temperature, frequency, seed))
        force = mm.CustomExternalForce("0")
        force.setName("PluginBarostat")
        return force

    def integrator(temperature, friction, step, comFrequency=100, seed=0):     # the plugin's defaults
        com_intervals.append(comFrequency)
        made = mm.BrownianIntegrator(temperature, friction, step)
        made.setRandomNumberSeed(seed)
        return made

    namespace = types.SimpleNamespace(NonbondedForce=nonbonded, MonteCarloBarostat=barostat,
                                      LangevinMiddleIntegrator=integrator, barostats=barostats,
                                      com_intervals=com_intervals)
    monkeypatch.setattr(backend, "plugin", lambda: namespace)
    return namespace


def cube(edge):
    return mm.Vec3(edge, 0, 0), mm.Vec3(0, edge, 0), mm.Vec3(0, 0, edge)


def add_pme(system, tolerance=5e-4, method=mm.NonbondedForce.PME):
    force = mm.NonbondedForce()
    force.setNonbondedMethod(method)
    force.setCutoffDistance(1.0)
    force.setEwaldErrorTolerance(tolerance)
    for _ in range(system.getNumParticles()):
        force.addParticle(0.0, 0.3, 0.0)
    force.setForceGroup(3)
    system.addForce(force)
    return force


def pme_system(box=None, **kwargs):
    system = mm.System()
    for _ in range(2):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(*(box or cube(4.0)))
    return system, add_pme(system, **kwargs)


# ---- which back end a run gets -----------------------------------------------


def test_auto_takes_the_plugin_where_its_conditions_hold_and_openmm_with_the_reason_elsewhere(
        stand_in, monkeypatch):
    monkeypatch.setitem(CONFIGURATION, "membrane", ("force",))
    system, _ = pme_system()
    membrane, _ = pme_system()
    membrane.addForce(mm.MonteCarloMembraneBarostat(
        1.0 * unit.bar, 0.0 * unit.bar * unit.nanometer, 300.0 * unit.kelvin,
        mm.MonteCarloMembraneBarostat.XYIsotropic, mm.MonteCarloMembraneBarostat.ZFree))
    assert resolve("auto", "CUDA", "mixed", system) == Backend("cupotential", CONFIGURATION["soluble"])
    assert resolve("auto", "CUDA", "single", membrane) == Backend("cupotential", ("force",))
    # a run that continues keeps what the plugin provided when it began
    assert resolve("auto", "CUDA", "mixed", system, parts=("force",)) == Backend("cupotential", ("force",))
    assert resolve("openmm", "CUDA", "mixed", system) == Backend("openmm")

    for platform, precision, reason in (("Reference", "mixed", "the platform is Reference"),
                                        ("OpenCL", "mixed", "the platform is OpenCL"),
                                        ("CUDA", "double", "the precision is double")):
        fallback = resolve("auto", platform, precision, system)
        assert fallback.name == "openmm" and fallback.reason.startswith(reason)
        assert fallback.record() == dict(backend="openmm", backend_reason=fallback.reason)
        with pytest.raises(ValueError, match=reason):                 # by name: no run with other forces
            resolve("cupotential", platform, precision, system)
    with pytest.raises(ValueError, match="run.backend"):
        resolve("cuda", "CUDA", "mixed", system)


def test_a_system_without_a_pme_force_gets_the_plugin_integrator_where_runs_get_it(stand_in, monkeypatch):
    """The toy system of the checks: its exact free energy then tests the integrator of the runs."""
    assert integrator_only("auto", "CUDA", "mixed") == Backend("cupotential", ("integrator",))
    assert integrator_only("openmm", "CUDA", "mixed") == Backend("openmm")
    fallback = integrator_only("auto", "Reference", "mixed")
    assert fallback.name == "openmm" and fallback.reason.startswith("the platform is Reference")
    with pytest.raises(ValueError, match="the platform is Reference"):
        integrator_only("cupotential", "Reference", "mixed")
    for kind in CONFIGURATION:                                 # no run gets that integrator: nor does the toy
        monkeypatch.setitem(CONFIGURATION, kind, ("force",))
    assert integrator_only("auto", "CUDA", "mixed") == Backend("openmm")


def test_a_plugin_that_is_missing_or_does_not_load_is_a_reason_with_a_name(monkeypatch):
    system, _ = pme_system()
    monkeypatch.setitem(sys.modules, "nv_openmm_plugin", None)                 # as if it is not installed
    assert "is not installed" in resolve("auto", "CUDA", "mixed", system).reason

    broken = types.ModuleType("nv_openmm_plugin")                              # imports, and loads nothing

    def load(name):
        raise RuntimeError("libcupo.so: libcuda.so.1: cannot open shared object file\nTraceback of the loader")

    broken.__getattr__ = load
    monkeypatch.setitem(sys.modules, "nv_openmm_plugin", broken)
    assert resolve("auto", "CUDA", "mixed", system).reason == (
        "the cuPotential plugin does not load (libcupo.so: libcuda.so.1: cannot open shared object file)")
    with pytest.raises(ValueError, match="does not load"):
        resolve("cupotential", "CUDA", "mixed", system)


def _tolerance(system, force):
    force.setEwaldErrorTolerance(1e-5)


def _mesh(system, force):
    force.setPMEParameters(3.0, 32, 32, 32)


def _cutoff_method(system, force):
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)


def _second_force(system, force):
    add_pme(system)


def _narrow(system, force):
    system.setDefaultPeriodicBoxVectors(*cube(2.9))


def _skewed(system, force):                 # every box vector is 4 nm long, and the box is 2.83 nm along z
    system.setDefaultPeriodicBoxVectors(*DODECAHEDRON)


@pytest.mark.parametrize("change, reason", [
    (_tolerance, "the Ewald error tolerance is 1e-05"),
    (_mesh, "the PME parameters are set by hand"),
    (_cutoff_method, "does not use PME"),
    (_second_force, "the system has 2 NonbondedForce objects"),
    (_narrow, r"the box \(2.90 x 2.90 x 2.90 nm along x, y, z\) is narrower than 3 cutoffs of 1 nm"),
    (_skewed, r"the box \(4.00 x 4.00 x 2.83 nm along x, y, z\) is narrower than 3 cutoffs"),
], ids=lambda value: getattr(value, "__name__", None))
def test_a_condition_of_the_plugin_that_does_not_hold_is_named(stand_in, change, reason):
    system, force = pme_system()
    assert unmet("CUDA", "mixed", system) is None
    change(system, force)
    assert resolve("auto", "CUDA", "mixed", system).name == "openmm"
    with pytest.raises(ValueError, match=reason):
        resolve("cupotential", "CUDA", "mixed", system)


def test_the_box_of_the_run_decides_and_not_the_box_the_system_was_built_with(stand_in):
    system, _ = pme_system()
    assert "narrower" in unmet("CUDA", "mixed", system, box=cube(2.9) * unit.nanometer)
    system.setDefaultPeriodicBoxVectors(*cube(2.9))
    assert unmet("CUDA", "mixed", system, box=cube(4.0) * unit.nanometer) is None


# ---- the constraints the integrator of the plugin solves ----------------------

H, C, O = 1.008, 12.011, 15.999
ETHANE = (C, C, H, H, H, H, H, H)
ETHANE_HYDROGENS = tuple((carbon, hydrogen, 0.109) for carbon, hydrogens in ((0, (2, 3, 4)), (1, (5, 6, 7)))
                         for hydrogen in hydrogens)
CHAIN = ((0, 1, 0.15), (1, 2, 0.15), (2, 3, 0.15))
# The masses of the particles, the constraints (particle, particle, length in nm), and how many of
# the constraints the integrator of the plugin does not solve: the number it names when it refuses
# the system (the test on the GPU below holds this table to the plugin).
CONSTRAINTS = {
    "one bond": ((C, H), ((0, 1, 0.109),), 0),
    "three hydrogens": ((C, H, H, H), tuple((0, i, 0.109) for i in (1, 2, 3)), 0),
    "four hydrogens": ((C, H, H, H, H), tuple((0, i, 0.109) for i in (1, 2, 3, 4)), 4),
    "partners of two lengths": ((C, H, H), ((0, 1, 0.109), (0, 2, 0.110)), 2),
    "partners of two masses": ((C, H, 2.014), ((0, 1, 0.109), (0, 2, 0.109)), 2),
    "water": ((O, H, H), ((0, 1, 0.09572), (0, 2, 0.09572), (1, 2, 0.15139)), 0),
    "water of two hydrogen masses": ((O, H, 2.014), ((0, 1, 0.09572), (0, 2, 0.09572), (1, 2, 0.15139)), 0),
    "triangle of three lengths": ((O, H, H), ((0, 1, 0.09572), (0, 2, 0.1), (1, 2, 0.15139)), 3),
    "chain of four": ((C, C, C, C), CHAIN, 3),
    "massless chain of four": ((0.0, 0.0, 0.0, 0.0), CHAIN, 0),
    "ethane, bonds to hydrogen": (ETHANE, ETHANE_HYDROGENS, 0),
    "ethane, all bonds": (ETHANE, ETHANE_HYDROGENS + ((0, 1, 0.153),), 7),
    "bonds to hydrogen beside a chain of four": (
        (C, C, C, C, C, H, H), CHAIN + ((4, 5, 0.109), (4, 6, 0.109)), 3),
}


def constrained(name):
    """The case ``name`` of the table as a system with a PME force."""
    masses, constraints, _ = CONSTRAINTS[name]
    system = mm.System()
    for mass in masses:
        system.addParticle(mass)
    system.setDefaultPeriodicBoxVectors(*cube(4.0))
    for constraint in constraints:
        system.addConstraint(*constraint)
    add_pme(system)
    return system


def test_constraints_the_plugin_integrator_does_not_solve_keep_a_run_from_it(stand_in):
    """The integrator of the plugin refuses such a system when the Context is made. With constraints
    on all bonds, a run under ``auto`` got the plugin and did not start (review of PR #3)."""
    system = constrained("ethane, all bonds")
    fallback = resolve("auto", "CUDA", "mixed", system)
    assert fallback.name == "openmm" and fallback.reason.startswith("7 of the 7 constraints")
    with pytest.raises(ValueError, match="7 of the 7 constraints"):
        resolve("cupotential", "CUDA", "mixed", system)
    # OpenMM's integrator solves them: a run that began with the force of the plugin alone goes on
    assert resolve("auto", "CUDA", "mixed", system, parts=("force",)) == Backend("cupotential", ("force",))
    assert resolve("auto", "CUDA", "mixed", constrained("ethane, bonds to hydrogen")).name == "cupotential"


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_the_constraints_outside_the_groups_the_plugin_integrator_solves_are_counted(name):
    assert unsolved_constraints(constrained(name)) == CONSTRAINTS[name][2]


@pytest.mark.gpu
@pytest.mark.parametrize("name", CONSTRAINTS)
def test_the_plugin_integrator_refuses_exactly_the_constraints_that_are_counted(name):
    """The table above is the rule of the plugin itself: its integrator takes a system where the
    number is zero, and names the number where it refuses."""
    missing = unavailable("CUDA", "mixed") if "CUDA" in available_platforms() else "no CUDA platform"
    if missing:
        pytest.skip(missing)
    system, unsolved = constrained(name), CONSTRAINTS[name][2]
    integrator = build(Backend("cupotential", ("force", "integrator")), system, 300.0, 1.0, 0.002, seed=1)
    platform = mm.Platform.getPlatformByName("CUDA")
    if unsolved == 0:
        mm.Context(system, integrator, platform, dict(Precision="mixed"))
    else:
        with pytest.raises(mm.OpenMMException, match=f"does not support {unsolved} constraint"):
            mm.Context(system, integrator, platform, dict(Precision="mixed"))


# ---- the conversion of a system ----------------------------------------------


def run_system(membrane=False):
    """The toy system as a run has it on a GPU platform: funnel forces, PME, barostat, and the
    force that removes the motion of the centre of mass."""
    toy = make_toy()
    add_pme(toy.system)
    toy.system.addForce(mm.CMMotionRemover())
    if membrane:
        toy.system.addForce(mm.MonteCarloMembraneBarostat(
            1.0 * unit.bar, 0.0 * unit.bar * unit.nanometer, 300.0 * unit.kelvin,
            mm.MonteCarloMembraneBarostat.XYIsotropic, mm.MonteCarloMembraneBarostat.ZFree))
    else:
        barostat = mm.MonteCarloBarostat(1.5 * unit.bar, 310.0 * unit.kelvin, 40)
        barostat.setRandomNumberSeed(9)
        toy.system.addForce(barostat)
    bias = FunnelBias(toy.system, toy.funnel, BiasSettings(**SETTINGS), platform="CUDA")
    return toy.system, bias


def names(system):
    return [force.getName() for force in system.getForces()]


def probe_is_out(bias, integrator):
    """The readout of the CVs is not among the forces of a step, and asking again changes nothing."""
    groups = integrator.getIntegrationForceGroups()
    bias.exclude_probe(integrator)
    return integrator.getIntegrationForceGroups() == groups and bin(groups & 0xFFFFFFFF).count("1") == 31


def test_the_plugin_takes_the_place_of_exactly_the_parts_it_provides(stand_in):
    box = cube(11.0) * unit.nanometer

    system, bias = run_system()
    before = names(system)
    integrator = build(Backend("cupotential", ("force", "integrator")), system, 300.0, 1.0, 0.004,
                       seed=7, box=box, bias=bias)
    assert names(system) == [name for name in before if name not in ("NonbondedForce", "CMMotionRemover")] + [
        "PluginNonbonded"]
    assert system.getForce(system.getNumForces() - 1).getForceGroup() == 3      # the group of the stock force
    assert system.getDefaultPeriodicBoxVectors()[0][0] == 11.0 * unit.nanometer    # the plugin sizes its mesh from it
    assert isinstance(integrator, mm.BrownianIntegrator) and integrator.getRandomNumberSeed() == 7
    assert integrator.getStepSize() == 0.004 * unit.picosecond
    assert probe_is_out(bias, integrator)

    system, bias = run_system()                                # the force alone: OpenMM moves the atoms
    before = names(system)
    integrator = build(Backend("cupotential", ("force",)), system, 300.0, 1.0, 0.004, seed=7, bias=bias)
    assert names(system) == [name for name in before if name != "NonbondedForce"] + ["PluginNonbonded"]
    assert type(integrator) is mm.LangevinMiddleIntegrator and integrator.getRandomNumberSeed() == 7
    assert probe_is_out(bias, integrator)

    system, bias = run_system()                                # the isotropic barostat, with its settings
    before = names(system)
    build(Backend("cupotential", ("force", "integrator", "barostat")), system, 300.0, 1.0, 0.004, bias=bias)
    assert names(system) == [name for name in before if name not in (
        "NonbondedForce", "CMMotionRemover", "MonteCarloBarostat")] + ["PluginNonbonded", "PluginBarostat"]
    assert stand_in.barostats == [(1.5 * unit.bar, 310.0 * unit.kelvin, 40, 9)]

    system, bias = run_system(membrane=True)                   # there is no plugin barostat for a membrane
    build(Backend("cupotential", ("force", "integrator", "barostat")), system, 300.0, 1.0, 0.004, bias=bias)
    assert "MonteCarloMembraneBarostat" in names(system) and "PluginBarostat" not in names(system)


def test_the_plugin_integrator_removes_centre_of_mass_motion_only_where_the_system_asked_for_it(stand_in):
    """Its removal is on unless it is told otherwise. On the toy, whose one massive particle is
    the centre of mass, the particle then had 247 K under a thermostat of 300 K (FM-D18)."""
    build(Backend("cupotential", ("integrator",)), make_toy().system, 300.0, 1.0, 0.02)
    system, _ = run_system()
    build(Backend("cupotential", ("force", "integrator")), system, 300.0, 1.0, 0.004)
    without_remover, with_remover = stand_in.com_intervals
    assert without_remover == 0 and with_remover > 0


def test_openmm_leaves_the_system_as_it_is(stand_in):
    system, bias = run_system()
    before = mm.XmlSerializer.serialize(system)
    integrator = build(Backend("openmm"), system, 300.0, 1.0, 0.004, seed=7, box=cube(11.0) * unit.nanometer,
                       bias=bias)
    assert mm.XmlSerializer.serialize(system) == before
    assert type(integrator) is mm.LangevinMiddleIntegrator and integrator.getRandomNumberSeed() == 7
    assert probe_is_out(bias, integrator)


# ---- one run, one back end ---------------------------------------------------


def write_run(directory, record=None, checkpoint=True):
    directory.mkdir(parents=True)
    (directory / "run.yaml").write_text(yaml.safe_dump(dict(timestep_ps=0.004, **(record or {}))))
    if checkpoint:
        (directory / "checkpoint.zip").write_bytes(b"")


def test_a_run_that_continues_keeps_its_back_end_and_its_parts(tmp_path):
    made = Backend("cupotential", ("force",)).record()
    write_run(tmp_path / "rep0", made)
    assert settle("auto", tmp_path / "rep0", tmp_path) == ("cupotential", ("force",))
    assert settle("cupotential", tmp_path / "rep0", tmp_path) == ("cupotential", ("force",))
    with pytest.raises(RuntimeError, match="made with the cupotential back end and is not continued with openmm"):
        settle("openmm", tmp_path / "rep0", tmp_path)

    write_run(tmp_path / "rep1")                               # a run from before the entry existed
    assert settle("auto", tmp_path / "rep1", tmp_path) == ("openmm", None)
    with pytest.raises(RuntimeError, match="made with the openmm back end"):
        settle("cupotential", tmp_path / "rep1", tmp_path)

    write_run(tmp_path / "rep2", made, checkpoint=False)       # no checkpoint: it starts again from nothing
    assert settle("openmm", tmp_path / "rep2", tmp_path) == ("openmm", None)


def test_a_new_run_under_auto_follows_the_runs_beside_it(tmp_path):
    new = tmp_path / "rep5"
    assert settle("auto", new, tmp_path) == ("auto", None)
    write_run(tmp_path / "rep0" / "w0", Backend("cupotential", ("force",)).record())
    assert settle("auto", new, tmp_path) == ("cupotential", None)      # the name; the parts are today's
    assert settle("openmm", new, tmp_path) == ("openmm", None)         # a back end by name is taken as asked
    write_run(tmp_path / "rep1")
    assert settle("auto", new, tmp_path) == ("auto", None)             # they disagree: nothing to follow


def test_the_record_of_a_run_names_its_back_end_and_another_one_is_refused_before_any_file_changes(tmp_path):
    """The workflow refuses before it reads the prepared system (there is none here) or takes a device."""
    def toy_run(directory, conditions, hills):
        toy = make_toy()
        bias = FunnelBias(toy.system, toy.funnel, BiasSettings(**SETTINGS))
        run = MetadynamicsRun(make_simulation(toy, seed=5), bias, directory, toy.restraint_group,
                              RunSettings(checkpoint_hills=10), conditions=conditions)
        run.run(hills)
        run.close()

    made = Backend("cupotential", ("force", "integrator"))
    toy_run(tmp_path / "rep0", made.record(), 10)
    assert settle("auto", tmp_path / "rep0", tmp_path) == (made.name, made.parts)
    toy_run(tmp_path / "rep1", None, 10)

    for replica, asked, there in ((0, "openmm", "cupotential"), (1, "cupotential", "openmm")):
        directory = tmp_path / f"rep{replica}"
        files = [directory / name for name in ("HILLS", "COLVAR", "checkpoint.zip", "run.yaml")]
        before = [path.read_bytes() for path in files]
        with pytest.raises(RuntimeError, match=f"made with the {there} back end"):
            run_stage(Config(run=RunConfig(backend=asked)), tmp_path, replica=replica)
        assert before == [path.read_bytes() for path in files]

    toy_run(tmp_path / "rep0", made.record(), 20)              # under its own back end the run goes on
    assert len(read_hills(tmp_path / "rep0" / "HILLS")) == 20
