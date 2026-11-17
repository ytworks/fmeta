"""The back end of a run (FM-D18): which one a run gets, the conditions of the plugin by name,
the conversion of a System, and a run that keeps the back end it began with.

NVIDIA's plugin is not needed here: a stand-in with its three entry points takes its place. What
the plugin itself computes is checked on a GPU (``fmeta check``, item ``backend``)."""

import sys
import types

import openmm as mm
import openmm.unit as unit
import pytest
import yaml

from fmeta.config import Config, RunConfig
from fmeta.engine import BiasSettings, FunnelBias, MetadynamicsRun, RunSettings, backend
from fmeta.engine.backend import CONFIGURATION, Backend, build, resolve, settle, unmet
from fmeta.engine.hills import read_hills
from fmeta.testing.toy import make_simulation, make_toy
from fmeta.workflow import run_stage

SETTINGS = dict(temperature=300.0, bias_factor=8.0, height=1.2, sigma_s=0.05, sigma_rho=0.05,
                pace=20, points_per_sigma=3.0)
DODECAHEDRON = (mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(2, 2, 4 / 2**0.5))     # 2.83 nm along z


@pytest.fixture
def stand_in(monkeypatch):
    """The plugin's entry points, made of OpenMM objects that can be told from the stock ones."""
    barostats = []

    def nonbonded(stock):
        force = mm.CustomExternalForce("0")
        force.setName("PluginNonbonded")
        return force

    def barostat(pressure, temperature, frequency, seed):
        barostats.append((pressure, temperature, frequency, seed))
        force = mm.CustomExternalForce("0")
        force.setName("PluginBarostat")
        return force

    def integrator(temperature, friction, step, seed=0):
        made = mm.BrownianIntegrator(temperature, friction, step)
        made.setRandomNumberSeed(seed)
        return made

    namespace = types.SimpleNamespace(NonbondedForce=nonbonded, MonteCarloBarostat=barostat,
                                      LangevinMiddleIntegrator=integrator, barostats=barostats)
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
