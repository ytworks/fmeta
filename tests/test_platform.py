"""Platform selection, and (opt-in) agreement of the GPU platforms with Reference."""

import numpy as np
import openmm as mm
import openmm.unit as unit
import pytest

from fmeta.cli import main as cli
from fmeta.engine import BiasSettings, FunnelBias, MetadynamicsRun
from fmeta.engine.hills import read_hills
from fmeta.engine.platform import GPU_PLATFORMS, available_platforms, select_platform
from fmeta.testing.toy import make_simulation, make_toy


def test_select_platform():
    platform, properties = select_platform("Reference")
    assert platform.getName() == "Reference" and properties == {}
    platform, properties = select_platform("auto")
    assert platform.getName() in available_platforms()
    if platform.getName() in GPU_PLATFORMS:
        assert properties["Precision"] == "mixed"
    with pytest.raises(ValueError, match="not available"):
        select_platform("NoSuchPlatform")


def test_doctor_runs(capsys):
    cli(["doctor", "--platform", "Reference"])
    out = capsys.readouterr().out
    assert "OpenMM" in out and "difference from Reference" in out


def _gpu_platform():
    for name in GPU_PLATFORMS:
        if name in available_platforms():
            try:
                return select_platform(name)
            except Exception:
                continue
    pytest.skip("no GPU platform available")


def _energy_and_forces(platform, properties, positions, hills):
    toy = make_toy()
    bias = FunnelBias(toy.system, toy.funnel, BiasSettings(), periodic=True)
    for hill in hills:
        bias.grid.add_hill(*hill)
    context = mm.Context(toy.system, mm.VerletIntegrator(0.001), platform, properties)
    context.setPositions(positions)
    bias.push(context)
    out = []
    for group in (bias.force_group, toy.restraint_group):
        state = context.getState(getEnergy=True, getForces=True, groups={group})
        out.append((state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer)))
    return out, bias.collective_variables(context)


@pytest.mark.gpu
def test_gpu_matches_reference():
    """Bias and wall energies, forces and CVs on the GPU agree with Reference,
    inside the funnel, against the walls, near the axis and across a periodic image."""
    platform, properties = _gpu_platform()
    reference = mm.Platform.getPlatformByName("Reference")
    toy = make_toy()
    hills = [(1.0, 0.05, 6.0), (1.6, 0.2, 4.0), (2.6, 0.1, 3.0), (3.4, 0.05, 2.0)]
    box = 12.0
    offsets = [[0.0, 0.0, 0.0], [0.15, -0.1, 0.5], [0.6, 0.3, 0.3], [0.02, 0.0, 2.6],
               [1e-4, 0.0, 1.2], [0.1, 0.05, -0.7], [0.15 + box, -0.1, 0.5 - box]]
    for offset in offsets:
        positions = toy.positions.copy()
        positions[0] += np.array(offset)
        (gpu, gpu_cv), (ref, ref_cv) = (_energy_and_forces(p, props, positions, hills)
                                        for p, props in ((platform, properties), (reference, {})))
        assert gpu_cv == pytest.approx(ref_cv, abs=2e-5)
        for (e_gpu, f_gpu), (e_ref, f_ref) in zip(gpu, ref):
            assert e_gpu == pytest.approx(e_ref, rel=1e-4, abs=1e-3)
            assert np.allclose(f_gpu, f_ref, rtol=2e-3, atol=0.05)


@pytest.mark.gpu
def test_gpu_run_and_restart(tmp_path):
    """A short toy run on the GPU, continued after a restart from its checkpoint."""
    platform, _ = _gpu_platform()
    for target in (40, 80):
        toy = make_toy()
        settings = BiasSettings(bias_factor=10.0, height=1.2, sigma_s=0.05, sigma_rho=0.05, pace=50)
        bias = FunnelBias(toy.system, toy.funnel, settings, periodic=True)
        simulation = make_simulation(toy, seed=2, platform=platform.getName())
        run = MetadynamicsRun(simulation, bias, tmp_path, toy.restraint_group)
        run.run(target)
        run.close()
    hills = read_hills(tmp_path / "HILLS")
    assert len(hills) == 80 and np.all(np.diff(hills.time) > 0)
    assert hills.s.min() > 0.2 and hills.s.max() < 3.8 and hills.rho.max() < 1.2


def test_check_command_plumbing(tmp_path):
    """The acceptance command: cheap checks run for real on the Reference platform,
    a failing check is recorded without stopping the rest, and dependants are skipped."""
    from fmeta.check import Checker
    checker = Checker(tmp_path / "ref", platform="Reference", quick=True)
    checker.run("platform", checker.check_platform)
    checker.run("forces", checker.check_forces, needs=("platform",))
    checker.run("toy_restart", checker.check_toy_restart, needs=("platform",))
    checker.run("broken", lambda: 1 / 0)
    checker.run("dependent", lambda: "never runs", needs=("broken",))
    assert {item.name: item.status for item in checker.items} == {
        "platform": "PASS", "forces": "PASS", "toy_restart": "PASS", "broken": "FAIL",
        "dependent": "SKIP"}
    assert "DRY RUN" in checker.items[0].detail
    assert "ZeroDivisionError" in (tmp_path / "ref" / "broken.traceback.txt").read_text()

    # without a GPU, the default (auto) run must fail loudly rather than pass on the CPU
    auto = Checker(tmp_path / "auto", platform="auto", quick=True)
    auto.run("platform", auto.check_platform)
    has_gpu = any(name in available_platforms() for name in GPU_PLATFORMS)
    if not has_gpu:
        assert auto.items[0].status == "FAIL" and "no usable GPU" in auto.items[0].detail
