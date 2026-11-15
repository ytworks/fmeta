"""Bias engine: agreement with openmm.app.Metadynamics, HILLS replay, exact restart."""

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest

from fmeta.engine import BiasGrid, BiasSettings, FunnelBias, MetadynamicsRun, RunSettings
from fmeta.engine.hills import read_colvar, read_hills, sum_hills
from fmeta.engine.trajectory import count_dcd_frames
from fmeta.funnel import forces
from fmeta.testing.toy import make_simulation, make_toy

SETTINGS = dict(temperature=300.0, bias_factor=8.0, height=1.2, sigma_s=0.05, sigma_rho=0.05,
                pace=20, points_per_sigma=3.0)


def test_grid_layout_and_kernels():
    grid = BiasGrid(0.0, 2.0, 41, 1.0, 21, 0.1, 0.05, reflect=False)
    grid.add_hill(1.0, 0.5, 2.0)
    assert grid.values.shape == (21, 41)
    assert grid.values[10, 20] == pytest.approx(2.0)                       # centre
    assert grid.values[10, 22] == pytest.approx(2.0 * np.exp(-0.5))        # one sigma along s
    assert grid.values[11, 20] == pytest.approx(2.0 * np.exp(-0.5))        # one sigma along rho
    assert grid.interpolate(1.025, 0.5) == pytest.approx(
        0.5 * (grid.values[10, 20] + grid.values[10, 21]))

    mirrored = BiasGrid(0.0, 2.0, 41, 1.0, 21, 0.1, 0.05, reflect=True)
    mirrored.add_hill(1.0, 0.03, 1.0)
    # even in rho: zero slope at the axis
    assert mirrored.values[0, 20] == pytest.approx(2.0 * np.exp(-0.5 * (0.03 / 0.05) ** 2))

    many = grid.empty_copy()
    rng = np.random.default_rng(0)
    s, rho, h = rng.uniform(0, 2, 50), rng.uniform(0, 1, 50), rng.uniform(0.1, 1, 50)
    many.add_hills(s, rho, h, chunk=7)
    one_by_one = grid.empty_copy()
    for args in zip(s, rho, h):
        one_by_one.add_hill(*args)
    assert np.allclose(many.values, one_by_one.values, rtol=1e-12, atol=1e-14)


def _reference_metadynamics(settings, grid):
    toy = make_toy()
    variables = [
        app.BiasVariable(forces.projection_force(toy.funnel, True), grid.s_min, grid.s_max,
                         settings.sigma_s, False, grid.n_s),
        app.BiasVariable(forces.extent_force(toy.funnel, True), 0.0, grid.rho_max,
                         settings.sigma_rho, False, grid.n_rho),
    ]
    reference = app.Metadynamics(toy.system, variables, settings.temperature * unit.kelvin,
                                 settings.bias_factor, settings.height * unit.kilojoule_per_mole,
                                 settings.pace)
    return toy, reference


def _reference_bias(reference, settings):
    free_energy = reference.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
    return -free_energy * (settings.bias_factor - 1.0) / settings.bias_factor


def test_matches_openmm_metadynamics_dynamics(tmp_path):
    """Same toy, same seed: our engine and OpenMM's reference class follow the same
    trajectory and build the same table. The horizon is short because any round-off
    difference grows exponentially along a Langevin trajectory."""
    settings = BiasSettings(reflect=False, **SETTINGS)
    n_hills = 25

    toy = make_toy()
    bias = FunnelBias(toy.system, toy.funnel, settings, periodic=True)
    simulation = make_simulation(toy, seed=7)
    MetadynamicsRun(simulation, bias, tmp_path, toy.restraint_group).run(n_hills)

    ref_toy, reference = _reference_metadynamics(settings, bias.grid)
    ref_simulation = make_simulation(ref_toy, seed=7)
    reference.step(ref_simulation, n_hills * settings.pace)

    assert bias.grid.values.max() > 4.0   # a real bias was built
    assert np.allclose(bias.grid.values, _reference_bias(reference, settings), rtol=0, atol=1e-6)
    assert bias.collective_variables(simulation.context) == pytest.approx(
        reference.getCollectiveVariables(ref_simulation), abs=1e-6)


def test_matches_openmm_metadynamics_table(tmp_path):
    """Feeding our logged hills to OpenMM's class reproduces our table."""
    settings = BiasSettings(reflect=False, **SETTINGS)
    toy = make_toy()
    bias = FunnelBias(toy.system, toy.funnel, settings, periodic=True)
    simulation = make_simulation(toy, seed=7)
    run = MetadynamicsRun(simulation, bias, tmp_path, toy.restraint_group)
    run.run(300)
    run.close()
    hills = read_hills(tmp_path / "HILLS")

    ref_toy, reference = _reference_metadynamics(settings, bias.grid)
    if not hasattr(reference, "_addGaussian"):
        pytest.skip("openmm.app.Metadynamics internals changed")
    context = make_simulation(ref_toy, seed=7).context
    for s, rho, height in zip(hills.s, hills.rho, hills.height):
        reference._addGaussian([s, rho], height * unit.kilojoule_per_mole, context)

    assert bias.grid.values.max() > 8.0
    # limited by the 10 significant digits of the HILLS text file
    assert np.allclose(bias.grid.values, _reference_bias(reference, settings), rtol=1e-6, atol=1e-8)


def test_hills_replay_reproduces_grid(tmp_path):
    toy = make_toy()
    bias = FunnelBias(toy.system, toy.funnel, BiasSettings(**SETTINGS), periodic=True)
    simulation = make_simulation(toy, seed=3)
    run = MetadynamicsRun(simulation, bias, tmp_path, toy.restraint_group)
    run.run(120)
    run.close()

    hills = read_hills(tmp_path / "HILLS")
    assert len(hills) == 120
    assert hills.bias_factor == pytest.approx(8.0)
    assert hills.time[0] == pytest.approx(20 * 0.02) and hills.time[-1] == pytest.approx(120 * 20 * 0.02)
    replayed = sum_hills(hills, bias.grid)
    assert np.allclose(replayed.values, bias.grid.values, rtol=1e-8, atol=1e-8)
    # well-tempered: heights decay where bias has accumulated
    assert hills.height[0] == pytest.approx(1.2) and hills.height.min() < 1.0

    colvar = read_colvar(tmp_path / "COLVAR")
    assert colvar["time"].size == 120
    # the bias column is the table value at the logged CVs
    check = bias.grid.empty_copy()
    check.add_hills(hills.s[:50], hills.rho[:50], hills.height[:50])
    assert colvar["bias"][50] == pytest.approx(float(check.interpolate(colvar["s"][50], colvar["rho"][50])),
                                               abs=0.05)


def _run(tmp_path, name, segments, seed=5, **run_kwargs):
    for target in segments:
        toy = make_toy()
        bias = FunnelBias(toy.system, toy.funnel, BiasSettings(**SETTINGS), periodic=True)
        simulation = make_simulation(toy, seed=seed)
        run = MetadynamicsRun(simulation, bias, tmp_path / name, toy.restraint_group,
                              RunSettings(checkpoint_hills=25, **run_kwargs))
        run.run(target)
        run.close()
    return tmp_path / name, bias


def test_restart_is_exact(tmp_path):
    """Two segments with a fresh process state in between equal one continuous run."""
    whole, bias_whole = _run(tmp_path, "whole", [100], trajectory_hills=10, colvar_stride=10)
    split, bias_split = _run(tmp_path, "split", [50, 100], trajectory_hills=10, colvar_stride=10)

    assert (whole / "HILLS").read_text() == (split / "HILLS").read_text()
    assert (whole / "COLVAR").read_text() == (split / "COLVAR").read_text()
    assert np.array_equal(bias_whole.grid.values, bias_split.grid.values)
    assert count_dcd_frames(whole / "traj.dcd") == count_dcd_frames(split / "traj.dcd") == 10
    assert (whole / "traj.dcd").read_bytes()[276:] == (split / "traj.dcd").read_bytes()[276:]
    assert read_colvar(whole / "COLVAR")["time"].size == 200


def test_restart_discards_work_after_the_last_checkpoint(tmp_path):
    """A crash between checkpoints leaves extra rows; restart must roll them back."""
    whole, _ = _run(tmp_path, "whole", [100], trajectory_hills=10)

    toy = make_toy()
    bias = FunnelBias(toy.system, toy.funnel, BiasSettings(**SETTINGS), periodic=True)
    simulation = make_simulation(toy, seed=5)
    run = MetadynamicsRun(simulation, bias, tmp_path / "crash", toy.restraint_group,
                          RunSettings(checkpoint_hills=25, trajectory_hills=10))
    run.run(50)

    # keep going past the checkpoint without writing a new one, then "crash"
    run.settings.checkpoint_hills = 10**9
    class Crash(Exception):
        pass
    original = run.bias.push
    def push_then_crash(context):
        original(context)
        if run.n_hills == 62:
            raise Crash
    run.bias.push = push_then_crash
    with pytest.raises(Crash):
        run.run(100)
    run.close()
    assert len(read_hills(tmp_path / "crash" / "HILLS")) > 50

    crashed, _ = _run(tmp_path, "crash", [100], trajectory_hills=10)
    assert (whole / "HILLS").read_text() == (crashed / "HILLS").read_text()
    assert (whole / "COLVAR").read_text() == (crashed / "COLVAR").read_text()
    assert count_dcd_frames(crashed / "traj.dcd") == 10


def test_bias_stays_continuous_outside_the_grid():
    """Past the grid edge the bias keeps its edge value instead of dropping to zero."""
    toy = make_toy()
    settings = BiasSettings(**SETTINGS)
    bias = FunnelBias(toy.system, toy.funnel, settings, periodic=True)
    simulation = make_simulation(toy)
    context = simulation.context
    grid = bias.grid
    for s in np.linspace(grid.s_max - 0.3, grid.s_max, 12):
        grid.add_hill(s, 0.05, 5.0)
    bias.push(context)

    def bias_at(s, rho=0.05):
        positions = toy.positions.copy()
        positions[0] = toy.positions[1] + np.array([rho, 0.0, s])
        context.setPositions(positions)
        return bias.energy(context)

    # the lookup is clamped 1e-4 nm inside the table
    edge = bias_at(grid.s_max - 1e-4)
    assert edge > 5.0
    assert bias_at(grid.s_max + 0.2) == pytest.approx(edge, rel=1e-6)
    assert bias_at(grid.s_max + 5e-5) == pytest.approx(edge, rel=1e-6)
    assert bias_at(grid.s_max + 0.2, rho=grid.rho_max + 0.3) == pytest.approx(
        bias_at(grid.s_max - 1e-4, rho=grid.rho_max - 1e-4), rel=1e-6, abs=1e-9)
    # an unclamped OpenMM table would give exactly zero here
    assert bias_at(grid.s_min - 0.2) == pytest.approx(bias_at(grid.s_min + 1e-4), rel=1e-6, abs=1e-12)
