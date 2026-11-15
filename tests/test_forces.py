"""CVs and funnel walls: OpenMM expressions against numpy and finite differences."""

import numpy as np
import openmm as mm
import openmm.unit as unit
import pytest

from fmeta.funnel import forces
from fmeta.funnel.geometry import Funnel, compute_cvs

BOX = 6.0


def make_funnel(**kwargs):
    # ligand: 3 atoms with unequal weights; p0: 4 atoms; p1: 3 atoms
    defaults = dict(ligand_atoms=[0, 1, 2], ligand_weights=[12.0, 14.0, 16.0],
                    p0_atoms=[3, 4, 5, 6], p1_atoms=[7, 8, 9], upper=2.5)
    defaults.update(kwargs)
    return Funnel(**defaults)


def make_positions(rng, ligand_center):
    """p0 near the origin, p1 about 1 nm up the z axis, ligand where asked."""
    pos = np.zeros((10, 3))
    pos[0:3] = ligand_center + rng.normal(scale=0.08, size=(3, 3))
    pos[3:7] = rng.normal(scale=0.25, size=(4, 3))
    pos[7:10] = np.array([0.1, -0.05, 1.0]) + rng.normal(scale=0.25, size=(3, 3))
    return pos


def make_context(funnel, extra_forces, periodic=True):
    system = mm.System()
    for _ in range(10):
        system.addParticle(10.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(BOX, 0, 0), mm.Vec3(0, BOX, 0), mm.Vec3(0, 0, BOX))
    for group, force in enumerate(extra_forces):
        force.setForceGroup(group)
        system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    return mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))


def energy(context, positions, group):
    context.setPositions(positions)
    state = context.getState(getEnergy=True, groups={group})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_cvs_match_numpy_and_angle_form():
    rng = np.random.default_rng(1)
    funnel = make_funnel()
    angle_s = forces._centroid_force("distance(g1,g2)*cos(angle(g1,g2,g3))", funnel, True)
    angle_rho = forces._centroid_force("distance(g1,g2)*sin(angle(g1,g2,g3))", funnel, True)
    context = make_context(funnel, [forces.projection_force(funnel), forces.extent_force(funnel),
                                    angle_s, angle_rho])
    for _ in range(50):
        center = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(0.2, 2.6)])
        pos = make_positions(rng, center)
        s_ref, rho_ref = compute_cvs(funnel, pos)
        assert energy(context, pos, 0) == pytest.approx(s_ref, abs=1e-9)
        assert energy(context, pos, 1) == pytest.approx(rho_ref, abs=1e-9)
        # same quantities as the BioSimSpace-style angle expressions
        assert energy(context, pos, 2) == pytest.approx(s_ref, abs=1e-6)
        assert energy(context, pos, 3) == pytest.approx(np.sqrt(rho_ref**2 - funnel.eps**2), abs=1e-6)


def test_cvs_are_invariant_under_periodic_images():
    """Moving the ligand, or the whole protein, by lattice vectors must not change the CVs."""
    rng = np.random.default_rng(2)
    funnel = make_funnel()
    context = make_context(funnel, [forces.projection_force(funnel), forces.extent_force(funnel)])
    pos = make_positions(rng, np.array([0.3, -0.2, 2.2]))
    s_ref, rho_ref = compute_cvs(funnel, pos)
    for shift in ([BOX, 0, 0], [0, -BOX, 0], [0, 0, 2 * BOX], [-BOX, BOX, -BOX]):
        moved = pos.copy()
        moved[0:3] += np.array(shift)
        assert energy(context, moved, 0) == pytest.approx(s_ref, abs=1e-9)
        assert energy(context, moved, 1) == pytest.approx(rho_ref, abs=1e-9)
        s_np, rho_np = compute_cvs(funnel, moved, box=np.array([BOX] * 3))
        assert (s_np, rho_np) == pytest.approx((s_ref, rho_ref), abs=1e-9)


def test_restraint_energy_matches_numpy():
    rng = np.random.default_rng(3)
    funnel = make_funnel()
    context = make_context(funnel, [forces.restraint_force(funnel)])
    seen_nonzero = 0
    for _ in range(200):
        # keep every centroid separation below BOX/2 so the minimum image is the identity
        center = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-0.2, 2.6)])
        pos = make_positions(rng, center)
        s, rho = compute_cvs(funnel, pos)
        expected = float(funnel.restraint_energy(s, rho))
        seen_nonzero += expected > 1e-6
        assert energy(context, pos, 0) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert seen_nonzero > 50  # the walls were actually exercised


def numerical_forces(context, positions, group, h=1e-5):
    grad = np.zeros_like(positions)
    for i in range(positions.shape[0]):
        for k in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[i, k] += h
            minus[i, k] -= h
            grad[i, k] = (energy(context, plus, group) - energy(context, minus, group)) / (2 * h)
    return -grad


@pytest.mark.parametrize("center", [
    [0.6, 0.3, 1.2],      # inside the cone, pushing on the side wall
    [0.1, 0.1, 2.8],      # beyond the upper wall
    [0.05, 0.0, 0.2],     # below the lower wall
    [0.4, -0.5, 2.6],     # side wall and upper wall together
])
def test_restraint_forces_match_finite_differences(center):
    rng = np.random.default_rng(4)
    funnel = make_funnel()
    force_objects = [forces.restraint_force(funnel), forces.projection_force(funnel),
                     forces.extent_force(funnel)]
    context = make_context(funnel, force_objects)
    pos = make_positions(rng, np.array(center))
    for group in range(3):
        context.setPositions(pos)
        analytic = context.getState(getForces=True, groups={group}).getForces(asNumpy=True)
        analytic = analytic.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        numeric = numerical_forces(context, pos, group)
        scale = max(1.0, np.abs(numeric).max())
        assert np.allclose(analytic, numeric, atol=2e-5 * scale)
    assert energy(context, pos, 0) > 1e-3  # a wall is active in every case


def test_gradient_is_finite_and_bounded_on_the_axis():
    """With the ligand exactly on the axis the extent gradient must stay finite."""
    funnel = make_funnel(ligand_atoms=[0], ligand_weights=[1.0], p0_atoms=[3], p1_atoms=[7])
    context = make_context(funnel, [forces.extent_force(funnel), forces.projection_force(funnel)])
    pos = np.zeros((10, 3))
    pos[7] = [0.0, 0.0, 1.0]
    for offset in (0.0, 1e-7, 1e-5, 1e-3):
        pos[0] = [offset, 0.0, 1.7]
        context.setPositions(pos)
        state = context.getState(getForces=True, getEnergy=True, groups={0})
        f = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        assert np.all(np.isfinite(f))
        # |d rho / d r_ligand| <= 1 for a true distance; regularisation only lowers it
        assert np.linalg.norm(f[0]) <= 1.0 + 1e-6
    assert energy(context, pos, 1) == pytest.approx(1.7, abs=1e-9)


def test_wall_radius_and_effective_area():
    funnel = make_funnel()
    assert funnel.wall_radius(funnel.inflection) == pytest.approx(funnel.width / 2 + funnel.buffer)
    assert funnel.wall_radius(50.0) == pytest.approx(funnel.buffer)
    # effective area against direct quadrature of int 2 pi rho exp(-U/kT) drho
    kT = 2.494
    for s in (0.8, 2.0, 2.4):
        rho = np.linspace(0.0, 2.0, 400001)
        boltz = np.exp(-funnel.restraint_energy(np.full_like(rho, s), rho) / kT)
        numeric = np.trapezoid(2 * np.pi * rho * boltz, rho)
        assert funnel.effective_area(s, kT) == pytest.approx(numeric, rel=1e-6)


def test_funnel_yaml_roundtrip(tmp_path):
    funnel = make_funnel(width=0.75, metadata={"note": "x"})
    path = tmp_path / "funnel.yaml"
    funnel.save(path)
    assert Funnel.load(path) == funnel
