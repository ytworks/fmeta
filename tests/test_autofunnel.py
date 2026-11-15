"""Automatic funnel placement on synthetic pockets with a known exit."""

import numpy as np
import pytest

from fmeta.funnel.autodetect import (AutoFunnelSettings, auto_funnel, channel_is_open,
                                     fibonacci_sphere)
from fmeta.funnel.geometry import Funnel, compute_cvs, funnel_frame
from fmeta.funnel.obstacles import MembraneSlab, Obstacles
from fmeta.funnel.validate import required_image_distance, validate_funnel
from fmeta.funnel.visualize import write_funnel_pdb
from fmeta.system.selection import classify

from synthetic import ball_with_channel, membrane_barrel

FAST = dict(n_directions=600)


def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def status(report, name):
    return {c.name: c.status for c in report.checks}.get(name)


def test_fibonacci_sphere_is_uniform():
    points = fibonacci_sphere(2000)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    assert np.abs(points.mean(axis=0)).max() < 1e-3


@pytest.mark.parametrize("direction", [[0, 0, 1], [1, 1, 0.2], [-0.4, 0.3, -0.85]])
def test_finds_the_channel(direction):
    topology, positions, site = ball_with_channel(direction, seed=3)
    funnel, report = auto_funnel(topology, positions, settings=AutoFunnelSettings(**FAST))

    assert report.ok, str(report)
    assert angle(funnel.metadata["exit_direction"], direction) < 10.0
    origin, axis, ligand = funnel_frame(funnel, positions)
    assert angle(axis, direction) < 10.0                       # realised axis, from the anchors
    assert status(report, "exit") == "ok" and report.info["channel_open"]

    s, rho = compute_cvs(funnel, positions)
    assert s == pytest.approx(1.2, abs=0.15) and rho < 0.1      # pose on the axis, 1.2 nm from p0
    assert funnel.lower == pytest.approx(s - 0.5, abs=0.02)
    # the site is 1.2 nm below the surface: the wall narrows around there, and the
    # upper wall sits where the ligand is a full nm clear of the protein
    assert s + 0.5 <= funnel.inflection <= s + 2.0
    assert funnel.upper > funnel.inflection + 0.5
    assert status(report, "unbound_region") == "ok"
    assert len(funnel.p0_atoms) >= 3 and len(funnel.p1_atoms) >= 3
    assert abs(sum(funnel.p0_weights) - 1) < 1e-9


def test_buried_site_is_reported_but_still_gets_a_funnel():
    topology, positions, site = ball_with_channel([0, 1, 0], seed=5, channel=False)
    funnel, report = auto_funnel(topology, positions, settings=AutoFunnelSettings(**FAST))
    assert status(report, "exit") == "warning"
    assert status(report, "channel") == "warning" and not report.info["channel_open"]
    # The least obstructed way out of an off-centre cavity is towards the nearest
    # surface. Not exactly along the normal: oblique lanes between lattice rows
    # cross fewer atoms than the straight line does.
    assert angle(funnel.metadata["exit_direction"], [0, 1, 0]) < 45.0


def test_channel_flood_fill_follows_bends():
    """A site whose exit is not a straight line is still recognised as open."""
    topology, positions, site = ball_with_channel([0, 0, 1], seed=7, channel=False)
    selection = classify(topology)
    protein = positions[selection.protein]
    # carve an L-shaped tunnel: sideways from the site, then up to the surface
    corner = site + np.array([0.9, 0.0, 0.0])
    def tube(p, a, b, r=0.45):
        ab = b - a
        t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        return np.linalg.norm(p - (a + np.outer(t, ab)), axis=1) < r
    keep = ~(tube(protein, site, corner) | tube(protein, corner, corner + np.array([0, 0, 3.0])))
    obstacles = Obstacles(protein[keep])
    ligand = positions[selection.ligand]
    assert channel_is_open(site, ligand, obstacles, bulk_clearance=1.2)
    assert not channel_is_open(site, ligand, Obstacles(protein), bulk_clearance=1.2)


def test_membrane_redirects_the_exit_to_water():
    topology, positions, site = membrane_barrel(seed=1)
    settings = AutoFunnelSettings(**FAST)

    # without a membrane the wide lateral channel is the easier way out
    funnel, report = auto_funnel(topology, positions, settings=settings)
    assert angle(funnel.metadata["exit_direction"], [1, 0, 0]) < 12.0

    # inside a bilayer that route ends in lipid: the funnel must point to the water
    slab = MembraneSlab(center_z=0.0, half_thickness=2.2)
    funnel, report = auto_funnel(topology, positions, membrane=slab, settings=settings)
    assert report.ok, str(report)
    assert angle(funnel.metadata["exit_direction"], [0, 0, 1]) < 10.0
    origin, axis, _ = funnel_frame(funnel, positions)
    assert angle(axis, [0, 0, 1]) < 10.0
    # the whole unbound window lies above the slab
    assert origin[2] + (funnel.upper - 0.6) * axis[2] > slab.half_thickness + 1.0


def test_manual_exit_and_overrides():
    topology, positions, site = ball_with_channel([0, 0, 1], seed=3)
    settings = AutoFunnelSettings(exit_direction=[0, 0, 2.0], **FAST)
    funnel, report = auto_funnel(topology, positions, settings=settings, upper=6.5, k_wall=5000.0)
    assert funnel.upper == 6.5 and funnel.k_wall == 5000.0
    assert angle(funnel.metadata["exit_direction"], [0, 0, 1]) < 1e-6
    with pytest.raises(TypeError):
        auto_funnel(topology, positions, settings=settings, not_a_parameter=1.0)


def test_bss_style_method_points_roughly_out():
    topology, positions, site = ball_with_channel([0, 0, 1], seed=3)
    funnel, report = auto_funnel(topology, positions, settings=AutoFunnelSettings(method="bss"))
    assert angle(funnel.metadata["exit_direction"], [0, 0, 1]) < 30.0
    assert (funnel.lower, funnel.upper, funnel.inflection) == (0.5, 4.0, 2.0)   # BSS defaults


def test_validation_catches_bad_funnels(tmp_path):
    topology, positions, site = ball_with_channel([0, 0, 1], seed=3)
    selection = classify(topology)
    funnel, report = auto_funnel(topology, positions, settings=AutoFunnelSettings(**FAST))

    # box too small for the minimum-image CVs
    needed = required_image_distance(funnel)
    small = validate_funnel(funnel, positions, selection, box=np.full(3, needed - 0.5))
    assert status(small, "box_cv") == "error" and not small.ok
    with pytest.raises(ValueError, match="box_cv"):
        small.raise_for_errors()
    big = validate_funnel(funnel, positions, selection, box=np.full(3, needed + 3.0))
    assert status(big, "box_cv") == "ok" and status(big, "box_images") == "ok" and big.ok

    # a funnel too narrow for the pose
    narrow = Funnel.from_dict({**funnel.to_dict(), "width": 0.0, "buffer": 0.05})
    assert status(validate_funnel(narrow, positions, selection), "pose") == "error"

    # an upper wall that stops inside the protein
    short = Funnel.from_dict({**funnel.to_dict(), "upper": funnel.lower + 1.0})
    assert status(validate_funnel(short, positions, selection), "unbound_region") == "error"

    # anchors scattered across a wrapped box
    wrapped = positions.copy()
    wrapped[funnel.p0_atoms[::2]] += np.array([8.0, 0.0, 0.0])
    assert status(validate_funnel(funnel, wrapped, selection), "p0_anchor") == "error"

    n_atoms = write_funnel_pdb(tmp_path / "funnel.pdb", funnel, positions)
    lines = (tmp_path / "funnel.pdb").read_text().splitlines()
    assert n_atoms > 100 and sum(line.startswith("HETATM") for line in lines) == n_atoms


def test_ligand_selection_rules():
    topology, positions, site = ball_with_channel([0, 0, 1], seed=3)
    selection = classify(topology)
    assert len(selection.ligand) == 8 and len(selection.protein) == len(positions) - 8
    assert np.array_equal(classify(topology, "LIG").ligand, selection.ligand)
    assert np.array_equal(classify(topology, [0, 1, 2]).ligand, [0, 1, 2])
    with pytest.raises(ValueError):
        classify(topology, "XYZ")


def test_charged_ligand_gets_a_longer_funnel():
    topology, positions, site = ball_with_channel([0, 0, 1], seed=3)
    settings = AutoFunnelSettings(**FAST)
    neutral, report = auto_funnel(topology, positions, settings=settings, ligand_charge=0.0)
    charged, charged_report = auto_funnel(topology, positions, settings=settings, ligand_charge=1.0)
    assert status(report, "ligand_charge") is None
    assert status(charged_report, "ligand_charge") == "warning" and charged_report.ok
    # the unbound region moves out by the extra 0.5 nm of clearance (a little less or
    # more, as clearance does not grow exactly one-to-one along the axis)
    assert 0.3 < charged.upper - neutral.upper < 1.0
    assert settings.bulk_margin == 1.0            # the caller's settings are left alone
