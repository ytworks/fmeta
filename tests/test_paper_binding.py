"""Checks on the explicit regions, bulk density and verification data of the opt-in analysis."""

import base64
import dataclasses

import numpy as np
import pytest
import yaml

from fmeta.analysis.fes import sliced_log_weights
from fmeta.analysis.paper import (
    _roundtrips,
    analyze_paper_replicas,
    profile_binding,
    sample_binding,
    saved_contact_wall,
)
from fmeta.engine.grid import BiasGrid, BiasSettings
from fmeta.engine.hills import ColvarWriter, HillsWriter, row_checksums
from fmeta.funnel.geometry import KB, STANDARD_VOLUME
from fmeta.testing.toy import make_toy


def test_fmap_cylinder_integral_with_fixed_site_and_bulk():
    """The two-basin PMF integral reduces to the volume of the bound basin times its affinity."""
    kt = KB * 300.0
    radius = 0.15
    funnel = make_toy(width=0., buffer=radius, k_wall=1e25, eps=0.).funnel
    s = np.linspace(0., 4., 101)
    site, bulk = (0.614, 1.316), (2.701, 3.299)
    profile = np.where(s < 1.5, -6 * kt, 0.)
    exact_kb = np.pi * radius**2 * np.ptp(site) * np.exp(6)
    expected = -kt * np.log(exact_kb / STANDARD_VOLUME)
    assert profile_binding(s, profile, funnel, kt, site, bulk) == pytest.approx(expected)
    assert profile_binding(s, profile + 10000., funnel, kt, site, bulk) == pytest.approx(expected)


def test_reweighted_gaussian_well_and_core_against_exact_toy():
    """Independent quadrature of the solvable 3D well validates the standard-state sign, both
    finite ends and the soft-wall and core normalizations. The weighted quadrature samples are
    not a PMF passed back into its own integration formula."""
    toy = make_toy(well_depth=20., eps=0.)
    kt = KB * 300.
    site, bulk = (0.60, 1.50), (2.80, 3.30)
    core_radius = 0.15
    s, rho = np.meshgrid(np.arange(0.501, 3.499, 0.002), np.arange(0.001, 1.20, 0.002))
    s, rho = s.ravel(), rho.ravel()
    log_weights = np.log(rho) - (toy.well_energy(s, rho) + toy.funnel.restraint_energy(s, rho)) / kt
    result = sample_binding(s, rho, log_weights, toy.funnel, kt, site, bulk, core_radius)
    exact = toy.exact_binding_free_energy(300., site_max=site[1], s_min=site[0])
    for route in ("dG", "dG_core"):
        assert result[route] == pytest.approx(exact, abs=0.001)

    # A PMF of a soft wall treated as a hard cylinder has a different bulk reference density.
    s_nodes = np.linspace(0.5, 3.4, 601)
    pmf = toy.exact_free_energy_profile(s_nodes, 300.)
    assert profile_binding(s_nodes, pmf, toy.funnel, kt, site, bulk) == pytest.approx(exact, abs=0.001)
    hard = dataclasses.replace(toy.funnel, k_wall=1.e25)
    assert profile_binding(s_nodes, pmf, hard, kt, site, bulk) > exact + 0.1


def test_core_and_reweighted_mbar_do_not_hide_missing_states():
    # Large offsets and an intermediate, out-of-site sample cannot change the normalized ratio.
    states = np.array([0, 0, 1, 2, 0, 1, 2, 2])
    bias = np.log(2.) * np.array([1., 0., -1.])
    reduced = np.array([np.zeros(8), 136. + bias[states]])
    counts = np.array([4, 4])
    ref = sliced_log_weights(reduced, counts)
    fast = sliced_log_weights(reduced, counts, fast=True)
    assert fast - fast.max() == pytest.approx(ref - ref.max(), abs=1e-6)
    combined = np.column_stack((reduced, reduced[:, [2, 6]]))
    evaluated = sliced_log_weights(reduced, counts, fast=True, evaluation_bias=combined)
    assert evaluated[:len(fast)] == pytest.approx(fast)
    assert evaluated[-2:] == pytest.approx(fast[[2, 6]])
    funnel = make_toy().funnel
    s = np.array([1., 1., 2., 3.])
    result = sample_binding(s, np.array([.1, .1, .1, .3]), np.array([0., 0., 200., -60.]),
                            funnel, KB * 300, (0.7, 1.4), (2.8, 3.2), .15)
    assert result["dG"] < 0 and result["dG_core"] is None
    assert result["support"]["site"]["ess"] == pytest.approx(2)


def test_a_recrossing_is_one_walker_returning_after_leaving_the_site():
    u = (3.0, 3.5)
    assert _roundtrips([3.1, 1.0, 3.2], 1.2, u) == {"unbinding": 1, "rebinding": 0}
    assert _roundtrips([3.2, 1.0, 2.0, 1.0, 3.2, 2.0, 1.0], 1.2, u) == {"unbinding": 1, "rebinding": 1}


def test_reweighted_wall_contacts_have_conditional_denominators():
    funnel = make_toy(eps=0.).funnel
    s = np.array([1., 1.1, 1.4, 3.])
    rho = np.array([.1, funnel.wall_radius(1.1) + .5, .1, .1])
    result = sample_binding(s, rho, np.log([1., 2., 6., 1.]), funnel, KB * 300., (.7, 1.5), (2.9, 3.2), .15, 1.2)
    wall = result["weighted_site_wall"]
    assert wall["site"]["above_kT"]["1.0"]["total"] == {"count": 1, "weight_fraction": pytest.approx(2/9)}
    assert wall["bound_side"]["site_weight_fraction"] == pytest.approx(1/3)
    assert wall["bound_side"]["above_kT"]["1.0"]["radial"]["weight_fraction"] == pytest.approx(2/3)
    assert wall["bound_side"]["above_kT"]["1.0"]["axial"]["count"] == 0


def test_saved_contact_wall_joins_only_the_actual_postdiscard_site_frames():
    walker = "rep0/w1"
    saved = dict(walker=walker, frames=9, frame_ps=20., frame_index=[2, 4, 5, 7],
                 log_weight=np.log([4., 8., 1., 2.]).tolist(), wall_kT=[1.5, 1.9, 1., 0.],
                 bound_side=[False, True, True, True])
    contacts = np.array([True, True, True, False, False, True, False, True, True])
    mask = base64.b64encode(np.packbits(contacts, bitorder="little").tobytes()).decode()
    structure = dict(walker=walker, frames=9, frame_ps=20., site_contact_bits_b64=mask)
    result = saved_contact_wall(dict(dcd_site_weights=[saved]), [structure])
    assert result["site_contact"]["count"] == 3
    assert result["site_contact"]["above_1kT_weight_fraction"] == pytest.approx(4/7)
    assert result["bound_side_contact"]["count"] == 2
    assert result["bound_side_contact"]["above_1kT_count"] == 0


def test_historical_cut_checks_the_peer_hills_actually_felt(tmp_path):
    dirs = [tmp_path / "rep0" / f"w{i}" for i in range(2)]
    f = make_toy().funnel
    b = BiasSettings(sigma_s=.15, sigma_rho=.1)
    meta = dict(funnel=f.to_dict(), bias=b.to_dict(), grid=BiasGrid.from_funnel(f, b).spec(),
                timestep_ps=1/b.pace, settings=dict(walker_dirs=[str(d) for d in dirs], trajectory_hills=2))
    for d in dirs:
        d.mkdir(parents=True)
        (d / "run.yaml").write_text(yaml.safe_dump(meta))
        hills = HillsWriter(d / "HILLS", b.bias_factor)
        colvar = ColvarWriter(d / "COLVAR")
        for i in range(1, 131):
            s = (1., 2., 3.008, 3.06, 3.103, 3.14)[i % 6]
            hills.write(i, s, .1, b.sigma_s, b.sigma_rho, .001)
            colvar.write(i, s, .1, 0., 0.)
        hills.close()
        colvar.close()
    crc = row_checksums(dirs[1] / "HILLS")[-1]
    (dirs[0] / "SYNC").write_text(f"0 w1:130:{crc}\n")
    (dirs[1] / "SYNC").write_text("0 w0:0:0\n")
    opts = dict(site=(.7, 1.4), unbound=(3., 3.15), cut_ns=.120, discard_ns=0.)
    result = analyze_paper_replicas([dirs[0].parent], **opts)
    assert result["status"] == "available"
    assert result["replicas"][0]["peer_postcut_samples"] == 120
    assert result["replicas"][0]["max_peer_time_ns"] == pytest.approx(.130)
    assert result["replicas"][0]["n_mbar_samples"] == 240
    first = result["replicas"][0]["dcd_site_weights"][0]
    assert first["frames"] == 60 and first["frame_ps"] == pytest.approx(2.)
    assert first["walker"] == "rep0/w0" and first["frame_index"] == list(range(2, 60, 3))

    (dirs[0] / "SYNC").write_text(f"0 w1:130:{crc + 1}\n")
    broken = analyze_paper_replicas([dirs[0].parent], **opts)
    assert broken["status"] == "unavailable" and broken["dG"] is None
    assert broken["replicas"][0]["unknown_samples"] == 120
    assert broken["dG_bias"] == result["dG_bias"]
