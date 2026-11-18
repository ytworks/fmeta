"""The independent replay reconstructs requested resolution and weights from raw samples."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.constants import Avogadro, R


def test_independent_audit_caps_rebuild_states_and_samples(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    replay = import_module("audit_binding_reweight").audit_replica
    bias = dict(sigma_s=.1, sigma_rho=.1, bias_factor=10, temperature=300)
    grid = dict(s_min=0., s_max=2., n_s=5, rho_max=1., n_rho=5, reflect=True)
    funnel = dict(buffer=.2, width=0., steepness=1., inflection=1., k_wall=10000., eps=.001)
    (tmp_path / 'run.yaml').write_text(yaml.safe_dump(dict(bias=bias, grid=grid, funnel=funnel)))
    times = np.arange(5., 1001., 5.)
    s = np.where(np.arange(len(times)) // 2 % 2 == 0, .25, 1.25)
    const = np.ones(len(times))
    np.savetxt(tmp_path / 'HILLS', np.column_stack([times, s, .1*const, .1*const, .1*const, 0*const, 10*const]))
    np.savetxt(tmp_path / 'COLVAR', np.column_stack([times, s, .1*const, 0*const, 0*const]))
    regions = dict(site=(0., .5), unbound=(1., 1.5))
    expected = dict(n_mbar_samples=100, n_mbar_states=2, dG=0., peer_postcut_samples=0, unknown_samples=0)
    with pytest.raises(ValueError, match='replay has 200 samples and 4 states'):
        replay([tmp_path], regions, 1., 0., expected)
    result = replay([tmp_path], regions, 1., 0., expected, max_slices=2, max_samples=100)
    kt, radius, k_wall = R*.3, funnel['buffer'], funnel['k_wall']
    area = np.pi*radius**2 + 2*np.pi*radius*np.sqrt(np.pi*kt/(2*k_wall)) + 2*np.pi*kt/k_wall
    # The zero bias gives equal site and bulk weight. The supplied comparison value is deliberately different.
    exact = -kt * np.log(.5*area / (1.e24/Avogadro)) / 4.184
    assert (result['samples'], result['states']) == (100, 2)
    assert result['independent_kcal_mol'] == pytest.approx(exact, abs=1.e-8)
    assert result['difference_kcal_mol'] == pytest.approx(exact, abs=1.e-8)
