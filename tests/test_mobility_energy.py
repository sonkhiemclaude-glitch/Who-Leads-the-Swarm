import math

import numpy as np
import pytest

from swarmleak.mobility import gauss_markov_3d, neighbours, rpgm_3d, two_hop
from swarmleak.protocol.energy import RadioModel


def test_gauss_markov_bounds_and_shape():
    tr = gauss_markov_3d(10, 120, np.random.default_rng(0), area=(800, 800))
    assert tr.shape == (121, 10, 3)
    assert tr[..., 0].min() >= 0 and tr[..., 0].max() <= 800
    assert tr[..., 2].min() >= 50 and tr[..., 2].max() <= 150
    sp = np.linalg.norm(np.diff(tr, axis=0), axis=2)
    assert sp.max() <= 2.0 * 15 + 1e-6  # speed clip [0.2, 2] x mean


def test_initial_positions_in_central_band():
    tr = gauss_markov_3d(50, 10, np.random.default_rng(1), area=(800, 800))
    assert tr[0, :, :2].min() >= 0.3 * 800 and tr[0, :, :2].max() <= 0.7 * 800


def test_rpgm_members_stay_within_radius_of_each_other():
    tr = rpgm_3d(8, 60, np.random.default_rng(2), radius=250)
    for k in (0, 30, 60):
        d = np.linalg.norm(tr[k][:, None] - tr[k][None], axis=2)
        assert d.max() <= 2 * 250 + 1e-6


def test_neighbours_and_two_hop():
    pos = np.array([[0, 0, 100], [200, 0, 100], [400, 0, 100], [900, 0, 100]], float)
    a = neighbours(pos, 250)
    assert a[0, 1] and not a[0, 2] and not a[0, 0]
    assert two_hop(a, 0) == [1, 2]


def test_airtime_formula_matches_section_5_3():
    r = RadioModel()
    # 500 B payload: 1 fragment, (500 + 64) * 8 / 6e6 + 20e-6 + 44e-6
    assert math.isclose(r.airtime(500), (500 + 64) * 8 / 6e6 + 64e-6)
    assert r.fragments(1472) == 1 and r.fragments(1473) == 2
    # c_tx = V (I_tx - I_idle) A(l + 29)
    assert math.isclose(r.tx_cost(500), 3.0 * (0.380 - 0.273) * r.airtime(529))
    assert math.isclose(r.rx_cost(500), 3.0 * (0.313 - 0.273) * r.airtime(529))
    assert RadioModel(include_security_overhead=False).tx_cost(500) < r.tx_cost(500)
    with pytest.raises(ValueError):
        r.airtime(0)
