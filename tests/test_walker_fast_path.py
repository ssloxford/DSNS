import numpy as np
import pytest

from dsns.constellation import WalkerConstellation
from dsns.helpers import IDHelper


RE = 6371e3

CONFIGS = {
    "iridium": dict(num_planes=6, sats_per_plane=11, inclination=86.4,
                    semi_major_axis=RE + 780e3, ascending_node_arc=180, phase_offset=0.0),
    "starlink": dict(num_planes=72, sats_per_plane=22, inclination=53.0,
                     semi_major_axis=RE + 550e3, ascending_node_arc=360, phase_offset=5.0),
    "small": dict(num_planes=3, sats_per_plane=4, inclination=30.0,
                  semi_major_axis=RE + 1200e3, ascending_node_arc=270, phase_offset=40.0,
                  right_ascending_node=25.0),
}


def make(eccentricity=0.0, **kw):
    from dsns.constellation import NullISLHelper as ISLHelper
    return WalkerConstellation("W", eccentricity=eccentricity, isl_helper=ISLHelper(),
                               id_helper=IDHelper(), **kw)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_fast_matches_kepler(name):
    c = make(**CONFIGS[name])
    assert c._circular_orbit
    worst = 0.0
    for t in np.linspace(0, 86400, 50):
        c.update_positions(t)
        fast = c.satellite_positions.copy()
        c._update_positions_kepler(t)
        worst = max(worst, np.abs(fast - c.satellite_positions).max())
    assert worst < 1e-6


def test_eccentric_uses_kepler():
    c = make(eccentricity=0.1, **CONFIGS["small"])
    assert not c._circular_orbit
    called = []
    orig = c._update_positions_kepler
    c._update_positions_kepler = lambda t: (called.append(t), orig(t))[1]
    c.update_positions(100.0)
    assert called == [100.0]


def test_positions_shape_and_no_aliasing():
    c = make(**CONFIGS["small"])
    arr = c.satellite_positions
    c.update_positions(1234.0)
    assert c.satellite_positions is arr
    assert arr.shape == (12, 3) and arr.dtype == np.float64
    for i, sat in enumerate(c.satellites):
        assert np.array_equal(sat.position, arr[i])
    old = [s.position.copy() for s in c.satellites]
    sat0 = c.satellites[0].position
    c.update_positions(2000.0)
    # previous position arrays are not mutated by a later update
    assert np.array_equal(sat0, old[0])
    c.satellites[0].position[:] = 0.0
    assert not np.array_equal(c.satellite_positions[0], 0.0 * arr[0])
