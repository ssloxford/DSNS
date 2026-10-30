import itertools

import numpy as np
import pytest

from dsns.constellation import (
    GroundConstellation, NullISLHelper, PlanetOrbitalCenter, WalkerConstellation,
)
from dsns.helpers import EARTH_RADIUS, IDHelper
from dsns.multiconstellation import MultiConstellation


def _walker(name, id_helper, center, planes=3, per_plane=4):
    return WalkerConstellation(
        name=name, num_planes=planes, sats_per_plane=per_plane, inclination=60.0,
        semi_major_axis=EARTH_RADIUS + 700e3, eccentricity=0.0, ascending_node_arc=360.0,
        phase_offset=0.0, isl_helper=NullISLHelper(), id_helper=id_helper,
        orbital_center=center,
    )


def _build(with_centers=True):
    ids = IDHelper()
    earth = PlanetOrbitalCenter("earth", (0., 0., 0.), 1.5e11, 3.15e7) if with_centers else None
    moon = PlanetOrbitalCenter("moon", earth, 3.8e8, 2.4e6) if with_centers else None
    kw = {} if not with_centers else {"orbital_center": earth}
    ground = GroundConstellation(
        "ground", np.array([[10.0, 20.0, 0.0], [-30.0, 100.0, 50.0], [60.0, -80.0, 0.0]]),
        EARTH_RADIUS, 86164.0, NullISLHelper(), ids, **kw)
    sats = _walker("sat", ids, earth) if with_centers else _walker("sat", ids, _null())
    mc = MultiConstellation()
    mc.add_constellation(ground)
    mc.add_constellation(sats)
    if with_centers:
        other = _walker("other", ids, moon, planes=2, per_plane=3)
        mc.add_constellation(other)
    return mc


def _null():
    from dsns.constellation import NullOrbitalCenter
    return NullOrbitalCenter


def _all_pairs(mc):
    ids = mc.satellites.ids
    return list(itertools.permutations(ids, 2))


@pytest.mark.parametrize("with_centers", [True, False])
def test_matches_get_delay_before_and_after_update(with_centers):
    mc = _build(with_centers)
    pairs = _all_pairs(mc)
    for t in (0.0, 1234.5, 98765.0):
        mc.update(t)
        expected = np.array([mc.get_delay(a, b) for a, b in pairs])
        got = mc.get_delays_batch(pairs)
        assert got.shape == (len(pairs),)
        assert np.allclose(got, expected, rtol=1e-12, atol=0.0)


def test_cache_invalidated_on_position_change():
    mc = _build()
    pairs = _all_pairs(mc)
    mc.update(0.0)
    first = mc.get_delays_batch(pairs).copy()
    mc.update(5000.0)
    second = mc.get_delays_batch(pairs)
    assert not np.allclose(first, second)
    assert np.allclose(second, [mc.get_delay(a, b) for a, b in pairs], rtol=1e-12, atol=0.0)


def test_cache_invalidated_on_add_constellation():
    mc = _build()
    mc.update(0.0)
    pairs = _all_pairs(mc)
    mc.get_delays_batch(pairs)
    # New ids must not collide with the existing ones
    extra = _walker("extra", _offset_ids(mc), _null(), planes=1, per_plane=2)
    mc.add_constellation(extra)
    with pytest.raises(RuntimeError):
        mc.get_delays_batch(pairs)
    mc.update(10.0)
    pairs = _all_pairs(mc)
    assert np.allclose(mc.get_delays_batch(pairs), [mc.get_delay(a, b) for a, b in pairs],
                       rtol=1e-12, atol=0.0)


def _offset_ids(mc):
    h = IDHelper()
    for _ in range(len(mc.satellites)):
        h.assign_id(f"dummy_{_}")
    return h


def test_empty_input():
    mc = _build()
    assert mc.get_delays_batch([]).shape == (0,)
    mc.update(0.0)
    assert mc.get_delays_batch([]).shape == (0,)
    assert mc.get_delays_batch(np.empty((0, 2), dtype=int)).shape == (0,)


def test_numpy_array_input():
    mc = _build()
    mc.update(0.0)
    pairs = _all_pairs(mc)[:10]
    assert np.allclose(mc.get_delays_batch(np.array(pairs)), [mc.get_delay(a, b) for a, b in pairs],
                       rtol=1e-12, atol=0.0)


def test_unknown_sat_id_raises():
    mc = _build()
    mc.update(0.0)
    good = mc.satellites.ids[0]
    with pytest.raises(ValueError, match="not found"):
        mc.get_delays_batch([(good, good + 10_000)])
    with pytest.raises(ValueError, match="not found"):
        mc.get_delays_batch([(good, mc.satellites.ids[1]), (-5, good)])


def test_malformed_pairs_raise():
    mc = _build()
    mc.update(0.0)
    with pytest.raises(ValueError):
        mc.get_delays_batch([(1, 2, 3)])
    with pytest.raises(ValueError):
        mc.get_delays_batch([1, 2])


def test_before_update_positions_raises():
    mc = _build()
    ids = mc.satellites.ids
    with pytest.raises(RuntimeError):
        mc.get_delays_batch([(ids[0], ids[1])])
