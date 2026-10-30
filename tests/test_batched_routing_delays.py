"""
Routing data providers take link weights from MultiConstellation.get_delays_batch. These tests check the batched
delays agree with get_delay to within 1e-12 (relative) and that routing results match the per-link path.
"""

import itertools
import math

import pytest

from dsns.message_actors import BestEffortRoutingDataProvider, LookaheadRoutingDataProvider, _link_delays
from dsns.multiconstellation import MultiConstellation
from dsns.presets import EarthMoonMarsMultiConstellation, IridiumMultiConstellation, MultiLayerMultiConstellation

REL_TOL = 1e-12


def per_link(mobility: MultiConstellation) -> MultiConstellation:
    """Make the mobility model use the per-link fallback (as a subclass overriding get_delay would)."""
    base = type(mobility)
    mobility.__class__ = type(f"PerLink{base.__name__}", (base,),
                              {"get_delay": lambda self, a, b: base.get_delay(self, a, b)})
    return mobility


@pytest.mark.parametrize("preset", [IridiumMultiConstellation, MultiLayerMultiConstellation, EarthMoonMarsMultiConstellation])
def test_batched_delays_match_get_delay(preset):
    mobility = preset()
    worst = 0.0
    for time in [0.0, 1234.5, 86400.0]:
        mobility.update(time)
        links = list(mobility.links)
        assert links
        batched = _link_delays(mobility, links)
        for (a, b), delay in zip(links, batched):
            reference = mobility.get_delay(a, b)
            worst = max(worst, abs(delay - reference) / reference)
    assert worst <= REL_TOL, worst


def snapshot(provider, sat_ids):
    return {
        (s, d): (provider.get_next_hop(s, d), provider.get_distance(s, d))
        for s, d in itertools.product(sat_ids, repeat=2)
    }


def assert_same_routes(a: dict, b: dict):
    assert a.keys() == b.keys()
    for key in a:
        (hop_a, dist_a), (hop_b, dist_b) = a[key], b[key]
        assert hop_a == hop_b, key
        if dist_a is None or dist_b is None:
            assert dist_a == dist_b, key
        else:
            assert math.isclose(dist_a, dist_b, rel_tol=REL_TOL), (key, dist_a, dist_b)


def test_best_effort_routes_match_per_link():
    batched, reference = IridiumMultiConstellation(), per_link(IridiumMultiConstellation())
    sat_ids = [sat.sat_id for sat in batched.satellites][::8]
    providers = [BestEffortRoutingDataProvider(), BestEffortRoutingDataProvider()]
    for time in [0.0, 15.0, 30.0, 600.0, 3600.0]:
        for mobility, provider in zip((batched, reference), providers):
            mobility.update(time)
            if time == 0.0:
                provider.initialize(mobility, time)
            else:
                provider.update(mobility, time)
        assert_same_routes(snapshot(providers[0], sat_ids), snapshot(providers[1], sat_ids))


def test_lookahead_routes_match_per_link():
    batched, reference = IridiumMultiConstellation(), per_link(IridiumMultiConstellation())
    sat_ids = [sat.sat_id for sat in batched.satellites][::8]
    providers = [LookaheadRoutingDataProvider(resolution=60.0, num_steps=5) for _ in range(2)]
    for mobility, provider in zip((batched, reference), providers):
        mobility.update(0.0)
        provider.initialize(mobility, 0.0)
    for time in [0.0, 120.0, 400.0]:
        for mobility, provider in zip((batched, reference), providers):
            mobility.update(time)
            provider.update(mobility, time)
        assert_same_routes(snapshot(providers[0], sat_ids), snapshot(providers[1], sat_ids))


def test_overridden_get_delay_is_respected():
    class SlowLinks(IridiumMultiConstellation):
        def get_delay(self, sat1, sat2):
            return super().get_delay(sat1, sat2) + 1.0

    mobility = SlowLinks()
    mobility.update(0.0)
    links = list(mobility.links)[:20]
    delays = _link_delays(mobility, links)
    assert delays == [mobility.get_delay(a, b) for a, b in links]
    assert all(d > 1.0 for d in delays)
