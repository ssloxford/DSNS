"""
BestEffortRoutingDataProvider updates edge weights in place when the link list is unchanged.
These tests check that the results are exactly the same as rebuilding the graph from scratch.
"""

import itertools
from dataclasses import dataclass

from dsns.message_actors import BestEffortRoutingDataProvider
from dsns.presets import IridiumMultiConstellation


def snapshot(provider: BestEffortRoutingDataProvider, sat_ids: list[int]):
    """All-pairs routing results of a provider."""
    result = {}
    for source, destination in itertools.product(sat_ids, repeat=2):
        result[(source, destination)] = (
            provider.get_next_hop(source, destination),
            provider.get_distance(source, destination),
            provider.get_path_cost(source, destination),
        )
    for source in sat_ids:
        result[source] = (provider.get_neighbors(source), provider.get_neighbors(source, max_distance=0.02))
    return result


def fresh_snapshot(mobility, sat_ids: list[int], time: float):
    provider = BestEffortRoutingDataProvider()
    provider.initialize(mobility, time)
    return snapshot(provider, sat_ids)


def in_place_used(provider: BestEffortRoutingDataProvider) -> bool:
    return provider._BestEffortRoutingDataProvider__graph_links_distinct is True


def test_constellation_matches_rebuild():
    mobility = IridiumMultiConstellation()
    mobility.update(0.0)
    # All-pairs over every node would be slow; use every 12th node (ground stations and satellites)
    sat_ids = [sat.sat_id for sat in mobility.satellites][::12]

    provider = BestEffortRoutingDataProvider()
    provider.initialize(mobility, 0.0)
    assert snapshot(provider, sat_ids) == fresh_snapshot(mobility, sat_ids, 0.0)

    changed = unchanged = 0
    previous_links = list(mobility.links)
    for time in [1.0, 2.0, 3.0, 10.0, 10.5, 300.0, 301.0, 900.0]:
        mobility.update(time)
        if mobility.links == previous_links:
            unchanged += 1
        else:
            changed += 1
        previous_links = list(mobility.links)

        provider.update(mobility, time)
        assert snapshot(provider, sat_ids) == fresh_snapshot(mobility, sat_ids, time), f"mismatch at t={time}"

    # Both code paths must have been exercised
    assert changed > 0 and unchanged > 0


@dataclass
class StubSatellite:
    sat_id: int


class StubMobility:
    """Minimal mobility model with explicit links and delays."""

    def __init__(self, num_sats: int):
        self.satellites = [StubSatellite(i) for i in range(num_sats)]
        self.links: list[tuple[int, int]] = []
        self.delays: dict[frozenset, float] = {}

    def get_delay(self, sat1: int, sat2: int) -> float:
        return self.delays[frozenset((sat1, sat2))]


def run_sequence(mobility: StubMobility, steps: list[tuple[list[tuple[int, int]], dict]], expect_in_place: bool):
    sat_ids = [sat.sat_id for sat in mobility.satellites]
    provider = BestEffortRoutingDataProvider()
    mobility.links, mobility.delays = list(steps[0][0]), dict(steps[0][1])
    provider.initialize(mobility, 0.0)
    for time, (links, delays) in enumerate(steps):
        mobility.links, mobility.delays = list(links), dict(delays)
        provider.update(mobility, float(time))
        assert snapshot(provider, sat_ids) == fresh_snapshot(mobility, sat_ids, float(time)), f"mismatch at step {time}"
    assert in_place_used(provider) == expect_in_place


def test_ties_match_rebuild():
    # Square 0-1-2-3-0 with a diagonal: many equal-cost paths, so next hops depend on tie-breaking.
    links = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    equal = { frozenset(link): 0.01 for link in links }
    equal[frozenset((0, 2))] = 0.02
    other = dict(equal)
    other[frozenset((0, 1))] = 0.005
    steps = [
        (links, equal),
        (links, equal),
        (links, other),
        (links, equal),
        (list(reversed(links)), equal), # same set, different order: must rebuild
        (list(reversed(links)), other),
        (links[:-1], equal), # link removed
        (links[:-1], equal),
        (links, equal), # link added back
        (links, other),
    ]
    run_sequence(StubMobility(5), steps, expect_in_place=True)


def test_duplicate_links_fall_back_to_rebuild():
    # Both directions of a link listed (parallel edges in the graph): weights cannot be set in place.
    links = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)]
    first = { frozenset(link): 0.01 for link in links }
    second = dict(first)
    second[frozenset((0, 1))] = 0.05
    steps = [(links, first), (links, second), (links, first), (links, second)]
    run_sequence(StubMobility(4), steps, expect_in_place=False)
