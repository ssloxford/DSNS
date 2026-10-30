"""Baseline behaviour of MultiConstellation (positions, links, distances) and the ground ILL helper."""
import unittest

import numpy as np

from dsns.constellation import OrbitalCenter, Satellite, Satellites
from dsns.helpers import EARTH_RADIUS, IDHelper
from dsns.multiconstellation import FixedILLHelper, GroundILLHelper, MultiConstellation

from _helpers import make_ground, make_multi, make_walker


class TestMultiConstellationBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ids = IDHelper()
        cls.walker = make_walker(id_helper=ids)
        cls.ground = make_ground([[0, 0, 0], [10, 20, 0]], ids)
        cls.multi = make_multi(cls.walker, cls.ground)
        cls.multi.update(0.0)

    def test_satellites_and_positions_are_concatenated_in_order(self):
        self.assertEqual(len(self.multi.satellites), 42)
        self.assertEqual(self.multi.satellite_positions.shape, (42, 3))
        np.testing.assert_array_equal(self.multi.satellite_positions[:40], self.walker.satellite_positions)
        np.testing.assert_array_equal(self.multi.satellite_positions[40:], self.ground.satellite_positions)

    def test_links_are_isls_plus_ills(self):
        self.assertEqual(self.multi.links, self.multi.isls + self.multi.ills)
        self.assertEqual(len(self.multi.isls), 80)  # 40 intra + 40 inter, one each per satellite
        self.assertEqual(self.multi.ills, [])

    def test_has_link_is_symmetric(self):
        a, b = self.multi.isls[0]
        self.assertTrue(self.multi.has_link(a, b))
        self.assertTrue(self.multi.has_link(b, a))
        self.assertFalse(self.multi.has_link(a, a))

    def test_distance_and_delay(self):
        a, b = self.multi.isls[0]
        pa = self.multi.satellites.by_id(a).position
        pb = self.multi.satellites.by_id(b).position
        d = self.multi.get_distance(a, b)
        self.assertAlmostEqual(d, np.linalg.norm(pa - pb), places=3)
        self.assertAlmostEqual(self.multi.get_delay(a, b), d / self.multi.speed_of_light, places=15)
        self.assertEqual(self.multi.get_distance(a, b), self.multi.get_distance(b, a))

    def test_unknown_satellite_raises(self):
        with self.assertRaises(ValueError):
            self.multi.get_distance(10 ** 6, 10 ** 6 + 1)

    def test_time_update_moves_satellites_not_ground_below_horizon_links(self):
        multi = make_multi(make_walker(id_helper=IDHelper()))
        multi.update(0.0)
        before = multi.satellite_positions.copy()
        multi.update(5.0)
        self.assertEqual(multi.time, 5.0)
        self.assertFalse(np.array_equal(before, multi.satellite_positions))

    def test_fixed_ill_helper(self):
        ids = IDHelper()
        walker = make_walker(2, 3, id_helper=ids, name="a")
        other = make_walker(2, 3, id_helper=ids, name="b")
        multi = make_multi(walker, other)
        ill = (walker.satellites.ids[0], other.satellites.ids[0])
        multi.add_ill_helper(FixedILLHelper([ill]))
        multi.update(0.0)
        self.assertEqual(multi.ills, [ill])
        self.assertTrue(multi.has_link(*ill))

    def test_delay_between_different_orbital_centers(self):
        mobility = MultiConstellation()
        a = Satellite("a", 91, "a", OrbitalCenter("a", np.array([1000., 0, 0])))
        b = Satellite("b", 7, "b", OrbitalCenter("b", np.array([1003., 4, 0])))
        a.position = np.array([0., 0, 2])
        b.position = np.array([0., 0, 14])
        mobility.satellites = Satellites([b, a])
        # |(b + cb) - (a + ca)| = |(3, 4, 12)| = 13
        self.assertAlmostEqual(mobility.get_distance(91, 7), 13.0, places=9)
        self.assertAlmostEqual(mobility.get_delay(7, 91), 13.0 / mobility.speed_of_light, places=15)


class TestGroundILLHelperAgainstReference(unittest.TestCase):
    """GroundILLHelper must return the same links as a naive per-pair elevation/distance calculation."""

    @classmethod
    def setUpClass(cls):
        ids = IDHelper()
        walker = make_walker(id_helper=ids)
        ground = make_ground([[0, 0, 0], [10, 20, 0], [-30, 50, 0], [45, -60, 0], [60, 120, 0]], ids)
        cls.ground_ids = list(ground.satellites.ids)
        cls.space_ids = list(walker.satellites.ids)
        cls.mobility = make_multi(walker, ground)
        cls.mobility.update(0.0)

    def _reference(self, helper):
        sats, pos = self.mobility.satellites, self.mobility.satellite_positions
        index = {s.sat_id: i for i, s in enumerate(sats)}
        out = []
        for gi in (index[s] for s in helper.satellites_ground):
            normal = pos[gi] / np.linalg.norm(pos[gi])
            for si in (index[s] for s in helper.satellites_space):
                if helper.min_elevation is not None:
                    v = pos[si] - pos[gi]
                    elevation = 90.0 - np.rad2deg(np.arccos(np.dot(normal, v) / np.linalg.norm(v)))
                    if elevation < helper.min_elevation:
                        continue
                if helper.max_distance is not None and np.linalg.norm(pos[gi] - pos[si]) > helper.max_distance:
                    continue
                out.append((sats[gi].sat_id, sats[si].sat_id))
        return out

    def _check(self, helper):
        got = helper.get_ills(self.mobility.satellites, self.mobility.satellite_positions)
        self.assertEqual(sorted(got), sorted(self._reference(helper)))
        return got

    def test_min_elevation_only(self):
        self.assertGreater(len(self._check(GroundILLHelper(self.ground_ids, self.space_ids, min_elevation=25.0))), 0)

    def test_max_distance_only(self):
        self._check(GroundILLHelper(self.ground_ids, self.space_ids, max_distance=1500e3))

    def test_both_constraints(self):
        self._check(GroundILLHelper(self.ground_ids, self.space_ids, min_elevation=10.0, max_distance=2000e3))

    def test_no_filters_returns_all_pairs(self):
        ills = self._check(GroundILLHelper(self.ground_ids, self.space_ids))
        self.assertEqual(len(ills), len(self.ground_ids) * len(self.space_ids))

    def test_after_time_update(self):
        self.addCleanup(self.mobility.update, 0.0)
        helper = GroundILLHelper(self.ground_ids, self.space_ids, min_elevation=25.0)
        self._check(helper)
        self.mobility.update(300.0)
        self._check(helper)

    def test_multiconstellation_uses_ill_helper(self):
        helper = GroundILLHelper(self.ground_ids, self.space_ids, min_elevation=25.0)
        self.mobility.add_ill_helper(helper)
        self.addCleanup(self.mobility.ill_helpers.remove, helper)
        self.addCleanup(self.mobility.update, 0.0)
        self.mobility.update(0.0)
        self.assertEqual(sorted(self.mobility.ills), sorted(self._reference(helper)))


class TestGroundILLKnownGeometry(unittest.TestCase):
    """Hand-computed answers on a flat arrangement of points near one ground station."""

    def setUp(self):
        # Row order differs from both ID order and requested link order.
        self.satellites = Satellites([Satellite(str(sid), sid, "geometry") for sid in (91, 7, 42, 300)])
        self.ground = np.array([EARTH_RADIUS, 0.0, 0.0])
        self.positions = np.array([
            self.ground + [500e3, 0, 0],   # 91: overhead
            self.ground,                   # 7: the ground station
            self.ground + [-500e3, 0, 0],  # 42: below horizon
            self.ground + [0, 500e3, 0],   # 300: on horizon
        ])

    def test_overhead_horizon_and_below_horizon(self):
        # (a satellite exactly on the horizon is a floating point coin toss, so it is not asserted on)
        helper = GroundILLHelper([7], [42, 300, 91], min_elevation=1.0)
        self.assertEqual(helper.get_ills(self.satellites, self.positions), [(7, 91)])
        helper = GroundILLHelper([7], [42, 300, 91], min_elevation=-1.0)
        self.assertEqual(helper.get_ills(self.satellites, self.positions), [(7, 300), (7, 91)])

    def test_elevation_just_inside_and_outside_limit(self):
        for elevation, expected in ((24.999, []), (25.001, [(7, 91)])):
            with self.subTest(elevation=elevation):
                angle = np.deg2rad(elevation)
                self.positions[0] = self.ground + 500e3 * np.array([np.sin(angle), np.cos(angle), 0])
                helper = GroundILLHelper([7], [91], min_elevation=25.0)
                self.assertEqual(helper.get_ills(self.satellites, self.positions), expected)

    def test_distance_limit_is_inclusive(self):
        for distance, expected in ((499999., [(7, 91)]), (500000., [(7, 91)])):
            with self.subTest(distance=distance):
                self.positions[0] = self.ground + [distance, 0, 0]
                helper = GroundILLHelper([7], [91], max_distance=500e3)
                self.assertEqual(helper.get_ills(self.satellites, self.positions), expected)

    def test_both_constraints_must_hold(self):
        self.positions[0] = self.ground + [600e3, 0, 0]  # too far
        self.positions[3] = self.ground + [400e3, 0, 0]  # passes both
        helper = GroundILLHelper([7], [42, 91, 300], min_elevation=25.0, max_distance=500e3)
        self.assertEqual(helper.get_ills(self.satellites, self.positions), [(7, 300)])

    def test_requested_order_is_preserved(self):
        helper = GroundILLHelper([300, 7], [42, 91])
        self.assertEqual(helper.get_ills(self.satellites, self.positions),
                         [(300, 42), (300, 91), (7, 42), (7, 91)])

    def test_ground_station_is_not_linked_to_itself(self):
        helper = GroundILLHelper([7], [7, 91])
        self.assertEqual(helper.get_ills(self.satellites, self.positions), [(7, 91)])

    def test_indices_are_cached_between_calls(self):
        helper = GroundILLHelper([7], [91, 42], min_elevation=25.0)
        self.assertEqual(helper.get_ills(self.satellites, self.positions), [(7, 91)])
        cached = (helper.index_ground, helper.index_space)
        moved = self.positions.copy()
        moved[[0, 2]] = moved[[2, 0]]
        self.assertEqual(helper.get_ills(self.satellites, moved), [(7, 42)])
        self.assertIs(helper.index_ground, cached[0])
        self.assertIs(helper.index_space, cached[1])


if __name__ == "__main__":
    unittest.main()
