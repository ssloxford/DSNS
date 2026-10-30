"""Baseline behaviour of WalkerConstellation and WalkerISLHelper (positions, orbit geometry, ISL structure)."""
import unittest

import numpy as np
from PyAstronomy.pyasl import KeplerEllipse

from dsns.constellation import NullISLHelper, WalkerConstellation
from dsns.helpers import EARTH_RADIUS, IDHelper

from _helpers import make_walker


class TestWalkerAgainstIndependentOracle(unittest.TestCase):
    """Positions must equal a from-scratch KeplerEllipse computation of the documented Walker layout."""

    # (num_planes, sats_per_plane, inclination, ascending_node_arc, phase_offset, right_ascending_node)
    CONFIGS = [
        (4, 10, 53.0, 360.0, 11 * 360 / 40, 0.0),
        (1, 1, 0.0, 360.0, 0.0, 0.0),
        (2, 1, 97.6, 360.0, 0.0, 0.0),
        (6, 4, 53.0, 180.0, 30.0, 0.0),
        (3, 8, 86.4, 360.0, 45.0, 123.4),
    ]
    TIMES = [0.0, 1.0, -500.0, 3600.0, 86400.0]

    @staticmethod
    def _oracle(walker, num_planes, sats_per_plane, inclination, arc, phase_offset, raan, time):
        period = walker.period
        out = np.zeros((num_planes * sats_per_plane, 3))
        for plane in range(num_planes):
            ellipse = KeplerEllipse(per=period, a=walker.semi_major_axis, e=0.0, i=inclination,
                                    Omega=(arc / num_planes) * plane + raan, w=0.0)
            for sat in range(sats_per_plane):
                offset = (period / sats_per_plane) * sat + (phase_offset / 360) * plane * period
                out[plane * sats_per_plane + sat] = ellipse.xyzPos(time + offset)
        return out

    def test_positions_match_oracle(self):
        ids = IDHelper()
        for planes, spp, inc, arc, phase, raan in self.CONFIGS:
            walker = make_walker(planes, spp, inc, ascending_node_arc=arc, phase_offset=phase,
                                 id_helper=ids, name=f"w-{planes}-{spp}", right_ascending_node=raan,
                                 intra_layer_links=spp > 1)
            for t in self.TIMES:
                with self.subTest(planes=planes, spp=spp, time=t):
                    walker.update_positions(t)
                    expected = self._oracle(walker, planes, spp, inc, arc, phase, raan, t)
                    np.testing.assert_allclose(walker.satellite_positions, expected, atol=1e-3, rtol=0)
                    np.testing.assert_array_equal([s.position for s in walker.satellites],
                                                  walker.satellite_positions)

    def test_satellite_naming_and_order_is_plane_major(self):
        walker = make_walker(3, 4)
        self.assertEqual(walker.num_sats, 12)
        self.assertEqual(walker.satellites[0].name, "test-walker_0_0")
        self.assertEqual(walker.satellites[5].name, "test-walker_1_1")
        self.assertEqual(walker.satellites[11].name, "test-walker_2_3")


class TestWalkerOrbitGeometry(unittest.TestCase):
    def _walker(self, eccentricity=0.0):
        return WalkerConstellation(
            name="boundary", num_planes=1, sats_per_plane=1, inclination=0.0,
            semi_major_axis=EARTH_RADIUS + 550e3, eccentricity=eccentricity,
            ascending_node_arc=360.0, phase_offset=0.0,
            isl_helper=NullISLHelper(), id_helper=IDHelper(),
        )

    def test_circular_quarter_orbits_and_constant_radius(self):
        walker = self._walker()
        r = EARTH_RADIUS + 550e3
        for fraction, direction in ((0, [1, 0, 0]), (0.25, [0, 1, 0]), (0.5, [-1, 0, 0]),
                                    (0.75, [0, -1, 0]), (1, [1, 0, 0])):
            with self.subTest(fraction=fraction):
                walker.update_positions(fraction * walker.period)
                np.testing.assert_allclose(walker.satellite_positions[0], r * np.array(direction),
                                           atol=1e-3, rtol=0)

    def test_orbit_is_periodic(self):
        walker = make_walker(2, 3)
        walker.update_positions(123.0)
        before = walker.satellite_positions.copy()
        walker.update_positions(123.0 + walker.period)
        np.testing.assert_allclose(walker.satellite_positions, before, atol=1e-2, rtol=0)

    def test_radius_is_semi_major_axis_for_circular_orbit(self):
        walker = make_walker(3, 5)
        for t in (0.0, 777.0, 5000.0):
            walker.update_positions(t)
            np.testing.assert_allclose(np.linalg.norm(walker.satellite_positions, axis=1),
                                       walker.semi_major_axis, rtol=1e-9)

    def test_eccentric_orbit_matches_kepler_ellipse(self):
        walker = self._walker(eccentricity=0.1)
        ellipse = KeplerEllipse(per=walker.period, a=walker.semi_major_axis, e=0.1, i=0.0, Omega=0.0, w=0.0)
        for time in np.linspace(0.0, walker.period, 9):
            walker.update_positions(time)
            np.testing.assert_allclose(walker.satellite_positions[0], ellipse.xyzPos(time), atol=1e-3, rtol=0)

    def test_orbital_period_matches_kepler_third_law(self):
        walker = make_walker(1, 1)
        # LEO at 550 km is about 95.5 minutes
        self.assertAlmostEqual(walker.period / 60, 95.5, delta=0.5)


class TestWalkerISLHelper(unittest.TestCase):
    def _isls(self, planes, spp, **kwargs):
        walker = make_walker(planes, spp, **kwargs)
        walker.update(0.0)
        return walker, walker.isls

    def test_intra_plane_only(self):
        walker, isls = self._isls(3, 4, inter_layer_links=None)
        self.assertEqual(len(isls), 12)
        ids = walker.satellites.ids
        self.assertIn((ids[0], ids[1]), isls)
        self.assertIn((ids[3], ids[0]), isls)  # ring wraps

    def test_inter_plane_links_with_offset(self):
        walker, isls = self._isls(3, 4, inter_layer_links=1, intra_layer_links=False)
        ids = walker.satellites.ids
        self.assertEqual(len(isls), 12)
        self.assertIn((ids[0], ids[5]), isls)  # plane 0 sat 0 -> plane 1 sat 1
        self.assertIn((ids[11], ids[0]), isls)  # last plane wraps to first plane, sat 3 + 1 -> 0

    def test_disable_cross_seam_links(self):
        from dsns.constellation import WalkerISLHelper
        walker = make_walker(3, 4)
        walker.isl_helper = WalkerISLHelper(3, 4, intra_layer_links=False, inter_layer_links=0,
                                            disable_cross_seam_links=True)
        walker.update(0.0)
        self.assertEqual(len(walker.isls), 8)
        last_plane = set(walker.satellites.ids[8:])
        self.assertTrue(all(a not in last_plane for a, _ in walker.isls))

    def test_isls_are_cached(self):
        walker, isls = self._isls(2, 3)
        self.assertIs(walker.isl_helper.get_isls(walker.satellites, walker.satellite_positions), isls)

    def test_isl_lengths_inside_the_constellation_are_physical(self):
        walker, isls = self._isls(6, 11, inclination=86.4, ascending_node_arc=180.0, phase_offset=0.0,
                                  inter_layer_links=0, altitude=780e3)
        index = {s.sat_id: i for i, s in enumerate(walker.satellites)}
        pos = walker.satellite_positions
        intra = [np.linalg.norm(pos[index[a]] - pos[index[b]]) for a, b in isls
                 if index[a] // 11 == index[b] // 11]
        # neighbours in a plane of 11 satellites at 780 km: chord of 2*pi/11 on radius a
        expected = 2 * walker.semi_major_axis * np.sin(np.pi / 11)
        np.testing.assert_allclose(intra, expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
