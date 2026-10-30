"""Shared builders for the baseline geometry tests."""
import numpy as np

from dsns.constellation import GroundConstellation, NullISLHelper, WalkerConstellation, WalkerISLHelper
from dsns.helpers import EARTH_RADIUS, EARTH_ROTATION_PERIOD, IDHelper
from dsns.multiconstellation import MultiConstellation


def make_walker(num_planes=4, sats_per_plane=10, inclination=53.0, altitude=550e3, eccentricity=1e-32,
                ascending_node_arc=360.0, phase_offset=11 * 360 / 40, id_helper=None, name="test-walker",
                inter_layer_links=0, intra_layer_links=True, **kwargs):
    return WalkerConstellation(
        name=name, num_planes=num_planes, sats_per_plane=sats_per_plane, inclination=inclination,
        semi_major_axis=altitude + EARTH_RADIUS, eccentricity=eccentricity,
        ascending_node_arc=ascending_node_arc, phase_offset=phase_offset,
        isl_helper=WalkerISLHelper(num_planes=num_planes, sats_per_plane=sats_per_plane,
                                   intra_layer_links=intra_layer_links, inter_layer_links=inter_layer_links),
        id_helper=id_helper or IDHelper(), **kwargs,
    )


def make_ground(positions, id_helper, name="test-ground"):
    return GroundConstellation(
        name=name, ground_station_positions=np.array(positions, dtype=float), host_radius=EARTH_RADIUS,
        rotation_period=EARTH_ROTATION_PERIOD, isl_helper=NullISLHelper(), id_helper=id_helper,
    )


def make_multi(*constellations, duration=1000.0):
    multi = MultiConstellation()
    multi.duration = duration
    for c in constellations:
        multi.add_constellation(c)
    return multi
