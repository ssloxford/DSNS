"""Walker Constellation Visualizer

This example demonstrates the use of the visualizer to display a Walker Constellation.

The parameters of the constellation match an early version of the Starlink constellation.
The constellation is composed of 72 orbital planes, each with 22 satellites.
Satellites are placed in a circular orbit with an altitude of 550 km.
The inclination of each orbital plane is 53 degrees.
There is a phase offset of (65 / 72) * 360 degrees between each orbital plane to reduce collisions.

Satellites are connected to their neighbors in the same orbital plane, and to adjacent satellites in the
next and previous orbital planes.
"""

from dsns.visualizer import ConstellationVisualizer
from dsns.constellation import WalkerISLHelper, WalkerConstellation
from dsns.helpers import IDHelper, EARTH_RADIUS, EARTH_ROTATION_PERIOD

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Demo script for the Walker constellation visualizer.")
    parser.add_argument("--num-planes", help="The number of orbital planes in the constellation", type=int, default=72)
    parser.add_argument("--sats-per-plane", help="The number of satellites per orbital plane", type=int, default=22)
    parser.add_argument("--altitude", help="The altitude of the orbits (km)", type=float, default=550)
    parser.add_argument("--inclination", help="The inclination of the orbits (degrees)", type=float, default=53.0)
    parser.add_argument("--phase-offset", help="The phase offset n between adjacent planes ((n / num_planes) * 360)", type=float, default=65)

    return parser

if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    id_helper = IDHelper()

    num_planes = args.num_planes
    sats_per_plane = args.sats_per_plane
    altitude = args.altitude * 1000
    inclination = args.inclination
    phase_offset = (args.phase_offset / 72) * 360.0

    walker_isl_helper = WalkerISLHelper(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        intra_layer_links=True,
        inter_layer_links=2
    )

    walker_constellation = WalkerConstellation(
        name="walker",
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        inclination=inclination,
        semi_major_axis=altitude + EARTH_RADIUS,
        eccentricity=0.0,
        ascending_node_arc=360.0,
        phase_offset=phase_offset,
        isl_helper = walker_isl_helper,
        id_helper = id_helper
    )

    visualiser = ConstellationVisualizer(
        walker_constellation,
        show_isls=True,
        time_scale=100.0,
        space_scale=1e-6,
        host_radius=EARTH_RADIUS,
        host_rotation_period=EARTH_ROTATION_PERIOD,
        host_materials=(
            "assets/2k_earth_daymap.jpg",
            "assets/2k_earth_normal_map.jpg",
            "assets/2k_earth_metallic_roughness_map.jpg"
        )
    )

    visualiser.run_simulation()
    visualiser.run_viewer()