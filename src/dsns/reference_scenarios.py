import numpy as np

from dsns.constellation import WalkerConstellation, GroundConstellation, NullISLHelper, PlanetOrbitalCenter, FixedISLHelper
from dsns.multiconstellation import MultiConstellation, GroundILLHelper, OcclusionILLHelper
from dsns.helpers import (
    IDHelper,
    EARTH_RADIUS,
    EARTH_ROTATION_PERIOD,
    EARTH_ORBITAL_PERIOD,
    EARTH_ORBITAL_RADIUS,
    MARS_RADIUS,
    MARS_ROTATION_PERIOD,
    MARS_ORBITAL_PERIOD,
    MARS_ORBITAL_RADIUS,
    MARS_MASS,
    MOON_RADIUS,
    MOON_ROTATION_PERIOD,
    MOON_ORBITAL_PERIOD,
    MOON_ORBITAL_RADIUS,
    MOON_MASS,
    get_semi_major_axis,
)
from dsns.message import Link
from dsns.transmission import LinkTransmissionActor, MessageLocationTracker
from dsns.traffic_sim import MultiPointToPointTrafficActor

GROUND_STATIONS_DSN = np.array([
    (35.244, -121.890, 0.0), # DSS 14, Goldstone
    (-35.221, 148.981, 0.0), # DSS 43, Canberra
    (40.241, -9.248, 0.0), # DSS 63, Madrid
]) # From https://deepspace.jpl.nasa.gov/dsndocs/810-005/301/301K.pdf

class EarthObservationMultiConstellation(MultiConstellation):
    """
    Implements the "Earth Observation" scenario from the CCSDS DTN reference scenarios.

    Nodes:
        - Mission Control Center (Earth)
        - Payload Control Center (Earth)
        - Ground Station 1 (Earth)
        - Ground Station 2 (Earth)
        - Earth Observation Satellite (Earth Orbit)
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        ground_stations = np.array([
            GROUND_STATIONS_DSN[0], # Mission Control Center
            GROUND_STATIONS_DSN[1], # Payload Control Center
            GROUND_STATIONS_DSN[0], # Ground Station 1
            GROUND_STATIONS_DSN[1], # Ground Station 2
        ])
        ground_links = [
            (0, 2),
            (1, 3),
            (0, 3),
            (1, 2),
        ]

        self.ground_constellation = GroundConstellation(
            name="ground",
            ground_station_positions=ground_stations,
            host_radius=EARTH_RADIUS,
            rotation_period=EARTH_ROTATION_PERIOD,
            isl_helper=FixedISLHelper(ground_links),
            id_helper=self.id_helper,
        )

        self.satellite_constellation = WalkerConstellation(
            name="satellite",
            num_planes=1,
            sats_per_plane=1,
            inclination=90.0,
            semi_major_axis=7000e3 + EARTH_RADIUS,
            eccentricity=0.0,
            ascending_node_arc=360.0,
            phase_offset=0.0,
            isl_helper=NullISLHelper(),
            id_helper=self.id_helper
        )

        ill_helper = GroundILLHelper(
            self.ground_constellation.satellites.ids[2:],
            self.satellite_constellation.satellites.ids,
            min_elevation=10.0,
        )

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.satellite_constellation)
        self.add_ill_helper(ill_helper)


class EarthObservationTransmissionActor(LinkTransmissionActor):
    """
    Transmission actor for the Earth Observation scenario.
    This actor handles bandwidth between nodes in the simulation.
    """

    def __init__(self, constellation: EarthObservationMultiConstellation, message_location_tracker: MessageLocationTracker):
        default_bandwidth = 0 # No link bandwidth by default
        per_link_bandwidth = dict()

        ground_ids = constellation.ground_constellation.satellites.ids
        satellite_id = constellation.satellite_constellation.satellites.ids[0]

        # Ground systems
        per_link_bandwidth[Link(ground_ids[0], ground_ids[2])] = 100e6 // 8 # 100 Mbps
        per_link_bandwidth[Link(ground_ids[2], ground_ids[0])] = 100e6 // 8 # 100 Mbps
        per_link_bandwidth[Link(ground_ids[0], ground_ids[3])] = 100e6 // 8 # 100 Mbps
        per_link_bandwidth[Link(ground_ids[3], ground_ids[0])] = 100e6 // 8 # 100 Mbps
        per_link_bandwidth[Link(ground_ids[1], ground_ids[2])] = 100e6 // 8 # 100 Mbps
        per_link_bandwidth[Link(ground_ids[2], ground_ids[1])] = 100e6 // 8 # 100 Mbps
        per_link_bandwidth[Link(ground_ids[1], ground_ids[3])] = 1e9 // 8 # 1 Gbps
        per_link_bandwidth[Link(ground_ids[3], ground_ids[1])] = 1e9 // 8 # 1 Gbps

        # Satellite
        per_link_bandwidth[Link(satellite_id, ground_ids[2])] = 8e6 // 8 # 8 Mbps
        per_link_bandwidth[Link(satellite_id, ground_ids[3])] = 10e9 // 8 # 10 Gbps
        per_link_bandwidth[Link(ground_ids[2], satellite_id)] = 64e3 // 8 # 64 kbps
        per_link_bandwidth[Link(ground_ids[3], satellite_id)] = 10e9 // 8 # 10 Gbps to prevent retransmission exhaust. Initially 512 bps - this should be unidirectional link
        # TODO: Add support for unidirectional links, remove the 512-bps link

        super().__init__(
            default_bandwidth=default_bandwidth,
            per_link_bandwidth=per_link_bandwidth,
            buffer_if_link_busy=True,
            reroute_on_link_down=True,
            message_location_tracker=message_location_tracker
        )


class EarthObservationTrafficActor(MultiPointToPointTrafficActor):
    """
    Traffic actor for the Earth Observation scenario.
    This actor generates TC/TM/Payload messages between the ground stations and the satellite.
    """

    def __init__(self, constellation: EarthObservationMultiConstellation, update_interval: float = 600, reliable_messages: bool = False):
        tc_source = constellation.ground_constellation.satellites.ids[0]
        tc_destination = constellation.satellite_constellation.satellites.ids[0]
        tc_size = 512 # 512 bytes
        tc_interval = 300 # 5 minutes

        tm_source = tc_destination
        tm_destination = tc_source
        tm_size = 64e3 # 64 kB
        tm_interval = 60 # 1 minute

        payload_source = constellation.satellite_constellation.satellites.ids[0]
        payload_destination = constellation.ground_constellation.satellites.ids[1]
        payload_size = 10e6 # 10 MB
        payload_interval = 11.5 # 11.5 seconds

        super().__init__(
            message_config=[
                ("TC", tc_source, tc_destination, tc_size, tc_interval),
                ("TM", tm_source, tm_destination, tm_size, tm_interval),
                ("Payload", payload_source, payload_destination, payload_size, payload_interval),
            ],
            update_interval=update_interval,
            reliable_messages=reliable_messages,
        )


class LunarCommunicationMultiConstellation(MultiConstellation):
    """
    Implements the "Lunar Communication" scenario from the CCSDS DTN reference scenarios.

    Earth nodes:
        - Base Control Center
        - User Control Center
        - Rover Control Center
        - Relay Control Center x2
        - Ground Station x2

    Lunar nodes:
        - Lunar Gateway
        - Relay Satellite x2
        - Lunar Base
        - Rover
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.earth_orbital_center = PlanetOrbitalCenter(
            name="earth",
            center=(0., 0., 0.),
            radius=EARTH_ORBITAL_RADIUS,
            rotation_period=EARTH_ORBITAL_PERIOD,
            angle_offset=0.0,
        )

        self.moon_orbital_center = PlanetOrbitalCenter(
            name="moon",
            center=self.earth_orbital_center,
            radius=MOON_ORBITAL_RADIUS,
            rotation_period=MOON_ORBITAL_PERIOD,
            angle_offset=0.0,
        )

        ground_stations = np.array([
            GROUND_STATIONS_DSN[0], # Base Control Center
            GROUND_STATIONS_DSN[0], # User Control Center
            GROUND_STATIONS_DSN[0], # Rover Control Center
            GROUND_STATIONS_DSN[0], # Relay Control Center 1
            GROUND_STATIONS_DSN[1], # Relay Control Center 2
            GROUND_STATIONS_DSN[0], # Ground Station 1
            GROUND_STATIONS_DSN[1], # Ground Station 2
        ])
        ground_links = [
            (0, 3),
            (0, 4),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 6),
            (4, 5),
        ]

        self.ground_constellation = GroundConstellation(
            name="ground",
            ground_station_positions=ground_stations,
            host_radius=EARTH_RADIUS,
            rotation_period=EARTH_ROTATION_PERIOD,
            isl_helper=FixedISLHelper(ground_links),
            id_helper=self.id_helper,
            orbital_center=self.earth_orbital_center,
        )

        lunar_ground_stations = np.array([
            (0.0, 0.0, 0.0), # TODO Lunar Base
            (0.0, 180.0, 0.0), # TODO Rover
        ])

        self.lunar_ground_constellation = GroundConstellation(
            name="lunar_ground",
            ground_station_positions=lunar_ground_stations,
            host_radius=MOON_RADIUS,
            rotation_period=MOON_ROTATION_PERIOD,
            isl_helper=NullISLHelper(),
            id_helper=self.id_helper,
            orbital_center=self.moon_orbital_center,
        )

        self.lunar_relay_constellation = WalkerConstellation(
            name="lunar_relay",
            num_planes=1,
            sats_per_plane=2,
            inclination=90.0,
            semi_major_axis=get_semi_major_axis(3600*4, MOON_MASS), # TODO real values
            eccentricity=0.0,
            ascending_node_arc=360.0,
            phase_offset=0.0,
            isl_helper=NullISLHelper(),
            id_helper=self.id_helper,
            orbital_center=self.moon_orbital_center,
        )

        self.lunar_gateway_constellation = WalkerConstellation(
            name="lunar_gateway",
            num_planes=1,
            sats_per_plane=1,
            inclination=90.0,
            semi_major_axis=get_semi_major_axis(3600*8, MOON_MASS), # TODO real values
            eccentricity=0.0,
            ascending_node_arc=360.0,
            phase_offset=0.0,
            isl_helper=NullISLHelper(),
            id_helper=self.id_helper,
            orbital_center=self.moon_orbital_center,
        )

        # Earth-Moon ILLs
        earth_relay_ill_helper_a = OcclusionILLHelper(
            self.ground_constellation.satellites.ids[-2:-1],
            self.lunar_relay_constellation.satellites.ids[:1],
            EARTH_RADIUS - 1e3,
            MOON_RADIUS,
            min_elevation=10.0,
        )
        earth_relay_ill_helper_b = OcclusionILLHelper(
            self.ground_constellation.satellites.ids[-1:],
            self.lunar_relay_constellation.satellites.ids[1:],
            EARTH_RADIUS - 1e3,
            MOON_RADIUS,
            min_elevation=10.0,
        )
        earth_gateway_ill_helper = OcclusionILLHelper(
            self.ground_constellation.satellites.ids[-2:],
            self.lunar_gateway_constellation.satellites.ids,
            EARTH_RADIUS - 1e3,
            MOON_RADIUS,
            min_elevation=10.0,
        )
        earth_base_ill_helper = OcclusionILLHelper(
            self.ground_constellation.satellites.ids[-2:],
            self.lunar_ground_constellation.satellites.ids[:1],
            EARTH_RADIUS - 1e3,
            MOON_RADIUS - 1e3,
            min_elevation=10.0,
        )

        # Lunar ILLs
        gateway_relay_ill_helper = OcclusionILLHelper(
            self.lunar_gateway_constellation.satellites.ids,
            self.lunar_relay_constellation.satellites.ids,
            MOON_RADIUS,
            MOON_RADIUS,
            min_elevation=10.0,
        )
        gateway_ground_ill_helper = OcclusionILLHelper(
            self.lunar_ground_constellation.satellites.ids,
            self.lunar_gateway_constellation.satellites.ids,
            MOON_RADIUS - 1e3,
            MOON_RADIUS - 1e3,
            min_elevation=10.0,
        )
        relay_ground_ill_helper = OcclusionILLHelper(
            self.lunar_ground_constellation.satellites.ids,
            self.lunar_relay_constellation.satellites.ids,
            MOON_RADIUS - 1e3,
            MOON_RADIUS - 1e3,
            min_elevation=10.0,
        )

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.lunar_ground_constellation)
        self.add_constellation(self.lunar_relay_constellation)
        self.add_constellation(self.lunar_gateway_constellation)

        self.add_ill_helper(earth_relay_ill_helper_a)
        self.add_ill_helper(earth_relay_ill_helper_b)
        self.add_ill_helper(earth_gateway_ill_helper)
        self.add_ill_helper(earth_base_ill_helper)
        self.add_ill_helper(gateway_relay_ill_helper)
        self.add_ill_helper(gateway_ground_ill_helper)
        self.add_ill_helper(relay_ground_ill_helper)


class LunarCommunicationTransmissionActor(LinkTransmissionActor):
    """
    Transmission actor for the Lunar Communication scenario.
    This actor handles bandwidth between nodes in the simulation.
    """

    def __init__(self, constellation: LunarCommunicationMultiConstellation, message_location_tracker: MessageLocationTracker):
        default_bandwidth = 0 # No link bandwidth by default
        per_link_bandwidth = dict()

        ground_ids = constellation.ground_constellation.satellites.ids
        gateway_id = constellation.lunar_gateway_constellation.satellites.ids[0]
        relay_ids = constellation.lunar_relay_constellation.satellites.ids
        lunar_base_id = constellation.lunar_ground_constellation.satellites.ids[0]
        rover_id = constellation.lunar_ground_constellation.satellites.ids[1]

        # Ground systems
        for i, j in [
            (0, 3), # Base to Relay1
            (0, 4), # Base to Relay2
            (1, 3), # User to Relay1
            (1, 4), # User to Relay2
            (2, 3), # Rover to Relay1
            (2, 4), # Rover to Relay2
            (3, 4), # Relay1 to Relay2
            (3, 5), # Relay1 to Ground1
            (3, 6), # Relay1 to Ground2
            (4, 6), # Relay2 to Ground2
            (4, 5), # Relay2 to Ground1
        ]:
            per_link_bandwidth[Link(ground_ids[i], ground_ids[j])] = 100e6 // 8
            per_link_bandwidth[Link(ground_ids[j], ground_ids[i])] = 100e6 // 8

        # Ground to Moon
        per_link_bandwidth[Link(ground_ids[5], gateway_id)] = 30e6 // 8
        per_link_bandwidth[Link(gateway_id, ground_ids[5])] = 100e6 // 8
        per_link_bandwidth[Link(ground_ids[6], gateway_id)] = 30e6 // 8
        per_link_bandwidth[Link(gateway_id, ground_ids[6])] = 100e6 // 8
        per_link_bandwidth[Link(ground_ids[5], relay_ids[0])] = 30e6 // 8
        per_link_bandwidth[Link(relay_ids[0], ground_ids[5])] = 100e6 // 8
        per_link_bandwidth[Link(ground_ids[6], relay_ids[1])] = 30e6 // 8
        per_link_bandwidth[Link(relay_ids[1], ground_ids[6])] = 100e6 // 8
        per_link_bandwidth[Link(ground_ids[5], lunar_base_id)] = 5e6 // 8
        per_link_bandwidth[Link(lunar_base_id, ground_ids[5])] = 25e6 // 8
        per_link_bandwidth[Link(ground_ids[6], lunar_base_id)] = 5e6 // 8
        per_link_bandwidth[Link(lunar_base_id, ground_ids[6])] = 25e6 // 8

        # Gateway, relays
        per_link_bandwidth[Link(gateway_id, relay_ids[0])] = 15e6 // 8
        per_link_bandwidth[Link(relay_ids[0], gateway_id)] = 15e6 // 8
        per_link_bandwidth[Link(gateway_id, relay_ids[1])] = 15e6 // 8
        per_link_bandwidth[Link(relay_ids[1], gateway_id)] = 15e6 // 8
        per_link_bandwidth[Link(gateway_id, rover_id)] = 100e6 // 8
        per_link_bandwidth[Link(rover_id, gateway_id)] = 15e6 // 8
        per_link_bandwidth[Link(gateway_id, lunar_base_id)] = 100e6 // 8
        per_link_bandwidth[Link(lunar_base_id, gateway_id)] = 15e6 // 8
        per_link_bandwidth[Link(relay_ids[0], lunar_base_id)] = 100e6 // 8
        per_link_bandwidth[Link(lunar_base_id, relay_ids[0])] = 15e6 // 8
        per_link_bandwidth[Link(relay_ids[1], lunar_base_id)] = 100e6 // 8
        per_link_bandwidth[Link(lunar_base_id, relay_ids[1])] = 15e6 // 8
        per_link_bandwidth[Link(relay_ids[0], rover_id)] = 100e6 // 8
        per_link_bandwidth[Link(rover_id, relay_ids[0])] = 15e6 // 8
        per_link_bandwidth[Link(relay_ids[1], rover_id)] = 100e6 // 8
        per_link_bandwidth[Link(rover_id, relay_ids[1])] = 15e6 // 8

        super().__init__(
            default_bandwidth=default_bandwidth,
            per_link_bandwidth=per_link_bandwidth,
            buffer_if_link_busy=True,
            reroute_on_link_down=True,
            message_location_tracker=message_location_tracker
        )


class LunarCommunicationTrafficActor(MultiPointToPointTrafficActor):
    """
    Traffic actor for the Lunar Communication scenario.
    This actor generates TC/TM/Payload messages between the ground stations and lunar nodes.
    """

    def __init__(self, constellation: LunarCommunicationMultiConstellation, update_interval: float = 600, reliable_messages: bool = False):
        message_config = []
        for asset, earth, moon in [
            ("Gateway", constellation.ground_constellation.satellites.ids[4], constellation.lunar_gateway_constellation.satellites.ids[0]),
            ("User", constellation.ground_constellation.satellites.ids[1], constellation.lunar_gateway_constellation.satellites.ids[0]),
            ("LunarBase", constellation.ground_constellation.satellites.ids[0], constellation.lunar_ground_constellation.satellites.ids[0]),
            ("Rover", constellation.ground_constellation.satellites.ids[2], constellation.lunar_ground_constellation.satellites.ids[1]),
            ("Relay1", constellation.ground_constellation.satellites.ids[3], constellation.lunar_relay_constellation.satellites.ids[0]),
            ("Relay2", constellation.ground_constellation.satellites.ids[4], constellation.lunar_relay_constellation.satellites.ids[1]),
        ]:
            message_config.append(
                (f"TC-{asset}", earth, moon, 512, 300) # 512 bytes, 5 minutes interval
            )
            message_config.append(
                (f"TM-{asset}", moon, earth, 64e3, 60) # 64 kB, 1 minute interval
            )
            if asset in ["LunarBase", "Rover", "User"]:
                message_config.append(
                    (f"Payload-{asset}", moon, earth, 64e3, 1) # 64 kB, 1 second interval
                )

        super().__init__(
            message_config=message_config,
            update_interval=update_interval,
            reliable_messages=reliable_messages,
        )


class MarsCommunicationMultiConstellation(MultiConstellation):
    """
    Implements the "Mars Communication" scenario from the CCSDS DTN reference scenarios.

    Earth nodes:
        - Rover Control Center x2
        - Relay Coordination Center
        - Relay Control Center x2
        - Ground Station x2

    Mars nodes:
        - Rover x2
        - Relay Satellite x3
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.earth_orbital_center = PlanetOrbitalCenter(
            name="earth",
            center=(0., 0., 0.),
            radius=EARTH_ORBITAL_RADIUS,
            rotation_period=EARTH_ORBITAL_PERIOD,
            angle_offset=0.0,
        )

        self.mars_orbital_center = PlanetOrbitalCenter(
            name="mars",
            center=(0., 0., 0.),
            radius=MARS_ORBITAL_RADIUS,
            rotation_period=MARS_ORBITAL_PERIOD,
            angle_offset=180.0,
        )

        ground_stations = np.array([
            GROUND_STATIONS_DSN[0], # Rover 1 Control Center
            GROUND_STATIONS_DSN[0], # Rover 2 Control Center
            GROUND_STATIONS_DSN[0], # Relay Coordination Center
            GROUND_STATIONS_DSN[0], # Relay Control Center 1
            GROUND_STATIONS_DSN[1], # Relay Control Center 2
            GROUND_STATIONS_DSN[0], # Ground Station 1
            GROUND_STATIONS_DSN[1], # Ground Station 2
        ])
        ground_links = [
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (4, 6),
        ]

        self.ground_constellation = GroundConstellation(
            name="ground",
            ground_station_positions=ground_stations,
            host_radius=EARTH_RADIUS,
            rotation_period=EARTH_ROTATION_PERIOD,
            isl_helper=FixedISLHelper(ground_links),
            id_helper=self.id_helper,
            orbital_center=self.earth_orbital_center,
        )

        mars_rovers = np.array([
            (0.0, 0.0, 0.0), # TODO Rover 1
            (0.0, 180.0, 0.0), # TODO Rover 2
        ])

        self.mars_rover_constellation = GroundConstellation(
            name="mars_rover",
            ground_station_positions=mars_rovers,
            host_radius=MARS_RADIUS,
            rotation_period=MARS_ROTATION_PERIOD,
            isl_helper=NullISLHelper(),
            id_helper=self.id_helper,
            orbital_center=self.mars_orbital_center,
        )

        self.mars_relay_low_constellation = WalkerConstellation(
            name="mars_relay_low",
            num_planes=1,
            sats_per_plane=2,
            inclination=90.0,
            semi_major_axis=get_semi_major_axis(3600*4, MARS_MASS), # TODO real values
            eccentricity=0.0,
            ascending_node_arc=360.0,
            phase_offset=0.0,
            isl_helper=NullISLHelper(),
            id_helper=self.id_helper,
            orbital_center=self.mars_orbital_center,
        )

        self.mars_relay_high_constellation = WalkerConstellation(
            name="mars_relay_high",
            num_planes=1,
            sats_per_plane=1,
            inclination=90.0,
            semi_major_axis=get_semi_major_axis(3600*8, MARS_MASS), # TODO real values
            eccentricity=0.0,
            ascending_node_arc=360.0,
            phase_offset=0.0,
            isl_helper=NullISLHelper(),
            id_helper=self.id_helper,
            orbital_center=self.mars_orbital_center,
        )

        # Earth-Mars ILLs
        earth_relay_low_a_ill_helper = OcclusionILLHelper(
            self.ground_constellation.satellites.ids[-2:-1],
            self.mars_relay_low_constellation.satellites.ids[:1],
            EARTH_RADIUS - 1e3,
            MARS_RADIUS,
            min_elevation=10.0,
        )
        earth_relay_low_b_ill_helper = OcclusionILLHelper(
            self.ground_constellation.satellites.ids[-1:],
            self.mars_relay_low_constellation.satellites.ids[1:],
            EARTH_RADIUS - 1e3,
            MARS_RADIUS,
            min_elevation=10.0,
        )
        earth_relay_high_ill_helper = OcclusionILLHelper(
            self.ground_constellation.satellites.ids[-2:],
            self.mars_relay_high_constellation.satellites.ids,
            EARTH_RADIUS - 1e3,
            MARS_RADIUS,
            min_elevation=10.0,
        )

        # Mars ILLs
        mars_relay_low_ill_helper = OcclusionILLHelper(
            self.mars_rover_constellation.satellites.ids,
            self.mars_relay_low_constellation.satellites.ids,
            MARS_RADIUS - 1e3,
            MARS_RADIUS - 1e3,
            min_elevation=10.0,
        )
        mars_relay_high_ill_helper = OcclusionILLHelper(
            self.mars_rover_constellation.satellites.ids,
            self.mars_relay_high_constellation.satellites.ids,
            MARS_RADIUS - 1e3,
            MARS_RADIUS - 1e3,
            min_elevation=10.0,
        )

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.mars_rover_constellation)
        self.add_constellation(self.mars_relay_low_constellation)
        self.add_constellation(self.mars_relay_high_constellation)
        self.add_ill_helper(earth_relay_low_a_ill_helper)
        self.add_ill_helper(earth_relay_low_b_ill_helper)
        self.add_ill_helper(earth_relay_high_ill_helper)
        self.add_ill_helper(mars_relay_low_ill_helper)
        self.add_ill_helper(mars_relay_high_ill_helper)


class MarsCommunicationTransmissionActor(LinkTransmissionActor):
    """
    Transmission actor for the Mars Communication scenario.
    This actor handles bandwidth between nodes in the simulation.
    """

    def __init__(self, constellation: MarsCommunicationMultiConstellation, message_location_tracker: MessageLocationTracker):
        default_bandwidth = 0 # No link bandwidth by default
        per_link_bandwidth = dict()

        ground_ids = constellation.ground_constellation.satellites.ids
        mars_rover_ids = constellation.mars_rover_constellation.satellites.ids
        relay_low_ids = constellation.mars_relay_low_constellation.satellites.ids
        relay_high_id = constellation.mars_relay_high_constellation.satellites.ids[0]

        # Ground systems
        for i, j in [
            (0, 2), # Rover Control 1 to Relay Coordination
            (1, 2), # Rover Control 2 to Relay Coordination
            (2, 3), # Relay Coordination to Relay 1
            (2, 4), # Relay Coordination to Relay 2
            (3, 4), # Relay 1 to Relay 2
            (3, 5), # Relay 1 to Ground Station 1
            (3, 6), # Relay 1 to Ground Station 2
            (4, 5), # Relay 2 to Ground Station 1
            (4, 6), # Relay 2 to Ground Station 2
        ]:
            per_link_bandwidth[Link(ground_ids[i], ground_ids[j])] = 100e6 // 8
            per_link_bandwidth[Link(ground_ids[j], ground_ids[i])] = 100e6 // 8

        # Ground to Relay
        per_link_bandwidth[Link(ground_ids[5], relay_low_ids[0])] = 30e6 // 8
        per_link_bandwidth[Link(relay_low_ids[0], ground_ids[5])] = 30e6 // 8
        per_link_bandwidth[Link(ground_ids[6], relay_low_ids[1])] = 30e6 // 8
        per_link_bandwidth[Link(relay_low_ids[1], ground_ids[6])] = 30e6 // 8
        per_link_bandwidth[Link(ground_ids[5], relay_high_id)] = 30e6 // 8
        per_link_bandwidth[Link(relay_high_id, ground_ids[5])] = 30e6 // 8
        per_link_bandwidth[Link(ground_ids[6], relay_high_id)] = 30e6 // 8
        per_link_bandwidth[Link(relay_high_id, ground_ids[6])] = 30e6 // 8

        # Relay to Rover
        for relay_id in [ relay_low_ids[0], relay_low_ids[1], relay_high_id ]:
            for rover_id in mars_rover_ids:
                per_link_bandwidth[Link(relay_id, rover_id)] = 1e6 // 8
                per_link_bandwidth[Link(rover_id, relay_id)] = 15e6 // 8

        super().__init__(
            default_bandwidth=default_bandwidth,
            per_link_bandwidth=per_link_bandwidth,
            buffer_if_link_busy=True,
            reroute_on_link_down=True,
            message_location_tracker=message_location_tracker,
        )


class MarsCommunicationTrafficActor(MultiPointToPointTrafficActor):
    """
    Traffic actor for the Mars Communication scenario.
    This actor generates TC/TM/Payload messages between the ground stations and lunar nodes.
    """

    def __init__(self, constellation: MarsCommunicationMultiConstellation, update_interval: float = 600, reliable_messages: bool = False):
        message_config = []
        for asset, earth, mars in [
            ("Relay1", constellation.ground_constellation.satellites.ids[5], constellation.mars_relay_low_constellation.satellites.ids[0]),
            ("Relay2", constellation.ground_constellation.satellites.ids[6], constellation.mars_relay_high_constellation.satellites.ids[0]),
            ("Relay3", constellation.ground_constellation.satellites.ids[5], constellation.mars_relay_low_constellation.satellites.ids[1]),
            ("Rover1", constellation.ground_constellation.satellites.ids[0], constellation.mars_rover_constellation.satellites.ids[0]),
            ("Rover2", constellation.ground_constellation.satellites.ids[1], constellation.mars_rover_constellation.satellites.ids[1]),
        ]:
            message_config.append(
                (f"TC-{asset}", earth, mars, 512, 300) # 512 bytes, 5 minutes interval
            )
            message_config.append(
                (f"TM-{asset}", mars, earth, 64e3, 60) # 64 kB, 1 minute interval
            )
            if asset in ["Rover1", "Rover2"]:
                message_config.append(
                    (f"Payload-{asset}", mars, earth, 64e3, 1) # 64 kB, 1 second interval
                )

        super().__init__(
            message_config=message_config,
            update_interval=update_interval,
            reliable_messages=reliable_messages,
        )

