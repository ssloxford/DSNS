from typing import Optional, Any

import numpy as np
import datetime

from dsns.constellation import Constellation, WalkerISLHelper, WalkerConstellation, GroundConstellation, NullISLHelper, PlanetOrbitalCenter, TLEConstellation, AdHocISLHelper, FixedISLHelper, FixedConstellation
from dsns.multiconstellation import MultiConstellation, GroundILLHelper, OcclusionILLHelper, DistanceILLHelper
from dsns.helpers import (
    IDHelper,
    SatID,
    EARTH_RADIUS,
    EARTH_ROTATION_PERIOD,
    EARTH_ORBITAL_PERIOD,
    EARTH_ORBITAL_RADIUS,
    EARTH_MASS,
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
    GROUND_STATIONS_AWS,
    GROUND_STATIONS_UNIFORM,
    get_semi_major_axis
)
from dsns.visualizer import MultiConstellationVisualizer


GROUND_STATIONS_DSN = np.array([
    (35.244, -121.890, 0.0), # DSS 14, Goldstone
    (-35.221, 148.981, 0.0), # DSS 43, Canberra
    (40.241, -9.248, 0.0), # DSS 63, Madrid
]) # From https://deepspace.jpl.nasa.gov/dsndocs/810-005/301/301K.pdf


def ground_constellation(id_helper: IDHelper, aws: bool = False, connected: bool = False, reduced: bool = False, **kwargs) -> Constellation:
    """
    Build a ground constellation using the locations of AWS ground stations.

    Parameters:
        id_helper: ID helper to use.
        aws: Whether to use the locations of AWS ground stations.
        connected: Whether to connect the ground stations using fixed links.
        reduced: If true, limit to ground stations within 60 degrees of the equator.
        **kwargs: Additional arguments to pass to the GroundConstellation constructor.
    """

    if connected:
        raise NotImplementedError("Connected ground constellations are no longer supported.")

    isl_helper = NullISLHelper()

    ground_station_positions = GROUND_STATIONS_UNIFORM if not aws else GROUND_STATIONS_AWS
    if reduced:
        ground_station_positions = np.array([ g for g in ground_station_positions if abs(g[0]) < 60.0 ])

    constellation_args: dict[str, Any] = dict(
        name="ground",
        ground_station_positions = ground_station_positions,
        host_radius=EARTH_RADIUS,
        rotation_period=EARTH_ROTATION_PERIOD,
        isl_helper=isl_helper,
        id_helper=id_helper,
    )
    constellation_args.update(kwargs)

    constellation = GroundConstellation(**constellation_args)

    return constellation

def gps_constellation(id_helper: IDHelper, **kwargs) -> Constellation:
    """
    Build a GPS constellation.

    Parameters:
        id_helper: ID helper to use.
        **kwargs: Additional arguments to pass to the WalkerConstellation constructor.
    """

    constellation_args: dict[str, Any] = dict(
        name="gps",
        num_planes=6,
        sats_per_plane=4,
        inclination=55.0,
        semi_major_axis=20200e3 + EARTH_RADIUS,
        eccentricity=0.0,
        ascending_node_arc=360.0,
        phase_offset=(2 / 6) * 360.0,
        isl_helper=NullISLHelper(),
        id_helper=id_helper
    )
    constellation_args.update(kwargs)

    constellation = WalkerConstellation(**constellation_args)

    return constellation

def iridium_constellation(id_helper: IDHelper, **kwargs) -> Constellation:
    """
    Build an Iridium constellation.

    Parameters:
        id_helper: ID helper to use.
        **kwargs: Additional arguments to pass to the WalkerConstellation constructor.
    """

    isl_helper = WalkerISLHelper(
        num_planes=6,
        sats_per_plane=11,
        intra_layer_links=True,
        inter_layer_links=5,
        disable_cross_seam_links=True,
    )

    constellation_args: dict[str, Any] = dict(
        name="walker",
        num_planes=6,
        sats_per_plane=11,
        inclination=86.4,
        semi_major_axis=781e3 + EARTH_RADIUS,
        eccentricity=0.0,
        ascending_node_arc=180.0,
        phase_offset=(3 / 6) * 360.0,
        isl_helper=isl_helper,
        id_helper=id_helper
    )
    constellation_args.update(kwargs)

    constellation = WalkerConstellation(**constellation_args)

    return constellation

def small_iridium_constellation(id_helper: IDHelper, **kwargs) -> Constellation:
    """
    Build a constellation similar to Iridium, but with only 2 planes and 4 satellites per plane.

    Parameters:
        id_helper: ID helper to use.
        **kwargs: Additional arguments to pass to the WalkerConstellation constructor.
    """

    isl_helper = WalkerISLHelper(
        num_planes=2,
        sats_per_plane=4,
        intra_layer_links=True,
        inter_layer_links=2,
        disable_cross_seam_links=True,
    )

    constellation_args: dict[str, Any] = dict(
        name="walker",
        num_planes=2,
        sats_per_plane=4,
        inclination=86.4,
        semi_major_axis=781e3 + EARTH_RADIUS,
        eccentricity=0.0,
        ascending_node_arc=180.0,
        phase_offset=(3 / 6) * 360.0,
        isl_helper=isl_helper,
        id_helper=id_helper
    )
    constellation_args.update(kwargs)

    constellation = WalkerConstellation(**constellation_args)

    return constellation

def starlink_constellation(id_helper: IDHelper, use_tle: bool = False, **kwargs) -> Constellation:
    """
    Build a Starlink constellation, based on the proposal for phase 1, or on TLE data.

    Parameters:
        id_helper: ID helper to use.
        use_tle: Whether to use TLE data.
        **kwargs: Additional arguments to pass to the Constellation constructor.
    """

    if not use_tle:
        isl_helper = WalkerISLHelper(
            num_planes=72,
            sats_per_plane=22,
            intra_layer_links=True,
            inter_layer_links=2
        )

        constellation_args: dict[str, Any] = dict(
            name="starlink",
            num_planes=72,
            sats_per_plane=22,
            inclination=53.0,
            semi_major_axis=550e3 + EARTH_RADIUS,
            eccentricity=0.0,
            ascending_node_arc=360.0,
            phase_offset=(65 / 72) * 360.0,
            isl_helper=isl_helper,
            id_helper=id_helper
        )
        constellation_args.update(kwargs)

        constellation = WalkerConstellation(**constellation_args)
    else:
        isl_helper = AdHocISLHelper(
            max_range=5e5,
            max_links=4,
            min_altitude=None,
        )

        constellation_args: dict[str, Any] = dict(
            isl_helper=isl_helper,
            id_helper=id_helper,
            ignore_errors=True,
        )
        constellation_args.update(kwargs)

        url_starlink = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
        constellation = TLEConstellation.from_url(
            "tle_starlink",
            url_starlink,
            **constellation_args,
        )

    return constellation

def cubesat_constellation(id_helper: IDHelper, url_cubesats: Optional[str] = None, file_cubesats: Optional[str] = None, epoch: Optional[datetime.datetime] = None, **kwargs) -> Constellation:
    """
    Build a constellation of all CubeSats currently in orbit.

    Parameters:
        id_helper: ID helper to use.
        **kwargs: Additional arguments to pass to the TLEConstellation constructor.
    """
    isl_helper = AdHocISLHelper(
        max_range=2.5e6,
        max_links=None,
        min_altitude=None,
    )

    constellation_args: dict[str, Any] = dict(
        isl_helper=isl_helper,
        id_helper=id_helper,
        ignore_errors=True,
    )
    constellation_args.update(kwargs)

    if file_cubesats is not None and url_cubesats is not None:
        raise Exception("Only one of file_cubesats and url_cubesats can be specified.")

    if file_cubesats is not None:
        constellation = TLEConstellation.from_file(
            "tle_cubesats",
            file_cubesats,
            epoch=epoch,
            **constellation_args,
        )
    else:
        url_cubesats = url_cubesats or "https://celestrak.org/NORAD/elements/gp.php?GROUP=cubesat&FORMAT=tle"
        constellation = TLEConstellation.from_url(
            "tle_cubesats",
            url_cubesats,
            epoch=epoch,
            **constellation_args,
        )

    return constellation

def viasat_constellation(id_helper: IDHelper, **kwargs) -> Constellation:
    """
    Build a Viasat-3 constellation.
    This constellation is composed of 3 geostationary satellites.

    Parameters:
        id_helper: ID helper to use.
        **kwargs: Additional arguments to pass to the WalkerConstellation constructor.
    """

    constellation_args: dict[str, Any] = dict(
        name="viasat",
        num_planes=1,
        sats_per_plane=3,
        inclination=0.0,
        semi_major_axis=get_semi_major_axis(EARTH_ROTATION_PERIOD, EARTH_MASS),
        eccentricity=0.0,
        ascending_node_arc=0.0,
        phase_offset=0.0,
        isl_helper=NullISLHelper(),
        id_helper=id_helper,
        right_ascending_node=-170.06,
    )
    constellation_args.update(kwargs)

    constellation = WalkerConstellation(**constellation_args)

    return constellation

def mpower_constellation(id_helper: IDHelper, **kwargs) -> Constellation:
    """
    Build an O3b mPOWER constellation.
    This constellation is composed of 7 satellites in MEO, with a semi-major axis of 14444 km.

    Parameters:
        id_helper: ID helper to use.
        **kwargs: Additional arguments to pass to the WalkerConstellation constructor.
    """

    constellation_args: dict[str, Any] = dict(
        name="mpower",
        num_planes=1,
        sats_per_plane=13,
        inclination=0.0,
        semi_major_axis=8000e3 + EARTH_RADIUS,
        eccentricity=0.0,
        ascending_node_arc=0.0,
        phase_offset=0.0,
        isl_helper=NullISLHelper(),
        id_helper=id_helper,
        right_ascending_node=0.0,
    )
    constellation_args.update(kwargs)

    constellation = WalkerConstellation(**constellation_args)

    return constellation

def dsn_constellation(id_helper: IDHelper, **kwargs) -> Constellation:
    """
    Build a Deep Space Network constellation.
    This constellation is composed of 3 ground stations.

    Parameters:
        id_helper: ID helper to use.
        **kwargs: Additional arguments to pass to the GroundConstellation constructor.
    """

    isl_helper = NullISLHelper()

    constellation_args: dict[str, Any] = dict(
        name="dsn",
        ground_station_positions=GROUND_STATIONS_DSN,
        host_radius=EARTH_RADIUS,
        rotation_period=EARTH_ROTATION_PERIOD,
        isl_helper=isl_helper,
        id_helper=id_helper,
    )
    constellation_args.update(kwargs)

    constellation = GroundConstellation(**constellation_args)

    return constellation

def ground_ill_helper(ground_constellation: Constellation, space_constellation: Constellation, min_elevation = 20.0) -> GroundILLHelper:
    """
    Build an inter-layer link helper for a ground constellation and a space constellation.

    Parameters:
        ground_constellation: Ground constellation.
        space_constellation: Space constellation.
        min_elevation: Minimum elevation angle for visibility (degrees).
    """

    ground_ill_helper = GroundILLHelper(
        ground_constellation.satellites.ids,
        space_constellation.satellites.ids,
        min_elevation = min_elevation,
    )

    return ground_ill_helper

def dsn_ill_helper(dsn_constellation: Constellation | list[SatID], space_constellation: Constellation | list[SatID], space_body_radius: float) -> OcclusionILLHelper:
    """
    Build an inter-layer link helper for a DSN constellation and a space constellation.

    Parameters:
        dsn_constellation: DSN constellation.
        space_constellation: Space constellation.
        space_body_radius: Radius of the body around which the space constellation orbits (m).
    """

    dsn_sats = dsn_constellation.satellites.ids if isinstance(dsn_constellation, Constellation) else dsn_constellation
    space_sats = space_constellation.satellites.ids if isinstance(space_constellation, Constellation) else space_constellation

    dsn_ill_helper = OcclusionILLHelper(
        dsn_sats,
        space_sats,
        EARTH_RADIUS - 1e3,
        space_body_radius,
        min_elevation = 6.0,
        max_links = 1,
    )

    return dsn_ill_helper

def interplanetary_ill_helper(constellation_a: Constellation | list[SatID], constellation_b: Constellation | list[SatID], radius_a: float, radius_b: float, max_links: int = 1) -> OcclusionILLHelper:
    """
    Build an inter-layer link helper for a connection between two constellations orbiting different bodies.

    Parameters:
        constellation_a: First constellation.
        constellation_b: Second constellation.
        radius_a: Radius of the body around which the first constellation orbits (m).
        radius_b: Radius of the body around which the second constellation orbits (m).
        max_links: Maximum number of links per satellite.
    """

    sats_a = constellation_a.satellites.ids if isinstance(constellation_a, Constellation) else constellation_a
    sats_b = constellation_b.satellites.ids if isinstance(constellation_b, Constellation) else constellation_b

    dsn_ill_helper = OcclusionILLHelper(
        sats_a,
        sats_b,
        radius_a,
        radius_b,
        max_links=max_links,
    )

    return dsn_ill_helper

def leo_leo_ill_helper(leo_constellation_a: Constellation, leo_constellation_b: Constellation) -> DistanceILLHelper:
    """
    Build an inter-layer link helper for two LEO constellations.

    Parameters:
        leo_constellation_a: First LEO constellation.
        leo_constellation_b: Second LEO constellation.
    """

    leo_leo_ill_helper = DistanceILLHelper(
        leo_constellation_b.satellites.ids,
        leo_constellation_a.satellites.ids,
        max_distance = 5e6,
        max_links = 1,
    )

    return leo_leo_ill_helper


class GroundMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for a ground "constellation" consisting of ground stations only.
    Ground stations are connected to each other using fixed links.
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(self.id_helper, connected=True)

        self.add_constellation(self.ground_constellation)


class GPSMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for a GPS constellation with ground stations.
    GPS satellites are connected to the ground stations based on visibility.
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.gps_constellation = gps_constellation(self.id_helper)

        self.ground_constellation = ground_constellation(self.id_helper)

        gps_ill_helper = GroundILLHelper(
            self.ground_constellation.satellites.ids,
            self.gps_constellation.satellites.ids,
            min_elevation=10.0,
        )

        self.add_constellation(self.gps_constellation)
        self.add_constellation(self.ground_constellation)
        self.add_ill_helper(gps_ill_helper)


class IridiumMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for the Iridium constellation connected to ground stations.
    """

    def __init__(self, iridium_kwargs: dict[str, Any] = {}):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(self.id_helper)
        self.iridium_constellation = iridium_constellation(self.id_helper, **iridium_kwargs)

        self.ground_ill_helper = ground_ill_helper(self.ground_constellation, self.iridium_constellation, min_elevation=8.2)

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.iridium_constellation)
        self.add_ill_helper(self.ground_ill_helper)


class StarlinkMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for a Walker constellation corresponding to a proposal for phase 1 of the Starlink
    constellation.
    """

    def __init__(self, starlink_kwargs: dict[str, Any] = {}):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(self.id_helper)
        self.starlink_constellation = starlink_constellation(self.id_helper, **starlink_kwargs)

        self.ground_ill_helper = ground_ill_helper(self.ground_constellation, self.starlink_constellation, min_elevation=25.0)

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.starlink_constellation)
        self.add_ill_helper(self.ground_ill_helper)


class StarlinkTLEMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for the Starlink constellation using TLE data.
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.starlink_constellation = starlink_constellation(self.id_helper, use_tle=True)

        self.add_constellation(self.starlink_constellation)


class CubesatMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for all CubeSats currently in orbit using TLE data.
    """

    def __init__(self, url_cubesats: Optional[str] = None, file_cubesats: Optional[str] = None, epoch: Optional[datetime.datetime] = None):
        super().__init__()

        if file_cubesats is not None and url_cubesats is not None:
            raise Exception("Only one of file_cubesats and url_cubesats can be specified.")

        self.id_helper = IDHelper()

        # 2023-10-03T00:00:00Z
        epoch = epoch or datetime.datetime(2024, 3, 26, 0, 0, 0, tzinfo=datetime.timezone.utc)

        if url_cubesats is None:
            file_cubesats = file_cubesats or "examples/assets/cubesats.txt"
            self.cubesat_constellation = cubesat_constellation(self.id_helper, file_cubesats=file_cubesats, epoch=epoch)
        else:
            self.cubesat_constellation = cubesat_constellation(self.id_helper, url_cubesats=url_cubesats, epoch=epoch)

        self.ground_constellation = ground_constellation(self.id_helper)

        self.ground_ill_helper = ground_ill_helper(self.ground_constellation, self.cubesat_constellation, min_elevation=8.2)

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.cubesat_constellation)
        self.add_ill_helper(self.ground_ill_helper)


class LeoLeoMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation containing a ground constellation, a Starlink constellation, and an Iridium
    constellation.
    All constellations are connected to the ground.
    Starlink and OneWeb satellites are also connected to each other.
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(self.id_helper)
        self.starlink_constellation = starlink_constellation(self.id_helper)
        self.iridium_constellation = iridium_constellation(self.id_helper)

        self.ground_starlink_ill_helper = ground_ill_helper(self.ground_constellation, self.starlink_constellation, min_elevation=25.0)
        self.ground_iridium_ill_helper = ground_ill_helper(self.ground_constellation, self.iridium_constellation, min_elevation=8.2)
        self.starlink_iridium_ill_helper = leo_leo_ill_helper(self.starlink_constellation, self.iridium_constellation)

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.starlink_constellation)
        self.add_constellation(self.iridium_constellation)
        self.add_ill_helper(self.ground_starlink_ill_helper)
        self.add_ill_helper(self.ground_iridium_ill_helper)
        self.add_ill_helper(self.starlink_iridium_ill_helper)


class LeoMeoMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation containing a ground constellation, an Iridium constellation, and an mPOWER constellation.
    All constellations are connected to the ground.
    Iridium and mPOWER satellites are also connected to each other.
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(self.id_helper)
        self.iridium_constellation = iridium_constellation(self.id_helper)
        self.mpower_constellation = mpower_constellation(self.id_helper)

        self.ground_iridium_ill_helper = ground_ill_helper(self.ground_constellation, self.iridium_constellation, min_elevation=8.2)
        self.ground_mpower_ill_helper = ground_ill_helper(self.ground_constellation, self.mpower_constellation, min_elevation=5.0)
        self.iridium_mpower_ill_helper = ground_ill_helper(self.iridium_constellation, self.mpower_constellation)

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.iridium_constellation)
        self.add_constellation(self.mpower_constellation)
        self.add_ill_helper(self.ground_iridium_ill_helper)
        self.add_ill_helper(self.ground_mpower_ill_helper)
        self.add_ill_helper(self.iridium_mpower_ill_helper)


class LeoGeoMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation containing a ground constellation, an Iridium constellation, and the Viasat-3 constellation.
    All constellations are connected to the ground.
    Iridium and Viasat-3 satellites are also connected to each other.
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(self.id_helper)
        self.iridium_constellation = iridium_constellation(self.id_helper)
        self.viasat_constellation = viasat_constellation(self.id_helper)

        self.ground_iridium_ill_helper = ground_ill_helper(self.ground_constellation, self.iridium_constellation, min_elevation=8.2)
        self.ground_viasat_ill_helper = ground_ill_helper(self.ground_constellation, self.viasat_constellation, min_elevation=5.0)
        self.iridium_viasat_ill_helper = ground_ill_helper(self.iridium_constellation, self.viasat_constellation)

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.iridium_constellation)
        self.add_constellation(self.viasat_constellation)
        self.add_ill_helper(self.ground_iridium_ill_helper)
        self.add_ill_helper(self.ground_viasat_ill_helper)
        self.add_ill_helper(self.iridium_viasat_ill_helper)


class LeoMeoGeoMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation containing a ground constellation, an Iridium constellation, an mPOWER constellation, and the Viasat-3 constellation.
    All constellations are connected to the ground.
    Iridium, mPOWER, and Viasat-3 satellites are also connected to each other.
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(self.id_helper)
        self.iridium_constellation = iridium_constellation(self.id_helper)
        self.mpower_constellation = mpower_constellation(self.id_helper)
        self.viasat_constellation = viasat_constellation(self.id_helper)

        self.ground_iridium_ill_helper = ground_ill_helper(self.ground_constellation, self.iridium_constellation, min_elevation=8.2)
        self.ground_mpower_ill_helper = ground_ill_helper(self.ground_constellation, self.mpower_constellation, min_elevation=5.0)
        self.ground_viasat_ill_helper = ground_ill_helper(self.ground_constellation, self.viasat_constellation, min_elevation=5.0)
        self.iridium_mpower_ill_helper = ground_ill_helper(self.iridium_constellation, self.mpower_constellation)
        self.mpower_viasat_ill_helper = ground_ill_helper(self.mpower_constellation, self.viasat_constellation)

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.iridium_constellation)
        self.add_constellation(self.mpower_constellation)
        self.add_constellation(self.viasat_constellation)
        self.add_ill_helper(self.ground_iridium_ill_helper)
        self.add_ill_helper(self.ground_mpower_ill_helper)
        self.add_ill_helper(self.ground_viasat_ill_helper)
        self.add_ill_helper(self.iridium_mpower_ill_helper)
        self.add_ill_helper(self.mpower_viasat_ill_helper)


class MultiLayerMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for a multi-layer constellation consisting of a ground constellation, a GPS
    constellation, a Starlink constellation, and an Iridium constellation.
    All constellations are connected to the ground.
    GPS and Iridium satellites are also connected to each other.
    """

    def __init__(self):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(self.id_helper)
        self.gps_constellation = gps_constellation(self.id_helper)
        self.starlink_constellation = starlink_constellation(self.id_helper)
        self.iridium_constellation = iridium_constellation(self.id_helper)

        ground_gps_ill_helper = GroundILLHelper(
            self.ground_constellation.satellites.ids,
            self.gps_constellation.satellites.ids,
            min_elevation=10.0,
        )
        ground_starlink_ill_helper = GroundILLHelper(
            self.ground_constellation.satellites.ids,
            self.starlink_constellation.satellites.ids,
            min_elevation=50.0,
        )
        ground_iridium_ill_helper = GroundILLHelper(
            self.ground_constellation.satellites.ids,
            self.iridium_constellation.satellites.ids,
            min_elevation=10.0,
        )

        gps_iridium_ill_helper = GroundILLHelper(
            self.gps_constellation.satellites.ids,
            self.iridium_constellation.satellites.ids,
            min_elevation=50.0,
        )

        self.add_constellation(self.gps_constellation)
        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.starlink_constellation)
        self.add_constellation(self.iridium_constellation)
        self.add_ill_helper(ground_gps_ill_helper)
        self.add_ill_helper(ground_starlink_ill_helper)
        self.add_ill_helper(ground_iridium_ill_helper)
        self.add_ill_helper(gps_iridium_ill_helper)


class EarthMoonMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for Earth-Mars communication.
    TODO finish description.
    """

    earth_nodes: list[SatID] = []
    moon_nodes: list[SatID] = []
    earth_moon_links: list[tuple[SatID, SatID]] = []

    def __init__(self, num_relays=1, configuration="base"):
        """
        Initialize the multi-constellation.

        Parameters:
            num_relays: Number of relay satellites around the Moon.
            configuration: Configuration to use (base, surface, single, multiple).
        """
        super().__init__()

        print("DEPRECATION WARNING: Use EarthMoonMarsMultiConstellation instead of EarthMoonMultiConstellation.")

        if configuration not in ["base", "surface", "single", "multiple"]:
            raise Exception(f"Unknown configuration: {configuration}")

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

        self.ground_constellation = ground_constellation(
            self.id_helper,
            orbital_center=self.earth_orbital_center,
        )
        self.earth_nodes.extend(self.ground_constellation.satellites.ids)

        self.earth_constellation = iridium_constellation(
            self.id_helper,
            name="earth_constellation",
            orbital_center=self.earth_orbital_center,
        )
        self.earth_nodes.extend(self.earth_constellation.satellites.ids)

        self.deep_space_network = dsn_constellation(
            self.id_helper,
            orbital_center=self.earth_orbital_center,
        )
        self.earth_nodes.extend(self.deep_space_network.satellites.ids)

        if configuration == "base":
            self.lunar_relay = WalkerConstellation(
                name="lunar_relay",
                num_planes=1,
                sats_per_plane=num_relays,
                inclination=0.0,
                semi_major_axis=1.5 * MOON_RADIUS,
                eccentricity=0.0,
                ascending_node_arc=0.0,
                phase_offset=0.0,
                isl_helper=NullISLHelper(),
                id_helper=self.id_helper,
                orbital_center=self.moon_orbital_center,
                host_mass=MOON_MASS,
            )
            self.moon_nodes.extend(self.lunar_relay.satellites.ids)

        self.lunar_constellation = small_iridium_constellation(
            self.id_helper,
            name="lunar_constellation",
            semi_major_axis=(781e3 * 2) + MOON_RADIUS,
            orbital_center=self.moon_orbital_center,
            host_mass=MOON_MASS,
        )
        self.moon_nodes.extend(self.lunar_constellation.satellites.ids)

        self.lunar_ground_constellation = ground_constellation(
            self.id_helper,
            name="lunar_ground",
            aws=True,
            orbital_center=self.moon_orbital_center,
            host_radius=MOON_RADIUS,
            rotation_period=MOON_ROTATION_PERIOD,
        )
        self.moon_nodes.extend(self.lunar_ground_constellation.satellites.ids)

        self.ground_ill_helper = ground_ill_helper(self.ground_constellation, self.earth_constellation, min_elevation=8.2)
        self.dsn_earth_ill_helper = ground_ill_helper(self.deep_space_network, self.earth_constellation, min_elevation=8.2)
        if configuration == "base":
            self.dsn_moon_ill_helper = dsn_ill_helper(self.deep_space_network, self.lunar_relay, MOON_RADIUS)
            self.earth_moon_links.extend([ (e, m) for e in self.deep_space_network.satellites.ids for m in self.lunar_relay.satellites.ids ])
            self.lunar_relay_ill_helper = leo_leo_ill_helper(self.lunar_constellation, self.lunar_relay)
        elif configuration == "surface":
            self.dsn_moon_ill_helper = dsn_ill_helper(self.deep_space_network, self.lunar_ground_constellation.satellites.ids[5:6], MOON_RADIUS - 100)
            self.earth_moon_links.extend([ (e, m) for e in self.deep_space_network.satellites.ids for m in self.lunar_ground_constellation.satellites.ids[5:6] ])
        elif configuration == "single":
            self.dsn_moon_ill_helper = dsn_ill_helper(self.deep_space_network, self.lunar_constellation.satellites.ids[:1], MOON_RADIUS)
            self.earth_moon_links.extend([ (e, m) for e in self.deep_space_network.satellites.ids for m in self.lunar_constellation.satellites.ids[:1] ])
        elif configuration == "multiple":
            self.dsn_moon_ill_helper = dsn_ill_helper(self.deep_space_network, self.lunar_constellation, MOON_RADIUS)
            self.earth_moon_links.extend([ (e, m) for e in self.deep_space_network.satellites.ids for m in self.lunar_constellation.satellites.ids ])
        self.lunar_ground_ill_helper = ground_ill_helper(self.lunar_ground_constellation, self.lunar_constellation, min_elevation=8.2)

        self.add_constellation(self.earth_constellation)
        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.deep_space_network)
        if configuration == "base":
            self.add_constellation(self.lunar_relay)
        self.add_constellation(self.lunar_constellation)
        self.add_constellation(self.lunar_ground_constellation)
        self.add_ill_helper(self.ground_ill_helper)
        self.add_ill_helper(self.dsn_earth_ill_helper)
        self.add_ill_helper(self.dsn_moon_ill_helper)
        if configuration == "base":
            self.add_ill_helper(self.lunar_relay_ill_helper)
        self.add_ill_helper(self.lunar_ground_ill_helper)


class EarthMarsMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for Earth-Mars communication.
    Earth and Mars each have a constellation of satellites.
    There is a deep space link between Earth and Mars.
    """

    earth_nodes: list[SatID] = []
    mars_nodes: list[SatID] = []
    earth_mars_links: list[tuple[SatID, SatID]] = []

    def __init__(self, earth_constellation: str = "iridium", mars_constellation: str = "iridium", num_relays: int = 1):
        """
        Initialize the multi-constellation.

        Parameters:
            earth_constellation: Constellation to use for Earth (iridium or starlink).
            mars_constellation: Constellation to use for Mars (iridium or starlink).
            num_relays: Number of relay satellites around Mars.
        """
        super().__init__()

        print("DEPRECATION WARNING: Use EarthMoonMarsMultiConstellation instead of EarthMarsMultiConstellation.")

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

        if earth_constellation == "iridium":
            self.earth_constellation = iridium_constellation(
                self.id_helper,
                name="earth_constellation",
                orbital_center=self.earth_orbital_center,
            )
        elif earth_constellation == "starlink":
            self.earth_constellation = starlink_constellation(
                self.id_helper,
                name="earth_constellation",
                orbital_center=self.earth_orbital_center,
            )
        else:
            raise Exception(f"Unknown constellation: {earth_constellation}")
        self.earth_nodes.extend(self.earth_constellation.satellites.ids)

        if mars_constellation == "iridium":
            self.mars_constellation = iridium_constellation(
                self.id_helper,
                name="mars_constellation",
                semi_major_axis=781e3 + MARS_RADIUS,
                orbital_center=self.mars_orbital_center,
                host_mass=MARS_MASS,
            )
        elif mars_constellation == "starlink":
            self.mars_constellation = starlink_constellation(
                self.id_helper,
                name="mars_constellation",
                semi_major_axis=550e3 + MARS_RADIUS,
                orbital_center=self.mars_orbital_center,
                host_mass=MARS_MASS,
            )
        else:
            raise Exception(f"Unknown constellation: {mars_constellation}")
        self.mars_nodes.extend(self.mars_constellation.satellites.ids)

        self.ground_constellation = ground_constellation(
            self.id_helper,
            orbital_center=self.earth_orbital_center,
        )
        self.earth_nodes.extend(self.ground_constellation.satellites.ids)

        self.deep_space_network = dsn_constellation(
            self.id_helper,
            orbital_center=self.earth_orbital_center,
        )
        self.earth_nodes.extend(self.deep_space_network.satellites.ids)

        self.mars_relay_constellation = WalkerConstellation(
            name="mars_relay",
            num_planes=1,
            sats_per_plane=num_relays,
            inclination=0.0,
            semi_major_axis=1000e3 + MARS_RADIUS,
            eccentricity=0.0,
            ascending_node_arc=0.0,
            phase_offset=0.0,
            isl_helper=NullISLHelper(),
            id_helper=self.id_helper,
            orbital_center=self.mars_orbital_center,
            host_mass=MARS_MASS,
        )
        self.mars_nodes.extend(self.mars_relay_constellation.satellites.ids)

        self.ground_ill_helper = ground_ill_helper(self.ground_constellation, self.earth_constellation, min_elevation=8.2)
        self.dsn_earth_ill_helper = ground_ill_helper(self.deep_space_network, self.earth_constellation, min_elevation=8.2)
        self.dsn_mars_ill_helper = dsn_ill_helper(self.deep_space_network, self.mars_relay_constellation, MARS_RADIUS)
        self.earth_mars_links.extend([ (e, m) for e in self.deep_space_network.satellites.ids for m in self.mars_relay_constellation.satellites.ids ])
        self.deep_space_mars_ill_helper = leo_leo_ill_helper(self.mars_constellation, self.mars_relay_constellation)

        self.add_constellation(self.earth_constellation)
        self.add_constellation(self.mars_constellation)
        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.deep_space_network)
        self.add_constellation(self.mars_relay_constellation)
        self.add_ill_helper(self.ground_ill_helper)
        self.add_ill_helper(self.dsn_earth_ill_helper)
        self.add_ill_helper(self.dsn_mars_ill_helper)
        self.add_ill_helper(self.deep_space_mars_ill_helper)


class EarthMoonMarsMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation for Earth-Moon-Mars communication.
    """

    earth_nodes: list[SatID] = []
    moon_nodes: list[SatID] = []
    mars_nodes: list[SatID] = []
    moon: bool
    mars: bool
    moon_mars_link: bool

    # needed for rogue actor to target these relays in their attack
    relay_constellations: Constellation = []

    def __init__(self, num_relays: int = 1, moon: bool = True, mars: bool = True, moon_mars_link: bool = True):
        """
        Initialize the multi-constellation.

        Parameters:
            num_relays: Number of relay satellites around the Moon and Mars.
            moon: Include the Moon in the simulation.
            mars: Include Mars in the simulation.
            moon_mars_link: Include a link between the Moon and Mars.
        """
        super().__init__()

        self.moon = moon
        self.mars = mars
        self.moon_mars_link = moon_mars_link

        self.id_helper = IDHelper()

        self.earth_orbital_center = PlanetOrbitalCenter(
            name="earth",
            center=(0., 0., 0.),
            radius=EARTH_ORBITAL_RADIUS,
            rotation_period=EARTH_ORBITAL_PERIOD,
            angle_offset=0.0,
        )

        if moon:
            self.moon_orbital_center = PlanetOrbitalCenter(
                name="moon",
                center=self.earth_orbital_center,
                radius=MOON_ORBITAL_RADIUS,
                rotation_period=MOON_ORBITAL_PERIOD,
                angle_offset=0.0,
            )

        if mars:
            self.mars_orbital_center = PlanetOrbitalCenter(
                name="mars",
                center=(0., 0., 0.),
                radius=MARS_ORBITAL_RADIUS,
                rotation_period=MARS_ORBITAL_PERIOD,
                angle_offset=180.0,
            )

        self.earth_ground = ground_constellation(
            self.id_helper,
            name="earth_ground",
            orbital_center=self.earth_orbital_center,
        )
        self.earth_nodes.extend(self.earth_ground.satellites.ids)

        self.earth_constellation = iridium_constellation(
            self.id_helper,
            name="earth_constellation",
            orbital_center=self.earth_orbital_center,
        )
        self.earth_nodes.extend(self.earth_constellation.satellites.ids)

        self.earth_relay = dsn_constellation(
            self.id_helper,
            orbital_center=self.earth_orbital_center,
        )
        self.earth_nodes.extend(self.earth_relay.satellites.ids)
        self.earth_relay.is_relay = True

        if moon:
            self.moon_ground = ground_constellation(
                self.id_helper,
                name="moon_ground",
                aws=True,
                orbital_center=self.moon_orbital_center,
                host_radius=MOON_RADIUS,
                rotation_period=MOON_ROTATION_PERIOD,
            )
            self.moon_nodes.extend(self.moon_ground.satellites.ids)

            self.moon_constellation = small_iridium_constellation(
                self.id_helper,
                name="moon_constellation",
                semi_major_axis=(781e3 * 2) + MOON_RADIUS,
                orbital_center=self.moon_orbital_center,
                host_mass=MOON_MASS,
            )
            self.moon_nodes.extend(self.moon_constellation.satellites.ids)

            self.moon_relay = WalkerConstellation(
                name="moon_relay",
                num_planes=1,
                sats_per_plane=num_relays,
                inclination=0.0,
                semi_major_axis=1.5 * MOON_RADIUS,
                eccentricity=0.0,
                ascending_node_arc=0.0,
                phase_offset=0.0,
                isl_helper=NullISLHelper(),
                id_helper=self.id_helper,
                orbital_center=self.moon_orbital_center,
                host_mass=MOON_MASS,
            )
            self.moon_nodes.extend(self.moon_relay.satellites.ids)
            self.moon_relay.is_relay = True
        if mars:
            self.mars_ground = ground_constellation(
                self.id_helper,
                name="mars_ground",
                aws=True,
                orbital_center=self.mars_orbital_center,
                host_radius=MARS_RADIUS,
                rotation_period=MARS_ROTATION_PERIOD,
            )
            self.mars_nodes.extend(self.mars_ground.satellites.ids)

            self.mars_constellation = iridium_constellation(
                self.id_helper,
                name="mars_constellation",
                semi_major_axis=781e3 + MARS_RADIUS,
                orbital_center=self.mars_orbital_center,
                host_mass=MARS_MASS,
            )
            self.mars_nodes.extend(self.mars_constellation.satellites.ids)

            self.mars_relay = WalkerConstellation(
                name="mars_relay",
                num_planes=1,
                sats_per_plane=num_relays,
                inclination=0.0,
                semi_major_axis=1000e3 + MARS_RADIUS,
                eccentricity=0.0,
                ascending_node_arc=0.0,
                phase_offset=0.0,
                isl_helper=NullISLHelper(),
                id_helper=self.id_helper,
                orbital_center=self.mars_orbital_center,
                host_mass=MARS_MASS,
            )
            self.mars_relay.is_relay = True
            self.mars_nodes.extend(self.mars_relay.satellites.ids)

        self.earth_ground_ill_helper = ground_ill_helper(self.earth_ground, self.earth_constellation, min_elevation=8.2)
        self.earth_relay_ill_helper = ground_ill_helper(self.earth_relay, self.earth_constellation, min_elevation=8.2)

        if moon:
            self.moon_ground_ill_helper = ground_ill_helper(self.moon_ground, self.moon_constellation, min_elevation=8.2)
            self.moon_relay_ill_helper = leo_leo_ill_helper(self.moon_constellation, self.moon_relay)
            self.earth_moon_ill_helper = dsn_ill_helper(self.earth_relay, self.moon_relay, MOON_RADIUS)

        if mars:
            self.mars_ground_ill_helper = ground_ill_helper(self.mars_ground, self.mars_constellation, min_elevation=8.2)
            self.mars_relay_ill_helper = leo_leo_ill_helper(self.mars_constellation, self.mars_relay)
            self.earth_mars_ill_helper = dsn_ill_helper(self.earth_relay, self.mars_relay, MARS_RADIUS)

        self.add_constellation(self.earth_ground)
        self.add_constellation(self.earth_constellation)
        self.add_constellation(self.earth_relay)
        self.add_ill_helper(self.earth_ground_ill_helper)
        self.add_ill_helper(self.earth_relay_ill_helper)
        if moon:
            self.add_constellation(self.moon_ground)
            self.add_constellation(self.moon_constellation)
            self.add_constellation(self.moon_relay)
            self.add_ill_helper(self.moon_ground_ill_helper)
            self.add_ill_helper(self.moon_relay_ill_helper)

            self.add_ill_helper(self.earth_moon_ill_helper)
        if mars:
            self.add_constellation(self.mars_ground)
            self.add_constellation(self.mars_constellation)
            self.add_constellation(self.mars_relay)
            self.add_ill_helper(self.mars_ground_ill_helper)
            self.add_ill_helper(self.mars_relay_ill_helper)

            self.add_ill_helper(self.earth_mars_ill_helper)

        if moon and mars and moon_mars_link:
            self.moon_mars_ill_helper = interplanetary_ill_helper(self.moon_constellation, self.mars_constellation, MOON_RADIUS, MARS_RADIUS)

            self.add_ill_helper(self.moon_mars_ill_helper)


class FixedMultiConstellation(MultiConstellation):
    """
    Preset multi-constellation with a fixed number of unmoving satellites, connected via fixed links.
    Satellites are distributed evenly on a circle around the origin, with a radius of 1e6 meters.
    Satellites are connected to each other in a ring.
    """

    def __init__(self, num_sats: int = 10):
        """
        Initialize the multi-constellation.

        Parameters:
            num_sats: Number of satellites to create.
        """

        super().__init__()

        self.id_helper = IDHelper()

        self.positions = np.array([
            (np.cos(2 * np.pi * i / num_sats) * 1e6, np.sin(2 * np.pi * i / num_sats) * 1e6, 0.0)
            for i in range(num_sats)
        ])
        self.links = [
            (i, (i + 1) % num_sats)
            for i in range(num_sats)
        ]

        self.constellation = FixedConstellation(
            name="ground",
            satellite_positions=self.positions,
            isl_helper=FixedISLHelper(self.links),
            id_helper=self.id_helper,
        )

        self.add_constellation(self.constellation)


class EarthVisualizer(MultiConstellationVisualizer):
    """
    Preset visualizer for rendering Earth satellites.
    """

    def __init__(
        self,
        multi_constellation: MultiConstellation,
        time_scale: float = 100.0,
        space_scale: float = 1e-6,
        earth_materials: Optional[tuple[str, str, str]] = None,
        viewport_size: tuple[int, int] = (800, 600),
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ):
        """
        Initialize the visualizer.

        Parameters:
            multi_constellation: Multi-constellation to visualize.
            time_scale: Time scale factor.
            space_scale: Space scale factor.
            earth_materials: Materials to use for the Earth.
            viewport_size: Size of the viewport.
            bg_color: Background color.
        """
        super().__init__(
            multi_constellation,
            time_scale=time_scale,
            space_scale=space_scale,
            viewport_size=viewport_size,
            bg_color=bg_color,
        )

        earth_color = (107/255, 147/255, 214/255)
        self.add_planet(
            radius=EARTH_RADIUS,
            rotation_period=EARTH_ROTATION_PERIOD,
            materials=earth_materials,
            color=earth_color,
        )

    def run_simulation(self):
        super().run_simulation()

    def run_viewer(self):
        super().run_viewer()


class EarthMarsVisualizer(MultiConstellationVisualizer):
    """
    Preset visualizer for rendering Earth-Mars communication.
    """

    def __init__(
        self,
        multi_constellation: EarthMarsMultiConstellation,
        time_scale: float = 100.0,
        space_scale: float = 1e-6,
        interplanetary_scale: float = 1e-10,
        earth_materials: Optional[tuple[str, str, str]] = None,
        mars_material: Optional[str] = None,
        viewport_size: tuple[int, int] = (800, 600),
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ):
        """
        Initialize the visualizer.

        Parameters:
            multi_constellation: Multi-constellation to visualize.
            time_scale: Time scale factor.
            space_scale: Space scale factor.
            interplanetary_scale: Interplanetary scale factor.
            earth_materials: Materials to use for the Earth.
            mars_material: Material to use for Mars.
            viewport_size: Size of the viewport.
            bg_color: Background color.
        """
        super().__init__(
            multi_constellation,
            time_scale=time_scale,
            space_scale=space_scale,
            interplanetary_scale=interplanetary_scale,
            viewport_size=viewport_size,
            bg_color=bg_color,
        )

        earth_color = (107/255, 147/255, 214/255)
        mars_color = (193/255, 68/255, 14/255)

        self.add_planet(
            radius=EARTH_RADIUS,
            rotation_period=EARTH_ROTATION_PERIOD,
            materials=earth_materials,
            color=earth_color,
            center=multi_constellation.earth_orbital_center,
        )

        mars_materials = (
            mars_material,
            None,
            None,
        ) if mars_material is not None else None
        self.add_planet(
            radius=MARS_RADIUS,
            rotation_period=MARS_ROTATION_PERIOD,
            materials=mars_materials,
            color=mars_color,
            center=multi_constellation.mars_orbital_center,
        )

    def run_simulation(self):
        super().run_simulation()

    def run_viewer(self):
        super().run_viewer()


class EarthMoonVisualizer(MultiConstellationVisualizer):
    """
    Preset visualizer for rendering Earth-Moon communication.
    """

    def __init__(
        self,
        multi_constellation: EarthMoonMultiConstellation,
        time_scale: float = 100.0,
        space_scale: float = 1e-6,
        interplanetary_scale: float = 1e-10,
        earth_materials: Optional[tuple[str, str, str]] = None,
        moon_material: Optional[str] = None,
        viewport_size: tuple[int, int] = (800, 600),
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ):
        """
        Initialize the visualizer.

        Parameters:
            multi_constellation: Multi-constellation to visualize.
            time_scale: Time scale factor.
            space_scale: Space scale factor.
            interplanetary_scale: Interplanetary scale factor.
            earth_materials: Materials to use for the Earth.
            moon_material: Material to use for the Moon.
            viewport_size: Size of the viewport.
            bg_color: Background color.
        """
        super().__init__(
            multi_constellation,
            time_scale=time_scale,
            space_scale=space_scale,
            interplanetary_scale=interplanetary_scale,
            viewport_size=viewport_size,
            bg_color=bg_color,
        )

        earth_color = (107/255, 147/255, 214/255)
        moon_color = (184/255, 184/255, 184/255)

        self.add_planet(
            radius=EARTH_RADIUS,
            rotation_period=EARTH_ROTATION_PERIOD,
            materials=earth_materials,
            color=earth_color,
            center=multi_constellation.earth_orbital_center,
        )

        moon_materials = (
            moon_material,
            None,
            None,
        ) if moon_material is not None else None
        self.add_planet(
            radius=MOON_RADIUS,
            rotation_period=MOON_ROTATION_PERIOD,
            materials=moon_materials,
            color=moon_color,
            center=multi_constellation.moon_orbital_center,
        )

    def run_simulation(self):
        super().run_simulation()

    def run_viewer(self):
        super().run_viewer()
