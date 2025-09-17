from abc import ABC, abstractmethod
from typing import Optional, Self

import numpy as np
from PyAstronomy.pyasl import KeplerEllipse # type: ignore
import urllib.request # type: ignore
import sgp4.api, sgp4.conveniences
import datetime

from . import helpers
from .helpers import SatID, IDHelper


class OrbitalCenter:
    """
    Orbital center class.
    Used to represent a body around which satellites orbit.
    """

    def __init__(self, name: str, position: np.ndarray = np.zeros(3)):
        """
        Initialize the orbital center with a fixed position.

        Parameters:
            name: Name of the orbital center.
            position: Position of the orbital center (m).
        """

        self.name = name
        self.position: np.ndarray = position

    def update(self, time: float):
        """
        Update the orbital center.

        Parameters:
            time: Time in seconds since the epoch.
        """
        pass


NullOrbitalCenter = OrbitalCenter("null_orbital_center", np.zeros(3))


class PlanetOrbitalCenter(OrbitalCenter):
    """
    Orbital center class for a planet with a fixed rotation period.
    We assume the orbit is a perfect circle centered on the origin, in the x-y plane.
    """

    def __init__(self, name: str, center: np.ndarray | tuple[float, float, float] | OrbitalCenter, radius: float, rotation_period: float, angle_offset: float = 0.0):
        """
        Initialize the orbital center.

        Parameters:
            name: Name of the orbital center.
            center: Center of the orbit (m).
            radius: Radius of the orbit (m).
            rotation_period: Rotation period of the planet (s).
            angle_offset: Initial angle of the planet (degrees).
        """

        self.name = name
        self.__center = center
        self.__radius = radius
        self.__rotation_period = rotation_period
        self.__angle_offset = np.radians(angle_offset)

        self.position: np.ndarray = np.zeros(3)

    def update(self, time: float):
        """
        Update the orbital center.

        Parameters:
            time: Time in seconds since the epoch.
        """

        angle = 2 * np.pi * (time / self.__rotation_period) + self.__angle_offset
        if isinstance(self.__center, OrbitalCenter):
            self.__center.update(time)
            c = self.__center.position
        else:
            c = np.array(self.__center)
        self.position = c + np.array([ np.cos(angle), np.sin(angle), 0.0 ]) * self.__radius


class Satellite:
    """
    Satellite class.
    Used to represent a satellite in a constellation.
    """

    def __init__(self, name: str, sat_id: SatID, constellation_name: str, orbital_center: OrbitalCenter = NullOrbitalCenter):
        self.name = name
        self.sat_id = sat_id
        self.constellation_name = constellation_name
        self.orbital_center: OrbitalCenter = orbital_center

        self.position: np.ndarray = np.zeros(3)


class Satellites:
    """
    List of satellites, with helper functions for lookup by name/ID.
    """

    def __init__(self, satellites: list[Satellite]):
        self.satellites: list[Satellite] = satellites

        self.satellites_by_name = {}
        self.satellites_by_id = {}
        for satellite in satellites:
            self.satellites_by_name[satellite.name] = satellite
            self.satellites_by_id[satellite.sat_id] = satellite

    @property
    def ids(self) -> list[SatID]:
        """
        Get a list of all satellite IDs.
        """
        return [ sat.sat_id for sat in self.satellites ]

    @property
    def names(self) -> list[str]:
        """
        Get a list of all satellite names.
        """
        return [ sat.name for sat in self.satellites ]

    def __getitem__(self, index: int) -> Satellite:
        return self.satellites[index]

    def __len__(self) -> int:
        return len(self.satellites)

    def __iter__(self):
        return iter(self.satellites)

    def append(self, satellite: Satellite):
        """
        Add a satellite to the list.

        Parameters:
            satellite: Satellite to add.
        """
        self.satellites.append(satellite)
        self.satellites_by_name[satellite.name] = satellite
        self.satellites_by_id[satellite.sat_id] = satellite

    def extend(self, satellites: Self | list[Satellite]):
        """
        Add multiple satellites to the list.

        Parameters:
            satellites: List of satellites to add.
        """
        for satellite in satellites:
            self.satellites.append(satellite)
            self.satellites_by_name[satellite.name] = satellite
            self.satellites_by_id[satellite.sat_id] = satellite

    def index(self, satellite: Satellite) -> int:
        """
        Get the index of a satellite in the list.

        Parameters:
            satellite: Satellite to get the index of.

        Returns:
            Index of the satellite.
        """
        return self.satellites.index(satellite)

    def by_id(self, sat_id: SatID) -> Optional[Satellite]:
        """
        Get a satellite by its ID.

        Parameters:
            sat_id: ID of the satellite to get.

        Returns:
            Satellite with the given ID, or None if no satellite with that ID exists.
        """
        return self.satellites_by_id.get(sat_id, None)

    def by_name(self, name: str) -> Optional[Satellite]:
        """
        Get a satellite by its name.

        Parameters:
            name: Name of the satellite to get.

        Returns:
            Satellite with the given name, or None if no satellite with that name exists.
        """
        return self.satellites_by_name.get(name, None)


class ISLHelper(ABC):
    """
    Base class for ISL (Inter-Satellite Link) helper classes.
    """

    @abstractmethod
    def get_isls(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        """
        Compute the ISLs between all satellites in the constellation, given their positions.

        Parameters:
            satellites: List of satellites in the constellation.
            positions: Satellite positions (num_sats, 3).

        Returns:
            List of ISLs, where each ISL is a tuple of satellite IDs.
        """
        pass


class MultiISLHelper(ISLHelper):
    """
    Helper class allowing the use of multiple ISL helpers at the same time.
    """

    def __init__(self, isl_helpers: list[ISLHelper]):
        self.isl_helpers = isl_helpers

    def get_isls(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        isls = []

        for isl_helper in self.isl_helpers:
            isls += isl_helper.get_isls(satellites, positions)

        # Remove duplicates
        isls = list(set(isls))

        return isls


class Constellation(ABC):
    """
    Base satellite constellation class.
    """

    name: str
    satellites: Satellites

    isl_helper: ISLHelper
    satellite_positions: np.ndarray
    isls: list[tuple[SatID, SatID]]
    orbital_center: OrbitalCenter

    is_relay: bool = False

    @property
    def num_sats(self) -> int:
        return len(self.satellites)

    @abstractmethod
    def update_positions(self, time: float):
        """
        Update the positions of all satellites in the constellation at a given time.

        Parameters:
            time: Time in seconds since the epoch.
        """
        pass

    def update_isls(self):
        """
        Update the ISLs for the constellation.
        """
        self.isls = self.isl_helper.get_isls(self.satellites, self.satellite_positions)

    def update(self, time: float):
        """
        Update the constellation.
        """
        self.update_positions(time)
        self.orbital_center.update(time)
        self.update_isls()


class WalkerConstellation(Constellation):
    """
    Walker constellation class.
    """

    def __init__(
            self,
            name: str,
            num_planes: int,
            sats_per_plane: int,
            inclination: float,
            semi_major_axis: float,
            eccentricity: float,
            ascending_node_arc: float,
            phase_offset: float,
            isl_helper: ISLHelper,
            id_helper: IDHelper,
            host_mass: float = helpers.EARTH_MASS,
            orbital_center: OrbitalCenter = NullOrbitalCenter,
            right_ascending_node: float = 0.0,
        ):
        """
        Initialize the Walker constellation.

        Parameters:
            name: Name of the constellation.
            num_planes: Number of orbital planes.
            sats_per_plane: Number of satellites per orbital plane.
            inclination: Orbital plane inclination (degrees).
            semi_major_axis: Semi-major axis of the orbit (m).
            eccentricity: Orbital eccentricity (0.0 - 1.0).
            ascending_node_arc: Angle of arc along which to space ascending nodes (degrees).
                Setting to 360 results in a standard Walker constellation.
                180 results in a "Pi" constellation like Iridium.
            phase_offset: Phase offset between planes (degrees).
            isl_helper: ISL helper class.
            id_helper: ID helper class.
            host_mass: Mass of the central body (kg).
            orbital_center: Orbital center of the constellation.
            right_ascending_node: Longitude of the right ascending node in the first plane (degrees).
        """

        self.name = name

        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.inclination = inclination
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.ascending_node_arc = ascending_node_arc
        self.phase_offset = phase_offset

        self.period = helpers.orbital_period(semi_major_axis, host_mass)

        self.satellites = Satellites([])

        # Kepler ellipse solver for each plane
        self.plane_ellipses = []
        for i in range(num_planes):
            omega = (ascending_node_arc / num_planes) * i + right_ascending_node
            self.plane_ellipses.append(KeplerEllipse(
                per = self.period,
                a = self.semi_major_axis,
                e = self.eccentricity,
                i = self.inclination,
                Omega = omega,
                w = 0.0,
            ))

        for plane in range(num_planes):
            for sat in range(sats_per_plane):
                sat_name = f"{name}_{plane}_{sat}"
                sat_id = id_helper.assign_id(sat_name)
                self.satellites.append(Satellite(sat_name, sat_id, self.name, orbital_center=orbital_center))

        self.satellite_positions = np.zeros((self.num_sats, 3))
        self.update_positions(0.0)

        self.isl_helper = isl_helper

        self.orbital_center = orbital_center

    def update_positions(self, time):
        for plane in range(self.num_planes):
            for sat in range(self.sats_per_plane):
                sat_index = (plane * self.sats_per_plane) + sat
                phase_time_offset = (self.phase_offset / 360) * plane * self.period
                time_offset = ((self.period / self.sats_per_plane) * sat) + phase_time_offset
                pos = self.plane_ellipses[plane].xyzPos(time + time_offset)

                self.satellites[sat_index].position = pos
                self.satellite_positions[sat_index] = pos


class FixedConstellation(Constellation):
    """
    Constellation class for satellites with fixed positions.
    """

    def __init__(self, name: str, satellite_positions: np.ndarray, isl_helper: ISLHelper, id_helper: IDHelper, orbital_center: OrbitalCenter = NullOrbitalCenter):
        """
        Initialize the fixed constellation.

        Parameters:
            name: Name of the constellation.
            satellite_positions: Array of satellite positions (x/y/z (m)).
            isl_helper: ISL helper class.
            id_helper: ID helper class.
            orbital_center: Orbital center of the constellation.
        """
        self.name = name

        self.satellites = Satellites([])

        for i in range(len(satellite_positions)):
            sat_name = f"{name}_{i}"
            sat_id = id_helper.assign_id(sat_name)
            self.satellites.append(Satellite(sat_name, sat_id, self.name, orbital_center=orbital_center))

        self.satellite_positions = satellite_positions

        self.isl_helper = isl_helper

        self.orbital_center = orbital_center

    def update_positions(self, time):
        self.satellite_positions = self.satellite_positions
        for i in range(len(self.satellites)):
            self.satellites[i].position = self.satellite_positions[i]


class TLEConstellation(Constellation):
    """
    Constellation composed of satellites described by TLEs.
    The SGP4 orbital propagator is used to compute satellite positions.
    """

    def __init__(self, name: str, tles: list[tuple[str, str]], isl_helper: ISLHelper, id_helper: IDHelper, epoch: Optional[datetime.datetime] = None, ignore_errors: bool = False, orbital_center: OrbitalCenter = NullOrbitalCenter):
        """
        Initialize the TLE constellation.

        Parameters:
            name: Name of the constellation.
            tles: List of TLEs.
            isl_helper: ISL helper class.
            id_helper: ID helper class.
            epoch: Epoch of the TLEs - the real time corresponding to simulation time 0.0.
            ignore_errors: Ignore errors from the SGP4 propagator.
            orbital_center: Orbital center of the constellation.
        """

        self.name = name

        self.tles = tles
        self.sats = sgp4.api.SatrecArray([ sgp4.api.Satrec.twoline2rv(tle[0], tle[1]) for tle in self.tles ])
        self.epoch = epoch or datetime.datetime.now()
        self.ignore_errors = ignore_errors

        self.satellites = Satellites([])
        for i in range(len(self.tles)):
            sat_name = f"{name}_{i}"
            sat_id = id_helper.assign_id(sat_name)
            self.satellites.append(Satellite(sat_name, sat_id, self.name, orbital_center=orbital_center))

        self.satellite_positions = np.zeros((self.num_sats, 3))

        self.isl_helper = isl_helper

        self.orbital_center = orbital_center

    @classmethod
    def from_url(cls, name: str, tle_url: str, isl_helper: ISLHelper, id_helper: IDHelper, epoch: Optional[datetime.datetime] = None, ignore_errors: bool = False, orbital_center: OrbitalCenter = NullOrbitalCenter):
        """
        Initialize the TLE constellation from a URL (e.g. Celestrak).

        Parameters:
            name: Name of the constellation.
            tle_url: URL of the TLE file.
            isl_helper: ISL helper class.
            id_helper: ID helper class.
            epoch: Epoch of the TLEs - the real time corresponding to simulation time 0.0.
            ignore_errors: Ignore errors from the SGP4 propagator.
            orbital_center: Orbital center of the constellation.
        """

        tles = []
        with urllib.request.urlopen(tle_url) as url:
            data = url.read().decode()
            data = data.splitlines()

            for i in range(0, len(data), 3):
                line1 = data[i+1]
                line2 = data[i+2]
                tles.append([line1, line2])

        return cls(name, tles, isl_helper, id_helper, epoch, ignore_errors, orbital_center=orbital_center)

    @classmethod
    def from_file(cls, name: str, tle_file: str, isl_helper: ISLHelper, id_helper: IDHelper, epoch: Optional[datetime.datetime] = None, ignore_errors: bool = False, orbital_center: OrbitalCenter = NullOrbitalCenter):
        """
        Initialize the TLE constellation from a file.

        Parameters:
            name: Name of the constellation.
            tle_file: Path to the TLE file.
            isl_helper: ISL helper class.
            id_helper: ID helper class.
            epoch: Epoch of the TLEs - the real time corresponding to simulation time 0.0.
            ignore_errors: Ignore errors from the SGP4 propagator.
            orbital_center: Orbital center of the constellation.
        """
        tles = []
        with open(tle_file, 'r') as f:
            data = f.read().splitlines()

            for i in range(0, len(data), 3):
                line1 = data[i+1]
                line2 = data[i+2]
                tles.append([line1, line2])

        return cls(name, tles, isl_helper, id_helper, epoch, ignore_errors, orbital_center=orbital_center)

    def update_positions(self, time):
        jdays = [ sgp4.conveniences.jday_datetime(self.epoch + datetime.timedelta(seconds=time)) ]
        jd = np.array([ jd for jd, fr in jdays ])
        fr = np.array([ fr for jd, fr in jdays ])

        error, positions, velocities = self.sats.sgp4(jd, fr)
        if not self.ignore_errors and not (error == 0).all():
            error_sats = [ (i, e) for i, e in enumerate(error) if e != 0 ]
            raise RuntimeError(f"Error in satellite position calculation: {error_sats}")

        # Reshape, convert km to m
        positions = positions[:,0,:] * 1000

        assert(positions.shape[0] == self.num_sats)
        for sat_index in range(self.num_sats):
            self.satellites[sat_index].position = positions[sat_index]
            self.satellite_positions[sat_index] = positions[sat_index]


class GroundConstellation(Constellation):
    """
    Constellation class for ground stations.
    These are simulated as satellites with a fixed position.
    """

    def __init__(self, name: str, ground_station_positions: np.ndarray, host_radius: float, rotation_period: Optional[float], isl_helper: ISLHelper, id_helper: IDHelper, orbital_center: OrbitalCenter = NullOrbitalCenter):
        """
        Initialize the ground constellation.

        Parameters:
            name: Name of the constellation.
            ground_station_positions: Array of ground station positions (lat/lon/alt (deg, deg, m)).
            host_radius: Radius of the central body (m).
            rotation_period: Rotation period of the body on which the ground stations are orbiting (s).
            isl_helper: ISL helper class.
            id_helper: ID helper class.
            orbital_center: Orbital center of the constellation.
        """
        self.name: str = name

        self.satellites = Satellites([])

        for i in range(len(ground_station_positions)):
            sat_name = f"{name}_{i}"
            sat_id = id_helper.assign_id(sat_name)
            self.satellites.append(Satellite(sat_name, sat_id, self.name, orbital_center=orbital_center))

        self.ground_station_positions = ground_station_positions
        self.host_radius = host_radius
        self.rotation_period = rotation_period

        self.satellite_positions = np.zeros((self.num_sats, 3))
        self.update_positions(0.0)

        self.isl_helper = isl_helper

        self.orbital_center = orbital_center

    def update_positions(self, time):
        lat_lon_alt = self.ground_station_positions.copy()
        if self.rotation_period is not None:
            rotation_angle = 360 * (time / self.rotation_period)
            lat_lon_alt[:, 1] += rotation_angle

        self.satellite_positions = helpers.lat_lon_alt_to_xyz(lat_lon_alt, self.host_radius)

        for i in range(len(self.satellites)):
            self.satellites[i].position = self.satellite_positions[i]


class WalkerISLHelper(ISLHelper):
    """
    Walker ISL helper class.
    """

    def __init__(self, num_planes: int, sats_per_plane: int, intra_layer_links: bool, inter_layer_links: Optional[int], disable_cross_seam_links: bool = False):
        """
        Initialize the Walker ISL helper class.

        Parameters:
            num_planes: Number of orbital planes.
            sats_per_plane: Number of satellites per orbital plane.
            intra_layer_links: Include intra-layer links (links between adjacent satellites on the same layer).
            inter_layer_links: Include inter-layer links (links between adjacent satellites on adjacent layers).
                If None, no inter-layer links are included.
                Otherwise, specify the offset between layers.
            disable_cross_seam_links: Disable inter-layer links between satellites on opposite sides of the seam.
                Only relevant for polar constellations.
        """
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.intra_layer_links = intra_layer_links
        self.inter_layer_links = inter_layer_links
        self.disable_cross_seam_links = disable_cross_seam_links

        self.isls: Optional[list[tuple[SatID, SatID]]] = None

    def get_isls(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        if self.isls is not None:
            return self.isls

        self.isls = []
        for i in range(self.num_planes):
            for j in range(self.sats_per_plane):
                sat_index = (i * self.sats_per_plane) + j
                sat = satellites[sat_index]
                if self.intra_layer_links:
                    next_sat_index = (i * self.sats_per_plane) + ((j + 1) % self.sats_per_plane)
                    next_sat = satellites[next_sat_index]
                    self.isls.append((sat.sat_id, next_sat.sat_id))
                if self.inter_layer_links is not None:
                    if self.disable_cross_seam_links and i == self.num_planes - 1:
                        continue
                    next_sat_index = (((i + 1) % self.num_planes) * self.sats_per_plane) + (j + self.inter_layer_links) % self.sats_per_plane
                    next_sat = satellites[next_sat_index]
                    self.isls.append((sat.sat_id, next_sat.sat_id))

        return self.isls


class FixedISLHelper(ISLHelper):
    """
    ISL helper class with fixed ISLs.
    """

    def __init__(self, isls: list[tuple[SatID, SatID]]):
        """
        Initialize the fixed ISL helper class.

        Parameters:
            isls: List of ISLs.
        """
        self.isls = isls

    @classmethod
    def from_names(cls, isls: list[tuple[str, str]], id_helper: IDHelper):
        """
        Initialize the fixed ISL helper class from a list of ISL names.

        Parameters:
            isls: List of ISL names.
        """
        isl_ids = [ (id_l, id_r) for isl_l, isl_r in isls if (id_l := id_helper.get_id(isl_l)) is not None and (id_r := id_helper.get_id(isl_r)) is not None ]
        return cls(isl_ids)

    def get_isls(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        return self.isls


class NullISLHelper(ISLHelper):
    """
    Null ISL helper class - no ISLs are returned.
    """

    def __init__(self):
        """
        Initialize the Null ISL helper class.
        """
        pass

    def get_isls(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        return []


class AdHocISLHelper(ISLHelper):
    """
    Ad-Hoc ISL helper class.
    Satellites are connected to the nearest satellite(s) within a specified range.
    """

    def __init__(self, max_range: float, max_links: Optional[int], min_altitude: Optional[float]):
        """
        Initialize the Ad-Hoc ISL helper class.

        Parameters:
            max_range: Maximum range for ISLs (m).
            max_links: Maximum number of links per satellite.
                If None, no maximum is enforced.
            min_altitude: Minimum altitude for line-of-sight between satellites (m).
                If None, no line-of-sight checks are performed.
        """

        self.max_range = max_range
        self.max_links = max_links
        self.min_altitude = min_altitude

    def __min_altitude(self, sat1: Satellite, sat2: Satellite) -> float:
        """
        Calculate the minimum altitude for a line-of-sight between two satellites.

        Parameters:
            sat1: First satellite.
            sat2: Second satellite.

        Returns:
            Minimum altitude for a line-of-sight between the two satellites (m).
        """
        a = sat1.position
        b = sat2.position
        v = a - b
        t = np.dot(a, v) / np.dot(v, v)
        t = np.clip(t, 0, 1)
        p = a + (t * (b - a))
        d = np.sqrt(np.sum(p**2))
        return d

    def get_isls(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        positions_binned = np.nan_to_num(np.floor(positions / self.max_range)).astype(int)

        position_bins = dict()

        for i in range(positions_binned.shape[0]):
            key = tuple(positions_binned[i])
            if key in position_bins:
                position_bins[key].append(i)
            else:
                position_bins[key] = [i]

        isls = []

        for i in range(len(satellites)):
            sat = satellites[i]
            pos = positions[i]

            links = 0

            position_bin = tuple(np.floor(pos / self.max_range).astype(int))
            neighbors = [ position_bins.get((x,y,z)) for x in range(position_bin[0]-1, position_bin[0]+2) for y in range(position_bin[1]-1, position_bin[1]+2) for z in range(position_bin[2]-1, position_bin[2]+2) ]
            neighbors = [ n for ns in neighbors if ns is not None for n in ns if n != i ]
            neighbors_dists = np.linalg.norm(positions[neighbors] - pos, axis=1)

            for n, dist in sorted(zip(neighbors, neighbors_dists), key=lambda x: x[1]):
                if dist > self.max_range:
                    break
                if self.max_links is not None and links >= self.max_links:
                    break

                if self.min_altitude is None or self.__min_altitude(sat, satellites[n]) > self.min_altitude:
                    isls.append((sat.sat_id, satellites[n].sat_id))
                    links += 1

        return isls