from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .constellation import Satellites, Constellation
from .helpers import SatID, IDHelper, SPEED_OF_LIGHT


class ILLHelper(ABC):
    """
    Base class for ILL (Inter-Layer Link) helpers.
    """

    @abstractmethod
    def get_ills(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        """
        Get the ILLs for a multi-constellation.

        Parameters:
            satellites: Constellation to get ILLs for.
            positions: Positions of the satellites in the constellation.

        Returns:
            List of ILLs.
        """
        pass


class MultiConstellation:
    """
    Combines multiple constellations into one, with Inter-Layer Links (ILLs) between constellations.
    """

    constellations: list[Constellation] # Constellations in the multi-constellation
    satellites: Satellites # All satellites in the multi-constellation
    satellite_positions: np.ndarray # Positions of all satellites in the multi-constellation (in the same order as satellites)
    ill_helpers: list[ILLHelper] # Helpers for generating ILLs
    isls: list[tuple[SatID, SatID]] # ISLs in the multi-constellation
    ills: list[tuple[SatID, SatID]] # ILLs in the multi-constellation
    links: list[tuple[SatID, SatID]] # All links, including ILLs and ISLs
    time: float = 0.0 # Current time in seconds since the epoch

    speed_of_light: float = SPEED_OF_LIGHT # Speed of light in m/s

    def __init__(self):
        self.constellations = []
        self.satellites = Satellites([])
        self.satellite_positions = np.empty((0, 3))
        self.ill_helpers = []
        self.isls = []
        self.ills = []
        self.links = []

    def add_constellation(self, constellation: Constellation):
        """
        Add a constellation to the multi-constellation.

        Parameters:
            constellation: The constellation to add.
        """
        self.constellations.append(constellation)
        self.satellites.extend(constellation.satellites)

        self.satellite_positions = np.empty((len(self.satellites), 3))

    def add_ill_helper(self, ill_helper: ILLHelper):
        """
        Add an ILL helper to the multi-constellation.

        Parameters:
            ill_helper: The ILL helper to add.
        """
        self.ill_helpers.append(ill_helper)

    def update_positions(self, time: float):
        """
        Update the positions of all satellites in the multi-constellation at a given time.

        Parameters:
            time: Time in seconds since the epoch.
        """
        for constellation in self.constellations:
            constellation.update_positions(time)
            constellation.orbital_center.update(time)

        self.satellite_positions = np.concatenate([ constellation.satellite_positions for constellation in self.constellations ])

    def update_links(self, time: float):
        """
        Update the links between satellites in the multi-constellation at a given time.

        Parameters:
            time: Time in seconds since the epoch.
        """
        self.isls = []
        self.ills = []
        self.links = []

        for constellation in self.constellations:
            constellation.update_isls()
            self.isls.extend(constellation.isls)
            self.links.extend(constellation.isls)

        for ill_helper in self.ill_helpers:
            ills = ill_helper.get_ills(self.satellites, self.satellite_positions)
            self.ills.extend(ills)
            self.links.extend(ills)

    def update(self, time: float):
        """
        Update the multi-constellation at a given time.

        Parameters:
            time: Time in seconds since the epoch.
        """
        self.time = time

        self.update_positions(time)
        self.update_links(time)

    def get_distance(self, sat1: SatID, sat2: SatID) -> float:
        """
        Get the distance between two satellites.

        Parameters:
            sat1: First satellite.
            sat2: Second satellite.

        Returns:
            Distance between the two satellites, in metres.
        """
        if s1 := self.satellites.by_id(sat1):
            if s2 := self.satellites.by_id(sat2):
                s1_center = s1.orbital_center.position if s1.orbital_center is not None else np.zeros(3)
                s2_center = s2.orbital_center.position if s2.orbital_center is not None else np.zeros(3)
                # Subtract the big numbers separately to minimise floating point errors
                # Equivalent to (s2.position + s2_center) - (s1.position + s1_center)
                position_difference = (s2.position - s1.position) + (s2_center - s1_center)
                return np.linalg.norm(position_difference).astype(float)

        raise ValueError(f"Satellite {sat1} or {sat2} not found in multi-constellation")

    def get_delay(self, sat1: SatID, sat2: SatID) -> float:
        """
        Get the time delay between two satellites.

        Parameters:
            sat1: First satellite.
            sat2: Second satellite.

        Returns:
            Time delay between the two satellites, in seconds.
        """
        return self.get_distance(sat1, sat2) / self.speed_of_light

    def has_link(self, sat1: SatID, sat2: SatID) -> bool:
        """
        Check if two satellites have a link between them.

        Parameters:
            sat1: First satellite.
            sat2: Second satellite.

        Returns:
            True if there is a link between the two satellites, False otherwise.
        """
        return (sat1, sat2) in self.links or (sat2, sat1) in self.links


class FixedILLHelper(ILLHelper):
    """
    ILL helper class with fixed ILLs.
    """

    def __init__(self, ills: list[tuple[SatID, SatID]]):
        """
        Initialize the fixed ILL helper class.

        Parameters:
            ills: List of ILLs.
        """
        self.ills = ills

    @classmethod
    def from_names(cls, ills: list[tuple[str, str]], id_helper: IDHelper):
        """
        Initialize the fixed ILL helper class from a list of ILL names.

        Parameters:
            ills: List of ILL names.
        """
        ill_ids = [ (id_l, id_r) for ill_l, ill_r in ills if (id_l := id_helper.get_id(ill_l)) is not None and (id_r := id_helper.get_id(ill_r)) is not None ]
        return cls(ill_ids)

    def get_ills(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        return self.ills


class GroundILLHelper(ILLHelper):
    """
    ILL helper class for ground-to-space links.
    """

    def __init__(self, satellites_ground: list[SatID], satellites_space: list[SatID], min_elevation: Optional[float] = None, max_distance: Optional[float] = None):
        """
        Initialize the ground-to-space ILL helper class.

        Parameters:
            satellites_ground: List of ground station satellite IDs.
            satellites_space: List of satellites which can connect to the ground station.
            min_elevation: Minimum elevation (degrees) of the satellite above which a connection can be established.
            max_distance: Maximum distance (m) between the ground station and satellites.
        """

        self.satellites_ground = satellites_ground
        self.satellites_space = satellites_space
        self.index_ground: Optional[list[int]] = None
        self.index_space: Optional[list[int]] = None

        self.min_elevation = min_elevation
        self.min_elevation_cosine = None if min_elevation is None else np.cos(np.deg2rad(90.0 - min_elevation))
        self.max_distance = max_distance

    def get_ills(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        if self.index_ground is None:
            self.index_ground = [ satellites.index(s) for sat_id in self.satellites_ground if (s := satellites.by_id(sat_id)) is not None ]
        if self.index_space is None:
            self.index_space = [ satellites.index(s) for sat_id in self.satellites_space if (s := satellites.by_id(sat_id)) is not None ]

        ills = np.array([ [i, j] for i in self.index_ground for j in self.index_space if i != j ])
        if self.max_distance is not None:
            distances = np.linalg.norm(positions[ills[:, 0]] - positions[ills[:, 1]], axis=1)
            # Filter out the links where the distance is greater than the maximum distance
            ills = ills[distances <= self.max_distance]
        if self.min_elevation_cosine is not None:
            normals = positions / np.linalg.norm(positions, axis=1).reshape((-1, 1))
            normals_i = normals[ills[:, 0]]
            i_to_j = positions[ills[:, 1]] - positions[ills[:, 0]]
            i_to_j_norm = i_to_j / np.linalg.norm(i_to_j, axis=1).reshape((-1, 1))
            # Find the dot product between the normal vector and the vector from the ground station to the satellite
            dot_products = np.einsum('ij,ij->i', normals_i, i_to_j_norm)
            # Filter out the links where the dot product is less than the cosine of the minimum elevation
            ills = ills[dot_products >= self.min_elevation_cosine]

        ills_out: list[tuple[SatID, SatID]] = [ (satellites[i].sat_id, satellites[j].sat_id) for i, j in ills ]
        return ills_out


class DistanceILLHelper(ILLHelper):
    """
    ILL helper class for space-to-space links, limited only by distance.
    """

    def __init__(self, satellites_a: list[SatID], satellites_b: list[SatID], max_distance: float, max_links: Optional[int] = None):
        """
        Initialize the ground-to-space ILL helper class.

        Parameters:
            satellites_a: List of satellite IDs for the first constellation.
            satellites_b: List of satellite IDs for the second constellation.
            max_distance: Maximum distance (m) between the satellites.
            max_links: Maximum number of links per satellite.
        """

        self.satellites_a = satellites_a
        self.satellites_b = satellites_b
        self.index_a: Optional[list[int]] = None
        self.index_b: Optional[list[int]] = None

        self.max_distance = max_distance
        self.max_links = max_links

    def get_ills(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        if self.index_a is None:
            self.index_a = [ satellites.index(s) for sat_id in self.satellites_a if (s := satellites.by_id(sat_id)) is not None ]
        if self.index_b is None:
            self.index_b = [ satellites.index(s) for sat_id in self.satellites_b if (s := satellites.by_id(sat_id)) is not None ]

        positions_b_binned = np.nan_to_num(np.floor(positions[self.index_b] / self.max_distance)).astype(int)

        position_bins = dict()

        for i, index in enumerate(self.index_b):
            key = tuple(positions_b_binned[i])
            if key in position_bins:
                position_bins[key].append(index)
            else:
                position_bins[key] = [index]

        ills: list[tuple[SatID, SatID]] = []

        for i in self.index_a:
            links = 0

            position_bin = tuple(np.floor(positions[i] / self.max_distance).astype(int))
            neighbors = [ position_bins.get((x,y,z)) for x in range(position_bin[0]-1, position_bin[0]+2) for y in range(position_bin[1]-1, position_bin[1]+2) for z in range(position_bin[2]-1, position_bin[2]+2) ]
            neighbors = [ n for ns in neighbors if ns is not None for n in ns if n != i ]
            neighbors_dists = np.linalg.norm(positions[neighbors] - positions[i], axis=1)

            for n, dist in sorted(zip(neighbors, neighbors_dists), key=lambda x: x[1]):
                if dist > self.max_distance:
                    break
                if self.max_links is not None and links >= self.max_links:
                    break

                ills.append((satellites[i].sat_id, satellites[n].sat_id))
                links += 1

        return ills


class OcclusionILLHelper(ILLHelper):
    """
    ILL helper class for links which are occluded by one or more planets.
    We use the orbital center of each satellite as the center of the planet.
    """

    def __init__(self, satellites_a: list[SatID], satellites_b: list[SatID], radius_a: float, radius_b: float, min_elevation: Optional[float] = None, max_links: Optional[int] = None):
        """
        Initialize the occlusion ILL helper class.

        Parameters:
            satellites_a: List of satellite IDs for the first constellation.
            satellites_b: List of satellite IDs for the second constellation.
            radius_a: Radius of the planet for the first constellation.
            radius_b: Radius of the planet for the second constellation.
            min_elevation: Minimum elevation (degrees) of the satellites above which a connection can be established.
                           Only considers the angles from the perspective of satellites_a.
            max_links: Maximum number of links per satellite.
        """

        self.satellites_a = satellites_a
        self.satellites_b = satellites_b
        self.radius_a = radius_a
        self.radius_b = radius_b
        self.min_elevation = min_elevation
        self.max_links = max_links

        self.index_a: Optional[list[int]] = None
        self.index_b: Optional[list[int]] = None

    def __min_altitude(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Get the minimum altitude (distance from the origin) of a point on the line between two points.

        Parameters:
            a: First point.
            b: Second point.

        Returns:
            Minimum altitude.
        """
        v = a - b
        t = np.dot(a, v) / np.dot(v, v)
        t = np.clip(t, 0, 1)
        p = a + (t * (b - a))
        d = np.sqrt(np.sum(p**2))
        return d

    def __intersects_planets(self, position_a: np.ndarray, position_b: np.ndarray, center_a: np.ndarray, center_b: np.ndarray) -> bool:
        """
        Check if the line between two points intersects the planets.

        Parameters:
            position_a: First point.
            position_b: Second point.
            center_a: Center of the first planet.
            center_b: Center of the second planet.

        Returns:
            True if the line intersects the planets, False otherwise.
        """
        min_altitude_a = self.__min_altitude(position_a, position_b + (center_b - center_a))
        min_altitude_b = self.__min_altitude(position_b, position_a + (center_a - center_b))
        return min_altitude_a < self.radius_a or min_altitude_b < self.radius_b

    def get_ills(self, satellites: Satellites, positions: np.ndarray) -> list[tuple[SatID, SatID]]:
        if self.index_a is None:
            self.index_a = [ satellites.index(s) for sat_id in self.satellites_a if (s := satellites.by_id(sat_id)) is not None ]
        if self.index_b is None:
            self.index_b = [ satellites.index(s) for sat_id in self.satellites_b if (s := satellites.by_id(sat_id)) is not None ]

        ills: list[tuple[SatID, SatID]] = []

        center_a = np.zeros(3)
        if len(self.index_a) > 0 and (c := satellites[self.index_a[0]].orbital_center) is not None:
            center_a = c.position
        center_b = np.zeros(3)
        if len(self.index_b) > 0 and (c := satellites[self.index_b[0]].orbital_center) is not None:
            center_b = c.position

        # For each satellite in constellation A, check if the line between it and each satellite in constellation B intersects the planets
        for i in self.index_a:
            position_i = positions[i] + center_a
            normal = positions[i] / np.linalg.norm(positions[i])
            if np.linalg.norm(positions[i]) < self.radius_a:
                continue
            for j in self.index_b:
                position_j = positions[j] + center_b
                if np.linalg.norm(positions[j]) < self.radius_b:
                    continue
                if self.min_elevation is not None:
                    elevation = 90.0 - np.rad2deg(np.arccos(np.dot(normal, position_j - position_i) / np.linalg.norm(position_j - position_i)))
                    if elevation < self.min_elevation:
                        continue
                if not self.__intersects_planets(positions[i], positions[j], center_a, center_b):
                    ills.append((satellites[i].sat_id, satellites[j].sat_id))

        if self.max_links is not None:
            ills_old = ills
            ills = []
            # For each satellite in constellation A, sort the links by distance and keep only the closest max_links
            for i in self.index_a:
                links_i = [ (ii, j, np.linalg.norm((positions[ii] - positions[j]) + (center_a - center_b), axis=0)) for ii, j in ills_old if ii == satellites[i].sat_id ]
                links_i = sorted(links_i, key=lambda x: x[2])
                links_i = links_i[:self.max_links]
                ills.extend([ (ii, j) for ii, j, _ in links_i ])
            ills_old = ills
            ills = []
            # For each satellite in constellation B, sort the links by distance and keep only the closest max_links
            for j in self.index_b:
                links_j = [ (i, jj, np.linalg.norm((positions[i] - positions[jj]) + (center_a - center_b), axis=0)) for i, jj in ills_old if jj == satellites[j].sat_id ]
                links_j = sorted(links_j, key=lambda x: x[2])
                links_j = links_j[:self.max_links]
                ills.extend([ (i, jj) for i, jj, _ in links_j ])

        return ills