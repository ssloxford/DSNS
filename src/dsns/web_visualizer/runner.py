"""
Web visualizer runner, executes inside Pyodide.
Builds a simulation preset and exposes a step(t) function that returns
serializable state for the Three.js renderer.

Performance design:

- All bulk data is returned as flat numpy typed arrays (float32 / int32).
  Pyodide shares these with JS as zero-copy Float32Array / Int32Array views.
- The return dict and all buffers are pre-allocated once and reused.
- No per-step Python list allocations for positions, links, or planet data.
"""

import numpy as np
import math

from dsns.presets import (
    GroundMultiConstellation,
    GPSMultiConstellation,
    IridiumMultiConstellation,
    StarlinkMultiConstellation,
    MultiLayerMultiConstellation,
    LeoLeoMultiConstellation,
    LeoMeoMultiConstellation,
    LeoGeoMultiConstellation,
    LeoMeoGeoMultiConstellation,
    EarthMoonMarsMultiConstellation,
    CubesatMultiConstellation,
    WalkerConstellation,
)
from dsns.constellation import NullISLHelper, WalkerISLHelper
from dsns.multiconstellation import MultiConstellation
from dsns.helpers import IDHelper, EARTH_RADIUS, EARTH_ROTATION_PERIOD, EARTH_MASS
from dsns.helpers import MARS_RADIUS, MARS_ROTATION_PERIOD, MARS_MASS
from dsns.helpers import MOON_RADIUS, MOON_ROTATION_PERIOD, MOON_MASS


PRESETS = [
    "ground",
    "gps",
    "iridium",
    "starlink",
    "cubesat-tle",
    "multi-layer",
    "leo-leo",
    "leo-meo",
    "leo-geo",
    "leo-meo-geo",
    "earth-moon",
    "earth-mars",
    "walker",
]


def _planet_rotation(t, rotation_period):
    """Return quaternion for planet rotation about the Z axis.

    Format is [x, y, z, w] for direct use with THREE.Quaternion.
    """
    angle = (2.0 * math.pi * t / rotation_period) * -1.0
    half = angle / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


class WebRunner:
    """
    Runner that builds a preset simulation and exposes step(t).
    """

    def __init__(
        self,
        preset: str = "walker",
        time_scale: float = 100.0,
        space_scale: float = 1e-6,
        interplanetary_scale: float = 1e-10,
        sat_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        isl_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        ill_color: tuple[float, float, float] = (1.0, 0.0, 1.0),
    ):
        self.preset = preset
        self.time_scale = time_scale
        self.space_scale = space_scale
        self.interplanetary_scale = interplanetary_scale or space_scale
        self.sat_color = list(sat_color)
        self.isl_color = list(isl_color)
        self.ill_color = list(ill_color)

        self.multi_constellation: MultiConstellation
        self.planets: list[dict] = []

        self._build_preset()

        # Initialize at t=0 so we can measure actual link counts
        self.multi_constellation.update(0.0)

        # Build a SatID -> array index map so link IDs align with position arrays.
        # This is required because constellations may be added in a different order
        # from their ID assignment order (e.g. cubesat-tle).
        self._id_to_index = {
            sat.sat_id: i
            for i, sat in enumerate(self.multi_constellation.satellites)
        }

        # --- Pre-allocate flat typed arrays ---
        max_sats = len(self.multi_constellation.satellites)
        max_isls = max(64, int(len(self.multi_constellation.isls) * 1.1))
        max_ills = max(64, int(len(self.multi_constellation.ills) * 1.1))

        self._max_sats = max_sats
        self._max_isls = max_isls
        self._max_ills = max_ills

        self._positions_buf = np.empty(max_sats * 3, dtype=np.float32)
        self._isls_buf = np.empty(max_isls * 2, dtype=np.int32)
        self._ills_buf = np.empty(max_ills * 2, dtype=np.int32)
        self._zero_vec = np.zeros(3, dtype=np.float32)

        # Pre-allocate planet position/rotation buffers
        self._planet_pos = np.empty(len(self.planets) * 3, dtype=np.float32)
        self._planet_rot = np.empty(len(self.planets) * 4, dtype=np.float32)
        self._num_planets = len(self.planets)

        # Pre-allocate return dict (updated in-place each step)
        self._state = {
            "satellite_positions": self._positions_buf,
            "num_sats": max_sats,
            "isls": self._isls_buf,
            "num_isls": 0,
            "ills": self._ills_buf,
            "num_ills": 0,
            "planets": [],
            "num_planets": self._num_planets,
            "sim_time": 0.0,
            "sat_color": self.sat_color,
            "isl_color": self.isl_color,
            "ill_color": self.ill_color,
        }

    def _build_preset(self):
        preset = self.preset

        if preset == "walker":
            id_helper = IDHelper()
            walker_isl = WalkerISLHelper(
                num_planes=72,
                sats_per_plane=22,
                intra_layer_links=True,
                inter_layer_links=2,
            )
            constellation = WalkerConstellation(
                name="walker",
                num_planes=72,
                sats_per_plane=22,
                inclination=53.0,
                semi_major_axis=550e3 + EARTH_RADIUS,
                eccentricity=0.0,
                ascending_node_arc=360.0,
                phase_offset=(65 / 72) * 360.0,
                isl_helper=walker_isl,
                id_helper=id_helper,
            )
            mc = MultiConstellation()
            mc.add_constellation(constellation)
            self.multi_constellation = mc
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "ground":
            self.multi_constellation = GroundMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "gps":
            self.multi_constellation = GPSMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "iridium":
            self.multi_constellation = IridiumMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "starlink":
            self.multi_constellation = StarlinkMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "cubesat-tle":
            import os
            tle_path = os.path.join(os.path.dirname(__file__), "..", "data", "cubesats.txt")
            self.multi_constellation = CubesatMultiConstellation(file_cubesats=tle_path)
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "multi-layer":
            self.multi_constellation = MultiLayerMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "leo-leo":
            self.multi_constellation = LeoLeoMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "leo-meo":
            self.multi_constellation = LeoMeoMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "leo-geo":
            self.multi_constellation = LeoGeoMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "leo-meo-geo":
            self.multi_constellation = LeoMeoGeoMultiConstellation()
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
            })
        elif preset == "earth-moon":
            mc = EarthMoonMarsMultiConstellation(moon=True, mars=False)
            self.multi_constellation = mc
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
                "orbital_center": "earth_orbital_center",
            })
            self.planets.append({
                "name": "moon",
                "radius": MOON_RADIUS * self.space_scale,
                "rotation_period": MOON_ROTATION_PERIOD,
                "color": [184/255, 184/255, 184/255],
                "texture": "assets/1k_moon.jpg",
                "orbital_center": "moon_orbital_center",
            })
        elif preset == "earth-mars":
            mc = EarthMoonMarsMultiConstellation(moon=False, mars=True)
            self.multi_constellation = mc
            self.planets.append({
                "name": "earth",
                "radius": EARTH_RADIUS * self.space_scale,
                "rotation_period": EARTH_ROTATION_PERIOD,
                "color": [107/255, 147/255, 214/255],
                "texture": "assets/1k_earth_daymap.jpg",
                "orbital_center": "earth_orbital_center",
            })
            self.planets.append({
                "name": "mars",
                "radius": MARS_RADIUS * self.space_scale,
                "rotation_period": MARS_ROTATION_PERIOD,
                "color": [193/255, 68/255, 14/255],
                "texture": "assets/1k_mars.jpg",
                "orbital_center": "mars_orbital_center",
            })
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def step(self, t: float) -> dict:
        """
        Advance the simulation to time *t* and return serializable state.
        All bulk data is written into pre-allocated flat buffers.
        """
        self.multi_constellation.update(t)
        self._state["sim_time"] = t

        # --- Satellite positions ---
        ss = self.space_scale
        ips = self.interplanetary_scale
        pos_buf = self._positions_buf
        for i, sat in enumerate(self.multi_constellation.satellites):
            p = sat.position
            if sat.orbital_center is not None:
                c = sat.orbital_center.position
                pos_buf[i * 3] = float(p[0] * ss + c[0] * ips)
                pos_buf[i * 3 + 1] = float(-p[1] * ss + c[1] * ips)
                pos_buf[i * 3 + 2] = float(-(p[2] * ss + c[2] * ips))
            else:
                pos_buf[i * 3] = float(p[0] * ss)
                pos_buf[i * 3 + 1] = float(-p[1] * ss)
                pos_buf[i * 3 + 2] = float(-p[2] * ss)

        # --- ISLs ---
        isl_buf = self._isls_buf
        num_isls = min(len(self.multi_constellation.isls), self._max_isls)
        for i in range(num_isls):
            id1, id2 = self.multi_constellation.isls[i]
            isl_buf[i * 2] = self._id_to_index[int(id1)]
            isl_buf[i * 2 + 1] = self._id_to_index[int(id2)]
        self._state["num_isls"] = num_isls

        # --- ILLs ---
        ill_buf = self._ills_buf
        num_ills = min(len(self.multi_constellation.ills), self._max_ills)
        for i in range(num_ills):
            id1, id2 = self.multi_constellation.ills[i]
            ill_buf[i * 2] = self._id_to_index[int(id1)]
            ill_buf[i * 2 + 1] = self._id_to_index[int(id2)]
        self._state["num_ills"] = num_ills

        # --- Planets ---
        planets_out = self._state["planets"]
        planets_out.clear()
        pp = self._planet_pos
        pr = self._planet_rot

        for j, planet in enumerate(self.planets):
            oc_name = planet.get("orbital_center")
            if oc_name:
                oc = getattr(self.multi_constellation, oc_name, None)
                if oc is not None:
                    oc.update(t)
                    c = oc.position
                    pp[j * 3] = float(c[0] * ips)
                    pp[j * 3 + 1] = float(c[1] * ips)
                    pp[j * 3 + 2] = float(-c[2] * ips)
                else:
                    pp[j * 3] = 0.0
                    pp[j * 3 + 1] = 0.0
                    pp[j * 3 + 2] = 0.0
            else:
                pp[j * 3] = 0.0
                pp[j * 3 + 1] = 0.0
                pp[j * 3 + 2] = 0.0

            rot = _planet_rotation(t, planet["rotation_period"])
            pr[j * 4] = rot[0]
            pr[j * 4 + 1] = rot[1]
            pr[j * 4 + 2] = rot[2]
            pr[j * 4 + 3] = rot[3]

            planets_out.append({
                "name": planet["name"],
                "position": [float(pp[j * 3]), float(pp[j * 3 + 1]), float(pp[j * 3 + 2])],
                "rotation": [float(pr[j * 4]), float(pr[j * 4 + 1]), float(pr[j * 4 + 2]), float(pr[j * 4 + 3])],
                "radius": float(planet["radius"]),
                "color": [float(c) for c in planet["color"]],
                "texture": planet.get("texture"),
            })

        return self._state
