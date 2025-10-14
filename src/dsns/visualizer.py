from typing import Optional

import pyrender
import trimesh
import trimesh.creation # type: ignore
import trimesh.visual # type: ignore
import numpy as np
import time
import PIL.Image

from .constellation import Constellation, Satellites, OrbitalCenter, NullOrbitalCenter
from .multiconstellation import MultiConstellation
from .helpers import threaded, SatID


class Visualizer:
    """
    Base class for visualizers, providing basic functionality for rendering bodies, constellations, and links.
    """

    def __init__(
        self,
        time_scale: float = 1.0,
        space_scale: float = 1e-6,
        viewport_size: tuple[int, int] = (800, 600),
        interplanetary_scale: Optional[float] = None,
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
        ):
        """
        Initialize the visualizer.

        Parameters:
            time_scale: Visualization timescale (speed-up factor).
            space_scale: Visualization space scale (should be small, to reduce numbers to a manageable level).
            viewport_size: Size of the viewport (width, height).
            interplanetary_scale: Scale for interplanetary distances. If None, this will be the same as space_scale (potentially resulting in very large distances).
            bg_color: Background color.
        """

        self.time_scale = time_scale
        self.space_scale = space_scale
        self.viewport_size = viewport_size
        self.interplanetary_scale = interplanetary_scale or space_scale

        self.scene = pyrender.Scene(
            bg_color=bg_color,
        )
        self.viewer = None

    def build_planet_material(
        self,
        texture_path: str,
        normal_map_path: str | None,
        metallic_roughness_map_path: str | None,
        ) -> trimesh.visual.material.PBRMaterial:
        """
        Build a material for a planet.

        Parameters:
            texture_path: Path to the texture image.
            normal_map_path: Path to the normal map image.
            roughness_map_path: Path to the roughness map image.

        Returns:
            Material for the planet.
        """
        texture_image = PIL.Image.open(texture_path)
        if normal_map_path is not None:
            normal_map_image = PIL.Image.open(normal_map_path)
        else:
            normal_map_image = None
        if metallic_roughness_map_path is not None:
            metallic_roughness_image = PIL.Image.open(metallic_roughness_map_path)
        else:
            metallic_roughness_image = None

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_image,
            normalTexture=normal_map_image,
            metallicRoughnessTexture=metallic_roughness_image,
        )

        return material

    def generate_mesh_uvs(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Generate UVs for a spherical mesh.
        """
        vertices = mesh.vertices.copy()
        uv = np.zeros((mesh.vertices.shape[0], 2))
        for i, vertex in enumerate(vertices):
            vertex /= np.linalg.norm(vertex)
            x = vertex[0]
            y = vertex[1]
            z = vertex[2]
            uv[i, 0] = 0.5 + (np.arctan2(y, x) / (2 * np.pi))
            uv[i, 1] = 0.5 + (np.arcsin(z) / np.pi)

        return uv

    def build_nodes_mesh(self, node_mesh: trimesh.creation.Trimesh, positions: np.ndarray) -> pyrender.Mesh:
        """
        Build a mesh for a set of nodes.

        Parameters:
            node_mesh: Mesh for a single node.
            positions: Positions of the nodes (before space_scale is applied).

        Returns:
            Mesh for the nodes.
        """
        positions_scaled = positions * self.space_scale
        poses = np.tile(np.eye(4), (len(positions_scaled), 1, 1))
        poses[:,:3,3] = positions_scaled
        nodes_mesh = pyrender.Mesh.from_trimesh(node_mesh, poses=poses)

        return nodes_mesh

    def build_nodes_mesh_from_sats(self, node_mesh: trimesh.creation.Trimesh, satellites: Satellites) -> pyrender.Mesh:
        """
        Build a mesh for a set of nodes from a set of satellites.

        Parameters:
            node_mesh: Mesh for a single node.
            satellites: Satellites to build a mesh for.

        Returns:
            Mesh for the nodes.
        """
        positions = np.array([ (s.position * self.space_scale) + (s.orbital_center.position * self.interplanetary_scale) for s in satellites ])
        poses = np.tile(np.eye(4), (len(positions), 1, 1))
        poses[:,:3,3] = positions
        nodes_mesh = pyrender.Mesh.from_trimesh(node_mesh, poses=poses)

        return nodes_mesh

    def build_links_mesh(self, satellites: Satellites, links: list[tuple[SatID, SatID]], link_color: tuple[float, float, float]) -> Optional[pyrender.Mesh]:
        """
        Build a mesh for a set of links.

        Parameters:
            satellites: Satellites in the constellation.
            links: Links to build a mesh for.
            link_color: Color of the links.

        Returns:
            Mesh for the links, or None if there are no links.
        """
        isl_sats = [ (satellites.by_id(id_left), satellites.by_id(id_right)) for id_left, id_right in links ]
        isl_positions = [
            (
                (l.position * self.space_scale) + (l.orbital_center.position * self.interplanetary_scale),
                (r.position * self.space_scale) + (r.orbital_center.position * self.interplanetary_scale)
            )
            for l, r in isl_sats
            if l is not None and r is not None
        ]

        if len(isl_positions) == 0:
            return None

        positions_flat = np.concatenate(isl_positions)

        isl_primitive = pyrender.Primitive(positions=positions_flat, color_0=link_color, mode=1) # 1 = LINES
        isl_mesh = pyrender.Mesh(primitives=[isl_primitive])

        return isl_mesh

    def update_simulation(self, t: float):
        """
        Update the simulation state.

        Parameters:
            t: Time in seconds since the epoch.
        """
        pass

    @threaded
    def run_simulation(self):
        """
        Run the simulation in a separate thread.
        This should be called before run_viewer().
        """

        t = 0.0
        time_last = time.time()

        while True:
            if self.viewer is None:
                time.sleep(0.1)
                continue

            if not self.viewer.is_active:
                break

            self.viewer.render_lock.acquire()

            time_new = time.time()
            dt = time_new - time_last
            time_last = time_new

            t += dt * self.time_scale
            self.update_simulation(t)

            self.viewer.render_lock.release()

            time.sleep(0.01)

    def run_viewer(self):
        """
        Run the viewer.
        This should be called after run_simulation().
        """
        self.viewer = pyrender.Viewer(
            self.scene,
            viewport_size=self.viewport_size,
            use_raymond_lighting=True,
            run_in_thread=True,
            )


class ConstellationVisualizer(Visualizer):
    """
    Visualizer class for rendering a constellation and ISLs.
    """

    def __init__(
        self,
        constellation: Constellation,
        show_isls: bool = True,
        time_scale: float = 1.0,
        space_scale: float = 1e-6,
        host_radius: Optional[float] = None,
        host_rotation_period: Optional[float] = None,
        viewport_size: tuple[int, int] = (800, 600),
        sat_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        isl_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        host_materials: Optional[tuple[str, str, str]] = None,
        host_color: tuple[float, float, float] = (0.0, 0.0, 0.5),
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ):
        """
        Initialize the visualizer.

        Parameters:
            constellation: The constellation to visualize.
            show_isls: Whether to show ISLs.
            time_scale: Visualization timescale (speed-up factor).
            space_scale: Visualization space scale (should be small, to reduce numbers to a manageable level).
            host_radius: Radius of the host body (e.g. Earth). If None, no host body will be rendered.
            host_rotation_period: Rotation period of the host body (in seconds).
            viewport_size: Size of the viewport (width, height).
            sat_color: Color of the satellites.
            isl_color: Color of the ISLs.
            host_materials: Tuple of paths to the host body's diffuse, normal, and metallic-roughness maps.
            host_color: Color of the host body.
            bg_color: Background color.
        """

        super().__init__(time_scale, space_scale, viewport_size, bg_color=bg_color)

        self.constellation = constellation
        self.show_isls = show_isls
        self.host_radius = host_radius
        self.host_rotation_period = host_rotation_period

        self.sat_color = sat_color
        self.isl_color = isl_color
        self.host_color = host_color

        self.sat_mesh = trimesh.creation.icosphere(radius=0.1, subdivisions=1)
        self.sat_mesh.visual.vertex_colors = self.sat_color # type: ignore

        self.host_node = None
        if host_radius is not None:
            host_mesh = trimesh.creation.uv_sphere(radius=host_radius * space_scale)
            if host_materials is not None:
                material = self.build_planet_material(*host_materials)
                uv = self.generate_mesh_uvs(host_mesh)

                host_mesh.visual = trimesh.visual.TextureVisuals(material=material, uv=uv)
            else:
                host_mesh.visual.vertex_colors = self.host_color # type: ignore

            self.host_node = pyrender.Node(
                mesh=pyrender.Mesh.from_trimesh(host_mesh),
                matrix=np.eye(4)
            )
            self.scene.add_node(self.host_node)

        self.sats_node = None
        self.isls_node = None

    def update_simulation(self, t: float):
        self.constellation.update(t)

        if self.host_rotation_period is not None:
            # Set rotation quaternion
            if self.host_node is not None:
                angle = (2 * np.pi * t / self.host_rotation_period) + np.pi
                base_rotation = trimesh.transformations.quaternion_about_axis(np.pi, [0, 1, 0])
                angle_rotation = trimesh.transformations.quaternion_about_axis(angle, [1, 0, 0])
                self.host_node.rotation = trimesh.transformations.quaternion_multiply(base_rotation, angle_rotation)

        sats_mesh = self.build_nodes_mesh(self.sat_mesh, self.constellation.satellite_positions)
        if self.sats_node is not None:
            self.scene.remove_node(self.sats_node)
        self.sats_node = self.scene.add(sats_mesh)

        isl_mesh = self.build_links_mesh(self.constellation.satellites, self.constellation.isls, self.isl_color)
        if self.isls_node is not None:
            self.scene.remove_node(self.isls_node)
            self.isls_node = None
        if isl_mesh is not None:
            self.isls_node = self.scene.add(isl_mesh)


class MultiConstellationVisualizer(Visualizer):
    """
    Visualizer class for rendering multiple constellations and their ISLs/ILLs.
    """

    planets: list[tuple[
        pyrender.Node, # Node
        OrbitalCenter, # Center
        float,         # Radius
        float,         # Rotation period
    ]] = []

    def __init__(
        self,
        multiconstellation: MultiConstellation,
        show_links: bool = True,
        time_scale: float = 1.0,
        space_scale: float = 1e-6,
        interplanetary_scale: Optional[float] = None,
        viewport_size: tuple[int, int] = (800, 600),
        sat_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        isl_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        ill_color: tuple[float, float, float] = (1.0, 0.0, 1.0),
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ):
        """
        Initialize the visualizer.

        Parameters:
            constellation: The constellation to visualize.
            show_isls: Whether to show ISLs.
            time_scale: Visualization timescale (speed-up factor).
            space_scale: Visualization space scale (should be small, to reduce numbers to a manageable level).
            interplanetary_scale: Scale for interplanetary distances. If None, this will be the same as space_scale (potentially resulting in very large distances).
            viewport_size: Size of the viewport (width, height).
            sat_color: Color of the satellites.
            isl_color: Color of the ISLs.
            bg_color: Background color.
        """

        super().__init__(time_scale, space_scale, viewport_size, interplanetary_scale=interplanetary_scale, bg_color=bg_color)

        self.multiconstellation = multiconstellation
        self.show_links = show_links

        self.sat_color = sat_color
        self.isl_color = isl_color
        self.ill_color = ill_color

        self.sat_mesh = trimesh.creation.icosphere(radius=0.1, subdivisions=1)
        self.sat_mesh.visual.vertex_colors = self.sat_color # type: ignore

        self.sats_node = None
        self.isls_node = None
        self.ills_node = None

    def add_planet(
        self,
        radius: float,
        rotation_period: float,
        materials: Optional[tuple[str, str | None, str | None]] = None,
        color: tuple[float, float, float] = (0.0, 0.0, 0.5),
        center: OrbitalCenter = NullOrbitalCenter,
        ):
        """
        Add a planet to the visualization.

        Parameters:
            radius: Radius of the planet.
            rotation_period: Rotation period of the planet (in seconds).
            materials: Tuple of paths to the planet's diffuse, normal, and metallic-roughness maps.
            color: Color of the planet.
            center: Orbital center of the planet.
        """
        planet_mesh = trimesh.creation.uv_sphere(radius=radius * self.space_scale)
        if materials is not None:
            material = self.build_planet_material(*materials)
            uv = self.generate_mesh_uvs(planet_mesh)

            planet_mesh.visual = trimesh.visual.TextureVisuals(material=material, uv=uv)
        else:
            planet_mesh.visual.vertex_colors = color # type: ignore

        planet_node = pyrender.Node(
            mesh=pyrender.Mesh.from_trimesh(planet_mesh),
            matrix=np.eye(4)
        )
        planet_node.translation = center.position * self.interplanetary_scale
        self.scene.add_node(planet_node)

        self.planets.append((planet_node, center, radius, rotation_period))

    def update_simulation(self, t: float):
        self.multiconstellation.update(t)

        for (planet_node, planet_center, planet_radius, planet_rotation_period) in self.planets:
            # Update position
            planet_node.translation = planet_center.position * self.interplanetary_scale

            # Set rotation quaternion
            angle = (2 * np.pi * t / planet_rotation_period) + np.pi
            base_rotation = trimesh.transformations.quaternion_about_axis(np.pi, [0, 1, 0])
            angle_rotation = trimesh.transformations.quaternion_about_axis(angle, [1, 0, 0])
            planet_node.rotation = trimesh.transformations.quaternion_multiply(base_rotation, angle_rotation)

        sats_mesh = self.build_nodes_mesh_from_sats(self.sat_mesh, self.multiconstellation.satellites)
        if self.sats_node is not None:
            self.scene.remove_node(self.sats_node)
        self.sats_node = self.scene.add(sats_mesh)

        isl_mesh = self.build_links_mesh(self.multiconstellation.satellites, self.multiconstellation.isls, self.isl_color)
        if self.isls_node is not None:
            self.scene.remove_node(self.isls_node)
            self.isls_node = None
        if isl_mesh is not None:
            self.isls_node = self.scene.add(isl_mesh)

        ill_mesh = self.build_links_mesh(self.multiconstellation.satellites, self.multiconstellation.ills, self.ill_color)
        if self.ills_node is not None:
            self.scene.remove_node(self.ills_node)
            self.ills_node = None
        if ill_mesh is not None:
            self.ills_node = self.scene.add(ill_mesh)