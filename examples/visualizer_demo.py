"""Constellation visualizer

This example demonstrates the use of the visualizer to display a variety of constellations and multiconstellations.
"""

from dsns.presets import (
    GroundMultiConstellation,
    GPSMultiConstellation,
    IridiumMultiConstellation,
    StarlinkMultiConstellation,
    StarlinkTLEMultiConstellation,
    CubesatMultiConstellation,
    MultiLayerMultiConstellation,
    EarthMoonMarsMultiConstellation,
    LeoLeoMultiConstellation,
    LeoMeoMultiConstellation,
    LeoGeoMultiConstellation,
    LeoMeoGeoMultiConstellation,
    EarthVisualizer,
    EarthMarsVisualizer,
    EarthMoonVisualizer,
)
import argparse

presets_earth = [
    "ground",
    "gps",
    "iridium",
    "starlink",
    "starlink-tle",
    "cubesat-tle",
    "multi-layer",
    "leo-leo",
    "leo-meo",
    "leo-geo",
    "leo-meo-geo",
]
presets_earth_mars = [
    "earth-mars",
]
presets_earth_moon = [
    "earth-moon",
]
presets = presets_earth + presets_earth_mars + presets_earth_moon

def get_parser():
    parser = argparse.ArgumentParser(description="Demo script for the constellation visualizer, with a range of configurable presets.")
    parser.add_argument("--preset", help="The preset to use (earth-mars, starlink, etc.)", choices=presets, default="earth-mars")
    parser.add_argument("--time-scale", help="The time scale to use (s/s)", type=float, default=100.0)
    parser.add_argument("--space-scale", help="The space scale to use (m/m)", type=float, default=1e-6)
    parser.add_argument("--interplanetary-scale", help="The interplanetary scale to use (m/m)", type=float, default=1e-10)
    parser.add_argument("--viewport-size", help="The viewport size to use (px, px)", type=int, nargs=2, default=(800, 600))
    parser.add_argument("-w", "--white-bg", help="Use a white background instead of the default black", action="store_true")

    return parser

if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    earth_visualizer_args = dict(
        time_scale=args.time_scale,
        space_scale=args.space_scale,
        earth_materials=(
            "assets/2k_earth_daymap.jpg",
            "assets/2k_earth_normal_map.jpg",
            "assets/2k_earth_metallic_roughness_map.jpg"
        ),
        viewport_size=args.viewport_size,
        bg_color=(1.0, 1.0, 1.0) if args.white_bg else (0.0, 0.0, 0.0),
    )

    earth_mars_visualizer_args = dict(
        interplanetary_scale=args.interplanetary_scale,
        mars_material="assets/2k_mars.jpg",
    )
    earth_mars_visualizer_args.update(earth_visualizer_args)

    earth_moon_visualizer_args = dict(
        interplanetary_scale=args.interplanetary_scale,
        moon_material="assets/2k_moon.jpg",
    )
    earth_moon_visualizer_args.update(earth_visualizer_args)

    if args.preset == "ground":
        multi_constellation = GroundMultiConstellation()
    elif args.preset == "gps":
        multi_constellation = GPSMultiConstellation()
    elif args.preset == "iridium":
        multi_constellation = IridiumMultiConstellation()
    elif args.preset == "starlink":
        multi_constellation = StarlinkMultiConstellation()
    elif args.preset == "starlink-tle":
        multi_constellation = StarlinkTLEMultiConstellation()
    elif args.preset == "cubesat-tle":
        multi_constellation = CubesatMultiConstellation(file_cubesats='assets/cubesats.txt')
    elif args.preset == "multi-layer":
        multi_constellation = MultiLayerMultiConstellation()
    elif args.preset == "leo-leo":
        multi_constellation = LeoLeoMultiConstellation()
    elif args.preset == "leo-meo":
        multi_constellation = LeoMeoMultiConstellation()
    elif args.preset == "leo-geo":
        multi_constellation = LeoGeoMultiConstellation()
    elif args.preset == "leo-meo-geo":
        multi_constellation = LeoMeoGeoMultiConstellation()
    elif args.preset == "earth-mars":
        multi_constellation = EarthMoonMarsMultiConstellation(moon=False, mars=True)
    elif args.preset == "earth-moon":
        multi_constellation = EarthMoonMarsMultiConstellation(moon=True, mars=False)
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    if args.preset in presets_earth:
        visualizer = EarthVisualizer(
            multi_constellation,
            **earth_visualizer_args
        )
    elif args.preset in presets_earth_mars:
        visualizer = EarthMarsVisualizer(
            multi_constellation, # type: ignore
            **earth_mars_visualizer_args
        )
    elif args.preset in presets_earth_moon:
        visualizer = EarthMoonVisualizer(
            multi_constellation, # type: ignore
            **earth_moon_visualizer_args
        )
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    visualizer.run_simulation()
    visualizer.run_viewer()