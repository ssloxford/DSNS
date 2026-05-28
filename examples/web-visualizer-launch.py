#!/usr/bin/env python3
"""
Web Visualizer Launcher

Builds a self-contained static site for the DSNS web visualizer and serves it
via a local HTTP server.  Open the displayed URL in a browser to run the
visualizer entirely client-side (Pyodide + Three.js).

Usage:
    python web-visualizer-launch.py --preset walker --port 8000
"""

import argparse
import http.server
import os
import shutil
import socketserver
import sys
import tempfile
import zipfile

def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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


def copytree(src, dst):
    if os.path.isdir(src):
        if not os.path.isdir(dst):
            os.makedirs(dst)
        for name in os.listdir(src):
            copytree(os.path.join(src, name), os.path.join(dst, name))
    else:
        shutil.copy2(src, dst)


def build_site(preset: str, time_scale: float, space_scale: float, interplanetary_scale: float, step_interval: float) -> str:
    """Create a temporary directory containing the complete static site."""
    root = _repo_root()
    src_dir = os.path.join(root, "src", "dsns")
    static_dir = os.path.join(src_dir, "web_visualizer", "static")
    assets_dir = os.path.join(root, "examples", "assets")
    build_dir = tempfile.mkdtemp(prefix="dsns_web_vis_")

    # 1. Copy static web files
    copytree(static_dir, build_dir)

    # 2. Copy only the web-visualizer texture assets (1k variants)
    dst_assets = os.path.join(build_dir, "assets")
    os.makedirs(dst_assets, exist_ok=True)
    for texture in ("1k_earth_daymap.jpg", "1k_moon.jpg", "1k_mars.jpg"):
        src = os.path.join(assets_dir, texture)
        if os.path.exists(src):
            shutil.copy2(src, dst_assets)

    # 3. Package dsns source (and any bundled data files) into dsns.zip
    zip_path = os.path.join(build_dir, "dsns.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_dir, _dirs, files in os.walk(src_dir):
            if "__pycache__" in root_dir:
                continue
            for fname in files:
                if fname.endswith((".pyc", ".pyo")) or fname.startswith("."):
                    continue
                full = os.path.join(root_dir, fname)
                arc = os.path.join("dsns", os.path.relpath(full, src_dir))
                zf.write(full, arc)

    # 4. Write a convenience index redirect that defaults to the chosen preset
    index_path = os.path.join(build_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            html = f.read()
        # Inject default preset into the query string by replacing the param parsing line
        # We leave the existing param parsing in main.js so manual overrides still work.
        # Just write a small note, the param parsing in main.js already reads URL params.
        pass

    return build_dir


def serve(directory: str, port: int, preset: str, time_scale: float, space_scale: float, interplanetary_scale: float, step_interval: float):
    os.chdir(directory)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\nServing DSNS web visualizer from {directory}")
        print(f"Open http://localhost:{port}/?preset={preset}&timeScale={time_scale}&spaceScale={space_scale}&interplanetaryScale={interplanetary_scale}&stepInterval={step_interval}")
        print("Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.shutdown()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Launch the DSNS web visualizer as a locally-served static site."
    )
    parser.add_argument(
        "--preset",
        help="Simulation preset to visualise",
        choices=PRESETS,
        default="walker",
    )
    parser.add_argument(
        "--time-scale",
        help="Simulation time scale factor (sim seconds per real second)",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--space-scale",
        help="Spatial scale factor (m -> visual units)",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--interplanetary-scale",
        help="Interplanetary distance scale factor",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--step-interval",
        help="Simulation step interval in ms (default 333 = ~3 steps/sec)",
        type=float,
        default=333.0,
    )
    parser.add_argument(
        "--port",
        help="HTTP server port",
        type=int,
        default=8000,
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    build_dir = build_site(
        preset=args.preset,
        time_scale=args.time_scale,
        space_scale=args.space_scale,
        interplanetary_scale=args.interplanetary_scale,
        step_interval=args.step_interval,
    )
    serve(build_dir, args.port, args.preset, args.time_scale, args.space_scale, args.interplanetary_scale, args.step_interval)
