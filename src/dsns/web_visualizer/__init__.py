"""
DSNS Web Visualizer — browser-based 3D satellite network visualizer.

Runs the full Python simulation inside Pyodide (WebAssembly) and renders
via Three.js in the browser.
"""

from .runner import WebRunner, PRESETS

__all__ = ["WebRunner", "PRESETS"]
