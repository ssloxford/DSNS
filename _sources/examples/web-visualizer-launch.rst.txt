Web Visualizer Launcher
=======================

The `examples/web-visualizer-launch.py` script builds and serves the DSNS web visualizer locally.
It copies the static assets, bundles the Python source code into a zip file for Pyodide, and serves the result using a web server.

.. argparse::
    :filename: examples/web-visualizer-launch.py
    :func: get_parser
    :prog: web-visualizer-launch.py
