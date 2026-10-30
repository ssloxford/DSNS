# DSNS Documentation

This folder contains the documentation source files for DSNS.

## Requirements

Provided DSNS and its dependencies have been properly installed (see the main [README](README.md)), all requirements should be present.

The documentation tools are not part of the default install. Install them with the `docs` extra from the repository root:
```bash
pip install ".[docs]"
```
This installs `sphinx`, `sphinx-argparse` and `sphinx-rtd-theme`.

Also ensure the DSNS module is installed and on your `PYTHONPATH`.

## Build

To build the docs, simply run the following from the repository root:

```bash
sphinx-build docs docs/_build -b html
```

The docs will be built as HTML, with the index present at `docs/_build/html/index.html`.

