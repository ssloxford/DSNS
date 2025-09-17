# DSNS Documentation

This folder contains the documentation source files for DSNS.

## Requirements

Provided DSNS and its dependencies have been properly installed (see the main [README](README.md)), all requirements should be present.

If this is not the case, ensure the following Python packages have been installed:
```
sphinx
sphinx-argparse
sphinx-rtd-theme
```

Also ensure the DSNS module is installed and on your `PYTHONPATH`.

## Build

To build the docs, simply run the following from the repository root:

```bash
sphinx-build docs docs/_build -b html
```

The docs will be built as HTML, with the index present at `docs/_build/html/index.html`.

