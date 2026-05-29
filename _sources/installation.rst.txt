Installation
============

DSNS is distributed on PyPI, and can be installed using `pip`:

.. code-block:: bash

    pip3 install dsns

However, many use cases will require modifications to the source code.
For this reason, we recommend installation from source, using the `--editable` flag to ensure the code can be updated without requiring reinstallation.
This can be done as follows:

.. code-block:: bash

    git clone https://github.com/ssloxford/dsns.git
    cd dsns
    pip3 install --editable .

Dependencies are listed in `pyproject.toml`, at the root of the repository.
We recommend the use of a virtual environment to reduce the likelihood of dependency conflicts.
