import os
import sys

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import dsns  # noqa: E402


def pytest_report_header(config):
    return f"dsns imported from: {os.path.dirname(dsns.__file__)}"


@pytest.fixture(scope="session", autouse=True)
def _check_dsns_source():
    # Make sure tests run against this checkout, not some other installed copy.
    assert os.path.dirname(dsns.__file__).startswith(SRC), dsns.__file__
