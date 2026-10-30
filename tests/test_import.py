import importlib

import pytest

MODULES = [
    "dsns.constellation", "dsns.events", "dsns.helpers", "dsns.logging", "dsns.message",
    "dsns.message_actors", "dsns.multiconstellation", "dsns.presets", "dsns.reference_scenarios",
    "dsns.simulation", "dsns.traffic_sim", "dsns.transmission",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
