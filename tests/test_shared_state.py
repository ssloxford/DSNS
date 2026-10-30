"""
Tests that mutable state is not shared between instances of a class, or between simulations run
one after another in the same process.
"""

import pytest

from dsns.helpers import IDHelper
from dsns.logging import PreprocessedLoggingActor
from dsns.message import BroadcastMessage, DirectMessage, LTPConfig, MessageCreatedEvent
from dsns.message_actors import (
    BestEffortRoutingDataProvider,
    LookaheadRoutingDataProvider,
    MessageBroadcastActor,
    MessageRoutingActor,
)
from dsns.multiconstellation import MultiConstellation
from dsns.presets import small_iridium_constellation
from dsns.simulation import FixedEventsActor, LoggingActor, Simulation
from dsns.transmission import LinkTransmissionActor, MessageLocationTracker


# ---------------------------------------------------------------------------
# (a) Two instances of a class do not share containers
# ---------------------------------------------------------------------------

def _routing_actor():
    return MessageRoutingActor(BestEffortRoutingDataProvider())


def _broadcast_actor():
    return MessageBroadcastActor(BestEffortRoutingDataProvider())


INSTANCE_CASES = [
    (LoggingActor, ["events"]),
    (PreprocessedLoggingActor, ["direct_messages", "broadcast_messages", "other_events"]),
    (_routing_actor, ["_stored_messages", "_delivered_messages", "_delivered_message_uids"]),
    (_broadcast_actor, ["_stored_messages", "_delivered_messages"]),
    (lambda: LookaheadRoutingDataProvider(resolution=60.0, num_steps=2), ["_LookaheadRoutingDataProvider__graphs"]),
    (MessageLocationTracker, ["_messages_at_link"]),
    (LTPConfig, ["_underlying_message_uid_to_latest_ltp_session"]),
]


@pytest.mark.parametrize("factory, attributes", INSTANCE_CASES, ids=[
    "LoggingActor", "PreprocessedLoggingActor", "MessageRoutingActor", "MessageBroadcastActor",
    "LookaheadRoutingDataProvider", "MessageLocationTracker", "LTPConfig",
])
def test_instances_do_not_share_containers(factory, attributes):
    a = factory()
    b = factory()
    for attribute in attributes:
        container_a = getattr(a, attribute)
        container_b = getattr(b, attribute)
        assert container_a is not container_b, attribute
        assert len(container_a) == 0 and len(container_b) == 0, attribute


def test_logging_actor_events_are_per_instance():
    a = LoggingActor()
    b = LoggingActor()
    event = MessageCreatedEvent(0.0, DirectMessage(source=0, destination=1, data="x"))
    a.handle_event(None, event)
    assert a.get_events() == [event]
    assert b.get_events() == []


def test_message_location_tracker_is_per_instance():
    from dsns.message import Link

    a = MessageLocationTracker()
    b = MessageLocationTracker()
    message = DirectMessage(source=0, destination=1, data="x")
    link = Link(source=0, destination=1)
    a.add_message_to_location(message, link)
    assert a.is_message_at_location(message, link)
    assert not b.is_message_at_location(message, link)
    assert b.get_messages_in_location(link) == []


def test_ltp_config_sessions_are_per_instance():
    from dsns.message import HybridDirectMessage

    message = HybridDirectMessage(source=0, destination=1, data="x", size=10, reliable_data_size=10, unreliable_data_size=0)
    a = LTPConfig()
    b = LTPConfig()
    assert a.get_messages(0, 0, 1, message)[0].session == 0
    assert a.get_messages(0, 0, 1, message)[0].session == 1
    assert b.get_messages(0, 0, 1, message)[0].session == 0


def test_default_message_location_tracker_still_shared_between_actors():
    # Actors created without an explicit tracker keep sharing one tracker, as before.
    from dsns.transmission import LTPMessageRetransmissionActor, RetransmissionConfig

    from dsns.message import Link

    with pytest.warns(DeprecationWarning):
        link_actor = LinkTransmissionActor(default_bandwidth=1e6)
    with pytest.warns(DeprecationWarning):
        ltp_actor = LTPMessageRetransmissionActor(config=RetransmissionConfig())
    message = DirectMessage(source=0, destination=1, data="default-tracker")
    link = Link(source=0, destination=1)
    link_actor._message_location_tracker.add_message_to_location(message, link)
    try:
        assert ltp_actor._message_location_tracker.is_message_at_location(message, link)
    finally:
        link_actor._message_location_tracker.remove_message_from_location(message, link)


MODULES = [
    "dsns.constellation", "dsns.events", "dsns.helpers", "dsns.logging", "dsns.message",
    "dsns.message_actors", "dsns.multiconstellation", "dsns.presets", "dsns.reference_scenarios",
    "dsns.simulation", "dsns.traffic_sim", "dsns.transmission",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_no_mutable_class_level_containers(module_name):
    # Mutable containers assigned in a class body are shared by every instance of the class.
    import collections
    import enum
    import importlib
    import inspect

    module = importlib.import_module(module_name)
    mutable_types = (list, dict, set, collections.deque, bytearray)
    offenders = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module_name:
            continue
        if issubclass(cls, enum.Enum):
            continue
        for name, value in vars(cls).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in ("_fields", "_field_defaults"): # NamedTuple internals
                continue
            if isinstance(value, mutable_types):
                offenders.append(f"{cls.__name__}.{name}")
    assert offenders == []


def test_preset_multiconstellation_node_lists_do_not_grow():
    from dsns.presets import EarthMoonMarsMultiConstellation

    a = EarthMoonMarsMultiConstellation(moon=True, mars=False)
    b = EarthMoonMarsMultiConstellation(moon=True, mars=False)
    assert a.earth_nodes is not b.earth_nodes
    assert a.earth_nodes == b.earth_nodes
    assert a.moon_nodes == b.moon_nodes
    assert len(set(a.earth_nodes)) == len(a.earth_nodes)


# ---------------------------------------------------------------------------
# (b) Back-to-back simulations in one process give the same results as each run alone
# ---------------------------------------------------------------------------

def _scenario_events(kind: str) -> list:
    # Messages are created with their time set, so that the logs do not depend on hashing order.
    events = []
    if kind == "broadcast":
        for i in range(6):
            t = 20.0 * i
            events.append(MessageCreatedEvent(t, BroadcastMessage(source=i % 8, data=f"b-{i}", time=t, size=20000)))
    else:
        for i in range(20):
            t = 15.0 * i
            events.append(MessageCreatedEvent(t, DirectMessage(source=0, destination=5, data=f"a-{i}", time=t, size=50000)))
            events.append(MessageCreatedEvent(t + 3.0, DirectMessage(source=2, destination=7, data=f"c-{i}", time=t + 3.0, size=20000)))
    return events


def run_scenario(kind: str) -> dict:
    """
    Run a tiny simulation and return a uid-independent summary of its logs.

    Parameters:
        kind: "store_and_forward", "best_effort" or "broadcast".

    Returns:
        Summary of the logged events and the per-message data.
    """
    mobility = MultiConstellation()
    mobility.add_constellation(small_iridium_constellation(IDHelper()))

    if kind == "store_and_forward":
        routing = LookaheadRoutingDataProvider(resolution=60.0, num_steps=5)
        router = MessageRoutingActor(routing, store_and_forward=True, model_bandwidth=True)
    elif kind == "best_effort":
        routing = BestEffortRoutingDataProvider()
        router = MessageRoutingActor(routing, store_and_forward=False, model_bandwidth=True)
    elif kind == "broadcast":
        routing = BestEffortRoutingDataProvider()
        router = MessageBroadcastActor(routing, store_and_forward=False, model_bandwidth=True)
    else:
        raise ValueError(kind)

    actors = [
        LinkTransmissionActor(default_bandwidth=1e5, buffer_if_link_busy=True, message_location_tracker=MessageLocationTracker()),
        FixedEventsActor(_scenario_events(kind)),
        router,
    ]
    logging_actor = LoggingActor()
    preprocessed_logging_actor = PreprocessedLoggingActor()
    simulation = Simulation(
        mobility,
        actors=actors,
        logging_actors=[logging_actor, preprocessed_logging_actor],
        data_providers=[routing],
        timestep=10.0,
    )
    simulation.initialize(0.0)
    simulation.run(600.0)

    def describe(message):
        return (
            type(message).__name__,
            getattr(message, "source", None),
            getattr(message, "destination", None),
            getattr(message, "data", None),
            getattr(message, "hops", None),
        )

    events = [
        (type(event).__name__, round(event.time, 9), describe(getattr(event, "message", None)))
        for event in logging_actor.get_events()
    ]
    messages = sorted(
        (
            repr((
                describe(data.message), data.source, data.destination, round(data.start_time, 9),
                None if data.end_time is None else round(data.end_time, 9),
                data.hops, data.delivered, data.dropped, data.drop_reason,
            ))
            for data in list(preprocessed_logging_actor.direct_messages.values()) + list(preprocessed_logging_actor.broadcast_messages.values())
        )
    )
    return {"events": events, "messages": messages}


@pytest.mark.parametrize("kind", ["store_and_forward", "best_effort", "broadcast"])
def test_back_to_back_simulations_match(kind):
    first = run_scenario(kind)
    second = run_scenario(kind)

    assert len(first["events"]) > 0
    assert any(name == "MessageDeliveredEvent" or name == "MessageBroadcastDeliveredEvent" for name, _, _ in first["events"])
    assert len(second["events"]) == len(first["events"])
    assert len(second["messages"]) == len(first["messages"])
    assert second == first
