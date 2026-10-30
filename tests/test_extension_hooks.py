import sys
import types
import warnings

import pytest

from dsns.events import Event
from dsns.message import DirectMessage, MessageCreatedEvent
import dsns.message
import dsns.message_actors
from dsns.logging import PreprocessedLoggingActor


class CustomCreatedEvent(Event):
    """Creation event that, like KeySpace's, subclasses plain Event."""

    def __init__(self, time, message):
        super().__init__(time)
        self.message = message


def _msg():
    return DirectMessage(source=0, destination=1, time=1.0)


def _no_key_management(monkeypatch):
    # A None entry in sys.modules makes the import raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "dsns.key_management", None)


def test_custom_event_type_is_logged(monkeypatch):
    _no_key_management(monkeypatch)
    actor = PreprocessedLoggingActor(created_event_types=(MessageCreatedEvent, CustomCreatedEvent))
    m = _msg()
    actor.handle_event(None, CustomCreatedEvent(1.0, m))
    assert m in actor.direct_messages
    assert actor.direct_messages[m].start_time == m.time


def test_default_behaviour_unchanged(monkeypatch):
    _no_key_management(monkeypatch)
    actor = PreprocessedLoggingActor()
    m = _msg()
    actor.handle_event(None, MessageCreatedEvent(1.0, m))
    assert m in actor.direct_messages
    m2 = DirectMessage(source=2, destination=3, time=2.0)
    actor.handle_event(None, CustomCreatedEvent(2.0, m2))
    assert m2 not in actor.direct_messages


def test_no_warning_without_key_management(monkeypatch):
    _no_key_management(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PreprocessedLoggingActor()


def test_deprecated_key_management_path(monkeypatch):
    fake = types.ModuleType("dsns.key_management")
    fake.UnsignedMessageCreatedEvent = CustomCreatedEvent
    monkeypatch.setitem(sys.modules, "dsns.key_management", fake)
    with pytest.warns(DeprecationWarning, match="created_event_types"):
        actor = PreprocessedLoggingActor()
    m = _msg()
    actor.handle_event(None, CustomCreatedEvent(1.0, m))
    assert m in actor.direct_messages


def test_no_warning_if_already_passed(monkeypatch):
    fake = types.ModuleType("dsns.key_management")
    fake.UnsignedMessageCreatedEvent = CustomCreatedEvent
    monkeypatch.setitem(sys.modules, "dsns.key_management", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PreprocessedLoggingActor(created_event_types=(MessageCreatedEvent, CustomCreatedEvent))


def test_attack_strategy_is_message_alias():
    assert dsns.message_actors.AttackStrategy is dsns.message.AttackStrategy
