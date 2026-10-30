import warnings

import pytest

from dsns.message import DirectMessage, Link
from dsns.transmission import (
    LinkTransmissionActor,
    LTPMessageRetransmissionActor,
    MessageLocationTracker,
    RetransmissionConfig,
)


def test_link_actor_without_tracker_warns():
    with pytest.warns(DeprecationWarning, match="message_location_tracker"):
        LinkTransmissionActor(default_bandwidth=1e6)


def test_ltp_actor_without_tracker_warns():
    with pytest.warns(DeprecationWarning, match="message_location_tracker"):
        LTPMessageRetransmissionActor(config=RetransmissionConfig())


def test_explicit_tracker_does_not_warn():
    tracker = MessageLocationTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        link_actor = LinkTransmissionActor(default_bandwidth=1e6, message_location_tracker=tracker)
        ltp_actor = LTPMessageRetransmissionActor(config=RetransmissionConfig(), message_location_tracker=tracker)
    assert link_actor._message_location_tracker is tracker
    assert ltp_actor._message_location_tracker is tracker


def test_default_tracker_still_shared_between_actors():
    # The deprecated fallback keeps working: both actors see the same tracker.
    with pytest.warns(DeprecationWarning):
        link_actor = LinkTransmissionActor(default_bandwidth=1e6)
    with pytest.warns(DeprecationWarning):
        ltp_actor = LTPMessageRetransmissionActor(config=RetransmissionConfig())
    assert link_actor._message_location_tracker is ltp_actor._message_location_tracker

    message = DirectMessage(source=0, destination=1, data="shared-default")
    link = Link(source=0, destination=1)
    link_actor._message_location_tracker.add_message_to_location(message, link)
    try:
        assert ltp_actor._message_location_tracker.is_message_at_location(message, link)
    finally:
        link_actor._message_location_tracker.remove_message_from_location(message, link)


def test_warning_points_at_caller():
    with pytest.warns(DeprecationWarning) as record:
        LinkTransmissionActor(default_bandwidth=1e6)
    assert record[0].filename == __file__
