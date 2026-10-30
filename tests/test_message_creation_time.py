from dsns.message import BroadcastMessage, DirectMessage, MessageCreatedEvent

from test_smoke import MESSAGE_INTERVAL, SIM_TIME, _build


def test_created_event_sets_message_time():
    message = DirectMessage(source=0, destination=1, data="m")
    assert message.time == 0.0
    MessageCreatedEvent(42.5, message)
    assert message.time == 42.5


def test_created_event_overrides_explicit_time():
    # Routing actors always treated the event time as the creation time; the event now applies it up front.
    message = BroadcastMessage(source=0, data="b", time=3.0)
    MessageCreatedEvent(7.0, message)
    assert message.time == 7.0


def _run(store_and_forward: bool):
    sim, logger, _, _ = _build(store_and_forward)
    sim.initialize(time=0)
    sim.run(SIM_TIME, progress=False)
    return logger


def _check_log(logger):
    entries = list(logger.direct_messages.items())
    assert entries, "no messages logged"
    labels = [m.data for m, _ in entries]
    # One entry per created message: no stale duplicates left behind under an old hash.
    assert len(labels) == len(set(labels))
    for message, data in entries:
        created = int(message.data.rsplit("-", 1)[1]) * MESSAGE_INTERVAL
        assert data.start_time == created
        assert message.time == created
        # The entry can still be found under the message's current hash.
        assert logger.direct_messages[message] is data
    assert any(d.delivered for _, d in entries)


def test_logged_start_times_best_effort():
    _check_log(_run(store_and_forward=False))


def test_logged_start_times_store_and_forward():
    _check_log(_run(store_and_forward=True))
