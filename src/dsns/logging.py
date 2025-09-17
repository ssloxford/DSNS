from dataclasses import dataclass
from typing import NamedTuple

from dsns.multiconstellation import MultiConstellation
from dsns.transmission import Link, MessageReceptionCanceledEvent, MessageScheduledSendCompletedEvent

from .simulation import Actor
from .events import Event
from .message import BaseMessage, DirectMessage, BroadcastMessage, DropReason, LTPSegment, LinkMessageID, MessageCreatedEvent, MessageDeliveredEvent, MessageDroppedEvent, MessageBroadcastDeliveredEvent, MessageLostEvent, MessageQueuedEvent, MessageSentEvent, MessageReceivedEvent
from .helpers import SatID
try:
    from .key_management import UnsignedMessageCreatedEvent
except ModuleNotFoundError:
    UnsignedMessageCreatedEvent = MessageCreatedEvent

# Named tuple for message data
# Message, source, destination, start time, end time, hops
class MessageData(NamedTuple):
    message: BaseMessage
    source: SatID
    destination: SatID | None
    start_time: float
    end_time: float | None
    hops: int | None
    delivered: bool = False
    dropped: bool = False
    drop_reason: DropReason | None = None
    aborted: bool = False

class PreprocessedLoggingActor(Actor):
    """
    Actor that logs events, producing a preprocessed log aggregating message information.
    """

    direct_messages: dict[DirectMessage, MessageData] = {}
    broadcast_messages: dict[BroadcastMessage, MessageData] = {}

    log_other: bool = False
    other_events: list[Event] = []

    def __init__(self, log_other: bool = False):
        """
        Parameters:
            log_other: Whether to log other events (i.e. non-message events).
        """
        super().__init__()

        self.log_other = log_other

    def initialize(self) -> list[Event]:
        return []

    def create_message_data(self, message: BaseMessage) -> MessageData:
        message_data = MessageData(
            message = message,
            source = message.source,
            destination = message.destination if isinstance(message, DirectMessage) else None,
            start_time = message.time,
            end_time = None,
            hops = None,
        )
        return message_data

    def update_message(self, message: BaseMessage, **kwargs):
        if isinstance(message, DirectMessage):
            data = self.direct_messages.get(message)
        elif isinstance(message, BroadcastMessage):
            data = self.broadcast_messages.get(message)
        else:
            raise Exception("Message type not recognized.")

        if data is None:
            data = self.create_message_data(message)

        # prevent timeout exhaustions on control segments in LTP after message has been delivered from changing to an incorrect end time
        true_end_time = data.end_time if data.delivered else None

        data = data._replace(**kwargs)

        if true_end_time is not None:
            data = data._replace(end_time=true_end_time)

        if isinstance(message, DirectMessage):
            self.direct_messages[message] = data
        else:
            self.broadcast_messages[message] = data

    def handle_event(self, mobility, event):
        if isinstance(event, MessageCreatedEvent) or isinstance(event, UnsignedMessageCreatedEvent):
            message = event.message
            message_data = self.create_message_data(message)
            if isinstance(message, DirectMessage):
                self.direct_messages[message] = message_data
            elif isinstance(message, BroadcastMessage):
                self.broadcast_messages[message] = message_data
            else:
                raise Exception("Message type not recognized.")

        elif isinstance(event, MessageDeliveredEvent):
            self.update_message(
                event.message,
                end_time = event.time,
                hops = event.message.hops,
                delivered = True,
            )

        elif isinstance(event, MessageDroppedEvent):
            self.update_message(
                event.message,
                end_time = event.time,
                dropped = True,
                drop_reason = event.reason,
            )

        elif isinstance(event, MessageReceptionCanceledEvent):
            self.update_message(
                event.message,
                end_time = event.time,
                aborted = True,
            )

        elif isinstance(event, MessageBroadcastDeliveredEvent):
            self.update_message(
                event.message,
                end_time = event.time,
                hops = event.message.hops,
                delivered = True,
            )

        elif not (isinstance(event, MessageSentEvent) or isinstance(event, MessageReceivedEvent)):
            if self.log_other:
                self.other_events.append(event)

        return []

@dataclass(frozen=True, kw_only=True)
class Interval:
    start_time: float
    end_time: float

AverageQueueSize = float
TotalDataSent = float
Throughput = float
BandwidthUseRatio = float
BandwidthStatistics = dict[Link, dict[Interval, tuple[AverageQueueSize, TotalDataSent, Throughput, BandwidthUseRatio]]]

DataSent = int
QueueLength = int
class BandwidthLoggingActor(Actor):
    """
    Actor that logs bandwidth congestion over time for each link.
    """
    def __init__(self):
        super().__init__()
        # queue length history: link to list of (time, length)
        self.queue_lengths: dict[Link, list[tuple[Time, QueueLength]]] = {}

        # transmission intervals: Link to list of (transmission start, end, # data sent)
        self.transmissions: dict[Link, list[tuple[Time, Time, DataSent]]] = {}

        # internal queue counters
        self._counters: dict[Link, QueueLength] = {}

    def initialize(self) -> list[Event]:
        return []

    # note only if this sends do we want to record it
    last_schedule_end_event: MessageScheduledSendCompletedEvent = None

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        if isinstance(event, MessageQueuedEvent):
            link = Link(event.source, event.destination)
            current = self._counters.get(link, 0) + 1
            self._counters[link] = current
            self.queue_lengths.setdefault(link, []).append((event.time, current))

        elif isinstance(event, MessageSentEvent):
            link = Link(event.source, event.destination)

            # update queue counter
            current = self._counters.get(link, 1) - 1
            self._counters[link] = current
            self.queue_lengths.setdefault(link, []).append((event.time, current))

            # update bandwidth use
            if self.last_schedule_end_event and event.message.uid == self.last_schedule_end_event.message.uid and event.time == self.last_schedule_end_event.time:
                self.transmissions.setdefault(link, []).append((self.last_schedule_end_event.send_start_time, event.time, event.message.size))

        elif isinstance(event, MessageScheduledSendCompletedEvent):
            link = Link(event.source, event.destination)
            self.last_schedule_end_event = event

        return []

    def aggregate(self, period: float, default_bandwidth: float, bandwidth_per_link=None) -> BandwidthStatistics:
        """
        Returns: link to interval period to [average size of queue, total data sent
        (note something may span across more than one interval so use bandwidth),
        throughput, bandwidth used ratio]
        """
        def get_bandwidth(link):
            if bandwidth_per_link and link in bandwidth_per_link:
                return bandwidth_per_link[link]
            return default_bandwidth

        result: dict[Link, dict[float, tuple[Interval, float, float, float]]] = {}
        links = set(self.queue_lengths.keys()).union(self.transmissions.keys())
        for link in links:
            # compute average queue size per interval
            queue_data = self.queue_lengths.get(link, [])

            queue_per_interval = {}
            for time, queue_length in queue_data:
                interval_start = period * int(time // period)
                interval = Interval(start_time=interval_start, end_time = interval_start + period)
                queue_per_interval.setdefault(interval,[]).append(queue_length)

            average_queue_size = {interval: sum(lengths) / len(lengths) for interval, lengths in queue_per_interval.items()}

            # compute total data sent per interval
            data_sent_per_interval = {}
            for (start, end, size) in self.transmissions.get(link, []):

                # transmission may span multiple intervals
                start_interval_start = int(start // period)
                end_interval_start = int(end // period)
                duration = end - start
                if duration == 0: # ignore no-size messages
                    continue

                for i in range(start_interval_start, end_interval_start + 1):
                    interval_start = i * period
                    interval_end = interval_start + period
                    interval = Interval(start_time=interval_start, end_time=interval_end)

                    # Overlap of transmission with this interval
                    overlap_start = max(start, interval_start)
                    overlap_end = min(end, interval_end)
                    overlap = overlap_end - overlap_start
                    data_fraction = (overlap / duration) * size
                    data_sent_per_interval[interval] = data_sent_per_interval.get(interval, 0.0) + data_fraction

            # compute required statistics
            intervals = set(average_queue_size.keys()).union(data_sent_per_interval.keys())
            for interval in intervals:
                average_size = average_queue_size.get(interval, 0.0)
                total_data = data_sent_per_interval.get(interval, 0.0)
                throughput = total_data / period
                capacity = get_bandwidth(link)
                assert capacity > 0
                usage_ratio = throughput / capacity

                result.setdefault(link, {})
                result[link][interval] = (round(average_size, 3), round(total_data, 3), round(throughput, 3), round(usage_ratio, 10))

        return result

NumTransmissions = int
UnderlyingMessageTransmissionStatistics = dict[LinkMessageID, dict[Interval, NumTransmissions]]
LTPSegmentTransmissionStatistics = dict[LinkMessageID, dict[Interval, NumTransmissions]]

NumMessagesLost = int
UnderlyingMessageLossStatistics = dict[LinkMessageID, dict[Interval, NumMessagesLost]]
LTPSegmentLossStatistics = dict[LinkMessageID, dict[Interval, NumMessagesLost]]

Time = float
SegmentUID = int
UnderlyingMessageUID = int
class LTPTransmissionLoggingActor(Actor):
    """
    Actor that logs retransmission events over time. Used to compute the number of retransmissions per message.
    """
    def __init__(self):
        super().__init__()
        # transmissions: list of (time, link, segment_uid, underlying_message_uid)
        self.transmissions: list[tuple[Time, Link, SegmentUID, UnderlyingMessageUID]] = []
        self.lost_messages: list[tuple[Time, Link, SegmentUID, UnderlyingMessageUID]] = []

    def initialize(self) -> list[Event]:
        return []

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        if isinstance(event, MessageSentEvent):
            if isinstance(event.message, LTPSegment):
                link = Link(event.source, event.destination)
                self.transmissions.append((event.time, link, event.message.uid, event.message.underlying_message.uid))
        if isinstance(event, MessageLostEvent):
            if isinstance(event.message, LTPSegment):
                link = Link(event.source, event.destination)
                self.lost_messages.append((event.time, link, event.message.uid, event.message.underlying_message.uid))
        return []

    def aggregate(self, period: float) -> tuple[UnderlyingMessageTransmissionStatistics, LTPSegmentTransmissionStatistics, UnderlyingMessageLossStatistics, LTPSegmentLossStatistics]:
        """
        Returns two dictionaries.
        First is link and underying message UID to number of transmissions in each interval
        Second is link and LTP segment UID to number of transmissions in each interval
        """
        segment_to_transmission_statistics: LTPSegmentTransmissionStatistics = {}
        underying_message_to_transmission_statistics: UnderlyingMessageTransmissionStatistics = {}
        for (time, link, segment_uid, underlying_message_uid) in self.transmissions:
            start_time = period * int(time // period)
            interval = Interval(start_time=start_time, end_time=start_time+period)

            underlying_message_link_id = LinkMessageID(source=link.source, destination=link.destination, uid=underlying_message_uid)
            underying_message_to_transmission_statistics.setdefault(underlying_message_link_id, {})
            curr = underying_message_to_transmission_statistics.get(underlying_message_link_id).get(interval, 0)
            underying_message_to_transmission_statistics[underlying_message_link_id][interval] = curr + 1

            segment_link_id = LinkMessageID(source=link.source, destination=link.destination, uid=segment_uid)
            segment_to_transmission_statistics.setdefault(segment_link_id, {})
            curr = segment_to_transmission_statistics.get(segment_link_id).get(interval, 0)
            segment_to_transmission_statistics[segment_link_id][interval] = curr + 1

        segment_to_loss_statistics: LTPSegmentLossStatistics = {}
        underying_message_to_loss_statistics: UnderlyingMessageLossStatistics = {}
        for (time, link, segment_uid, underlying_message_uid) in self.lost_messages:
            start_time = period * int(time // period)
            interval = Interval(start_time=start_time, end_time=start_time+period)

            underlying_message_link_id = LinkMessageID(source=link.source, destination=link.destination, uid=underlying_message_uid)
            underying_message_to_loss_statistics.setdefault(underlying_message_link_id, {})
            curr = underying_message_to_loss_statistics.get(underlying_message_link_id).get(interval, 0)
            underying_message_to_loss_statistics[underlying_message_link_id][interval] = curr + 1

            segment_link_id = LinkMessageID(source=link.source, destination=link.destination, uid=segment_uid)
            segment_to_loss_statistics.setdefault(segment_link_id, {})
            curr = segment_to_loss_statistics.get(segment_link_id).get(interval, 0)
            segment_to_loss_statistics[segment_link_id][interval] = curr + 1

        return underying_message_to_transmission_statistics, segment_to_transmission_statistics, underying_message_to_loss_statistics, segment_to_loss_statistics

