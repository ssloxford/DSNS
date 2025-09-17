from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .simulation import Actor
from .events import Event, InstantEvent, LinkDownEvent, LinkUpEvent
from .message import (BaseMessage, DirectMessage, DropReason, 
                      LTPCheckpointData, LTPDataSegment, LTPDataType, 
                      LTPSegment, LTPReportAcknowledgementSegment, 
                      LTPReportSegment, LTPSegmentCreatedEvent, LTPSegmentReceivedEvent, Link, 
                      LinkMessageID, MessageDroppedEvent, 
                      MessageQueuedEvent, MessageReceivedEvent, MessageRerouteEvent, MessageSentEvent)
from .multiconstellation import MultiConstellation
from .helpers import SatID

"""
=======
Message Tracking Functionality
Used for Rerouting Messages
Unfortunately, this must be shared between
Link Actor and Reliable Transfer Actor.
Without LTP: Location changes when we send a message
With LTP: Location changes when:
    - DirectMessage:
        - All Green: on send
        - Red part: got report that everything received
    - not DirectMessage:
        - on send
=======
"""
class MessageLocationTracker:
    _messages_at_link: dict[Link, dict[int, DirectMessage]] = {}

    def __init__(self):
        pass 

    def remove_message_from_location(self, message: DirectMessage, link: Link):
        if link in self._messages_at_link:
            self._messages_at_link[link].pop(message.uid, None)
            if not self._messages_at_link[link]:
                self._messages_at_link.pop(link)

    def add_message_to_location(self, message: DirectMessage, link: Link):
        assert not isinstance(message, LTPSegment)
        if link not in self._messages_at_link or message.uid not in self._messages_at_link.get(link, {}):
            self._messages_at_link.setdefault(link, {})[message.uid] = message

    def get_messages_in_location(self, link: Link):
        uid_to_message = self._messages_at_link.get(link, {})
        return [message for uid, message in uid_to_message.items()]
    
    def is_message_at_location(self, message: DirectMessage, link: Link):
        return message.uid in self._messages_at_link.get(link, {})

"""
==================================
Bandwidth Management Functionality
==================================
"""

class MessageScheduledSendCompletedEvent(Event):
    """
    Event for when a message was scheduled to 
    have been fully transmitted onto the link.
    """
   
    message: BaseMessage

    def __init__(self, time: float, send_start_time: float, source: SatID, destination: SatID, message: BaseMessage):
        super().__init__(time, ["send_start_time", "source", "destination", "message"])
        self.send_start_time = send_start_time
        self.source = source
        self.destination = destination
        self.message = message

class BandwidthUpdateEvent(Event):
    """
    Event to update the bandwidth of a link.
    """  
    def __init__(self, time: float, source: SatID, destination: SatID, bandwidth: float):
        super().__init__(time, [ "source", "destination", "bandwidth"])
        self.source = source
        self.destination = destination
        self.bandwidth = bandwidth

@dataclass
class TransmissionInterval:
    start_time: float
    end_time: float 

class LinkTransmissionActor(Actor):
    """
    Models bandwidth to schedule when messages are sent
    This actor schedules one message at a time
    """
    def __init__(self, 
                 default_bandwidth: float, 
                 per_link_bandwidth: dict | None = None, 
                 buffer_if_link_busy: bool = False, 
                 reroute_on_link_down: bool = True, 
                 message_location_tracker: MessageLocationTracker = MessageLocationTracker()):
        
        super().__init__()
        self._default_bandwidth: float = default_bandwidth
        self._per_link_bandwidth: dict | None = per_link_bandwidth or {}
        self._buffer_if_link_busy: bool = buffer_if_link_busy
        self._message_location_tracker = message_location_tracker

        self._link_next_available_time: dict[Link, float] = {}
        # messages being transmitted into link
        self._link_queued_messages: dict[Link, list[BaseMessage]] = {}

        # messages buffered waiting for link to go up
        self._link_buffered_messages: dict[Link, list[BaseMessage]] = {}

        self._scheduled_message_transmit_interval: dict[LinkMessageID, TransmissionInterval] = {}

        self._reroute_on_link_down: bool = reroute_on_link_down

    def initialize(self) -> list[Event]:
        """
        Initialize the actor.

        Returns:
            List of events to add to the event queue.
        """
        return []
    
    def __transmit_next_message(self, time: float, source: SatID, destination: SatID) -> list[Event]:
        """
        We first compute when the link is next free. If it's free now get the next message
        for the queue for that link and schedule it for transmission. 
        """
        link = Link(source, destination)
        if self._link_queued_messages.get(link, []):
            bandwidth = self._per_link_bandwidth.get(link, self._default_bandwidth)
            link_available_time = self._link_next_available_time.get(link, time)
            transmit_start_time = max(time, link_available_time)

            # edge case: when a message is queued on a link exactly when the message being transmitted on that link is scheduled
            # to finish being transmitted. But that message is not yet 'sent' - and so it's transmission period is recomputed.
            if transmit_start_time == time:
                # link is free now
                next_message = self._link_queued_messages.get(link)[0]
                transmit_end_time = transmit_start_time + next_message.size / bandwidth
                self._link_next_available_time[link] = transmit_end_time

                link_message_id = LinkMessageID(source, destination, next_message.uid)
                self._scheduled_message_transmit_interval[link_message_id] = TransmissionInterval(transmit_start_time, transmit_end_time)
                return [MessageScheduledSendCompletedEvent(transmit_end_time, transmit_start_time, source, destination, next_message)] 
        return []

    def __queue_message_and_transmit_if_link_free(self, event_time: float, message: BaseMessage, source: SatID, destination: SatID) -> list[Event]:
        """
        Add message to the queue if it can be sent immediately, 
        otherwise only if buffering is enabled and if not we drop it.
        Then transmit next message if link is free (nothing in queue).
        Otherwise we wait for the message to be sent to trigger transmission
        of next message!
        """
        events = []
        link = Link(source, destination)
        if self._link_queued_messages.get(link, []):
            # queue is non-empty
            if self._buffer_if_link_busy:
                self._link_queued_messages.get(link).append(message)
            else: 
                events.append( MessageDroppedEvent(event_time, source, message, DropReason.INSUFFICIENT_BUFFER) )
        else:
            # queue is empty  
            self._link_queued_messages.setdefault(link, []).append(message) 
            # only transmit message if queue is empty
            events.extend( self.__transmit_next_message(event_time, source, destination) )

        return events                     

    def handle_message_queued_event(self, mobility: MultiConstellation, event: MessageQueuedEvent, message: BaseMessage) -> list[Event]:
        assert message.size > 0, f"Event Queue ordering may work incorrectly when messages have size {message.size}. If we model bandwidth ensure size is positive."

        link = Link(event.source, event.destination)
        # TODO: message can be received and this gets cleaned up, but if LTP we can resend a checkpoint and add it back here (it is then uncleaned - how to fix)
        if isinstance(message, DirectMessage):
            # Control segments don't count as message in that link still (as we don't want to reroute after message)
            if not isinstance(message, LTPReportSegment) and not isinstance(message, LTPReportAcknowledgementSegment):
                underlying_message = message.underlying_message if isinstance(message, LTPSegment) else message
                self._message_location_tracker.add_message_to_location(underlying_message, link)

        if mobility.has_link(event.source, event.destination):
            if self._link_buffered_messages.get(link, []):
                # something was buffered. So new message comes in interval [link up on mobility, LinkUpEvent)
                # we want to buffer it to maintain order and transmit everything on LinkUpEvent
                assert not self._link_queued_messages.get(link, [])
                self._link_buffered_messages[link].append(message)
                return []
            else:
                return self.__queue_message_and_transmit_if_link_free(event.time, message, event.source, event.destination)
        else:
            if self._buffer_if_link_busy:
                self._link_buffered_messages.setdefault(link, []).append(message)
                return []
            else:
                if isinstance(message, DirectMessage):
                    # Control segments don't count as message in that link still (as we don't want to reroute after message)
                    if not isinstance(message, LTPReportSegment) and not isinstance(message, LTPReportAcknowledgementSegment):
                        underlying_message = message.underlying_message if isinstance(message, LTPSegment) else message
                        self._message_location_tracker.remove_message_from_location(underlying_message, link)

                return [ MessageDroppedEvent(event.time, event.source, message, DropReason.INSUFFICIENT_BUFFER) ]
    
    def handle_message_scheduled_send_completed_event(self, mobility: MultiConstellation, event: MessageScheduledSendCompletedEvent, message: BaseMessage) -> list[Event]:
        """
        Check if this scheduled send completion is stale. If not raise the corresponding MessageSentEvent
        to signal that the entire message was successfully transmitted onto the link.
        """
        if mobility.has_link(event.source, event.destination):
            # handles cases where message queued and scheduled, link goes down and back up, then old scheduled send completed comes through
            link_message_id = LinkMessageID(event.source, event.destination, message.uid)
            if link_message_id in self._scheduled_message_transmit_interval:
                scheduled_transmission_interval = self._scheduled_message_transmit_interval[link_message_id]
                if scheduled_transmission_interval == TransmissionInterval(event.send_start_time, event.time):
                    return [ MessageSentEvent(event.time, event.source, event.destination, event.message) ] 
            print(f"Message is: {message}")
            print(f"Message id in transmit schedules: {link_message_id in self._scheduled_message_transmit_interval}")
            assert False, f"This is a sanity check. In theory can get here, but it is very rare and unlikely with lookahead routing. Make sure its a valid case. Event is: {event}. Scheduled Message Transmit Intervals are: {self._scheduled_message_transmit_interval}, Is link if buffered: {Link(link_message_id.source, link_message_id.destination) in self._link_buffered_messages}, Is message in buffered: {Link(link_message_id.source, link_message_id.destination) in self._link_buffered_messages and message in self._link_buffered_messages[Link(link_message_id.source, link_message_id.destination)]}"

        """
        If we get here then either:
        (1) link is down during transmission of message onto link but message is queued so will be added to buffer on LinkDownEvent
        (2) scheduled message does not have an interval. E.g. if somehow bandwidth changed, link went down and back up and the rescheduled transmission completed first
        (3) scheduled message did not have latest scheduled transmission interval. Happens when link went down and back up before this event, and the message was rescheduled
        """
        return []

    def handle_link_down_event(self, mobility: MultiConstellation, event: LinkDownEvent) -> list[Event]:
        """
        Assume for now that if buffering is enabled
        a message is buffered (not sent) until it 
        can and is successfully sent
        TODO: expand this later for specific behavior depending on what goes down
        """
        events = []
        links = [Link(event.sat1, event.sat2), Link(event.sat2, event.sat1)]
        for link in links:
            messages = self._link_queued_messages.pop(link, []) + self._link_buffered_messages.pop(link, [])
            if self._reroute_on_link_down:
                # reroute the directmessages, keep rest buffered
                messages_to_reroute = self._message_location_tracker.get_messages_in_location(link)
                
                messages_to_buffer = []
                for message in messages:
                    if not isinstance(message, DirectMessage):
                        messages_to_buffer.append(message)
                if messages_to_buffer:
                    self._link_buffered_messages[link] = messages_to_buffer

                for message in messages_to_reroute:
                    # drop latest transmission and clear sent stuff
                    events.append(MessageRerouteEvent(time=event.time, source=link.source, previous_destination=link.destination, message=message))
                    self._message_location_tracker.remove_message_from_location(message, link)
            else:
                if messages:
                    self._link_buffered_messages[link] = messages
            self._link_next_available_time.pop(link, None)    

        return events 

    def handle_message_sent_event(self, mobility: MultiConstellation, event: MessageSentEvent, message: BaseMessage) -> list[Event]:
        """
        MessageSentEvent signifies the entire message was transmitted onto the link successfully.
        If there is a next message waiting to be sent on that link, schedule it.
        """
        events = []
        link = Link(event.source, event.destination)
        if not isinstance(message, LTPSegment):
            self._message_location_tracker.remove_message_from_location(message, link)

        if link in self._link_queued_messages and message in self._link_queued_messages[link]:
            index = self._link_queued_messages[link].index(message)

            assert index == 0, f"Sent message out of order. Message is {message} at index {index} of queue {self._link_queued_messages[link]}"

            # must do this here. Cannot do this after transmitting next message in case its the same message as was sent now.
            link_message_id = LinkMessageID(event.source, event.destination, message.uid)
            if link_message_id in self._scheduled_message_transmit_interval:
                del self._scheduled_message_transmit_interval[link_message_id]

            del self._link_queued_messages[link][index]
            if not self._link_queued_messages[link]:
                del self._link_queued_messages[link]
            else:
                # schedule next message
                events.extend(self.__transmit_next_message(event.time, event.source, event.destination))
        else:
            raise Exception(f"Got a message Sent that wasn't Queued. It was likely buffered and this is a resolution issue. Handle this explicitly.")

        return events
    
    def handle_link_up_event(self, mobility: MultiConstellation, event: LinkUpEvent) -> list[Event]:
        """
        Recomputes scheduled send completion events for buffered messages
        Note these must be scheduled before any new store and forward messages are queued and scheduled for
        """
        source, destination = event.sat1, event.sat2 
        events = []

        link = Link(source, destination)
        messages = self._link_buffered_messages.pop(link, [])
        for message in messages:
            events.extend(self.__queue_message_and_transmit_if_link_free(event.time, message, source, destination))

        link = Link(destination, source)
        messages = self._link_buffered_messages.pop(link, [])
        for message in messages:
            events.extend(self.__queue_message_and_transmit_if_link_free(event.time, message, destination, source))

        return events
    
    def handle_bandwidth_update_event(self, mobility: MultiConstellation, event: BandwidthUpdateEvent):
        link = link(event.source, event.destination)
        self._per_link_bandwidth[link] = event.bandwidth
        return []

    def handle_event(self, mobility: MultiConstellation, event) -> list[Event]:
        if isinstance(event, MessageQueuedEvent):
            return self.handle_message_queued_event(mobility, event, event.message)
        elif isinstance(event, MessageScheduledSendCompletedEvent):
            return self.handle_message_scheduled_send_completed_event(mobility, event, event.message)
        elif isinstance(event, LinkUpEvent):
            return self.handle_link_up_event(mobility, event)
        elif isinstance(event, LinkDownEvent):
            return self.handle_link_down_event(mobility, event)
        elif isinstance(event, MessageSentEvent):
            return self.handle_message_sent_event(mobility, event, event.message)
        elif isinstance(event, BandwidthUpdateEvent):
            return self.handle_bandwidth_update_event(mobility, event)
        return []
    







"""
====================================
Message Retransmission Functionality
====================================
"""

class MessageTransmissionTimeoutEvent(Event):
    
    message: int
    source: SatID
    destination: SatID

    def __init__(self, time: float, message: BaseMessage, source: SatID, destination: SatID):
        super().__init__(time, ["message", "source", "destination"])
        self.message = message
        self.source = source
        self.destination = destination
    
class LTPSegmentDroppedEvent(InstantEvent):
    """
    Event for when an LTP Segment is dropped.
    Source refers to the satellite that dropped the message.
    """

    message: LTPSegment

    def __init__(self, time: float, source: SatID, message: LTPSegment, reason: DropReason = DropReason.UNKNOWN):
        super().__init__(time, ["source", "message", "reason"])
        self.source = source
        self.message = message
        self.reason = reason

class AbortReason(Enum):
    LTP_SESSION_CANCELATION = "LTP_SESSION_CANCELATION"
    UNKNOWN = "UNKNWON"

class MessageReceptionCanceledEvent(InstantEvent):
    """
    Event for when a receiver aborts receiving a message from a sender.
    Note sender can drop the message, but receiver cannot drop something they don't have yet.
    They can only abort its reception.
    """

    message: BaseMessage

    def __init__(self, time: float, source: SatID, message: BaseMessage, destination: SatID, reason: AbortReason = AbortReason.UNKNOWN):
        super().__init__(time, ["source", "destination", "message", "reason"])
        self.source = source
        self.destination = destination
        self.message = message
        self.reason = reason

@dataclass
class RetransmissionConfig:
    """
    Configuration for retransmission timeouts.

    "Then, to account for the additional delay imposed by interrupted
        connectivity, we dynamically suspend timers during periods when
        the relevant remote LTP engines are known to be unable to
        transmit responses." - Not a concern as we start timers when message fully transmitted.

    LTP RFC 5325 - Read Section '3.1.3.  Timers' for Justification
    "So the initial expected arrival time for each acknowledging segment
    is typically computed as simply the current time at the moment that
    radiation of the original segment begins, plus twice the one-way
    light time, plus 2*N seconds of margin to account for processing and
    queuing delays and for radiation time at both ends.  N is a parameter
    set by network management for which 2 seconds seem to be a reasonable
    default value."

    We simplify this by starting timers when the message is fully transmitted into the link 
    (but since message size <<< bandwidth this is negligible).

    Then compute timeout by default as recommende in the RFC as RTT + 2*N, where N=2 as recommended
    """
    timeout: Callable[[float], float] = lambda RTT: RTT + 2 * 2

    """
    No real optimal value. Based on RFC this is a limit dependent on the network config. 
    Given the loose retransmission timeout (extra 4 seconds on RTT), 10 retries should be sufficient.

    https://support.microsoft.com/en-gb/topic/how-to-modify-the-tcp-ip-maximum-retransmission-time-out-7ae0982a-4963-fa7e-ee79-ff6d0da73db8
    TCP default value for TcpMaxDataRetransmissions is 5. Standard value I would argue. So this is more than enough.
    """
    max_retries: int = 10

@dataclass(frozen=True)
class LTPSessionID:
    source: SatID
    destination: SatID
    underlying_message_uid: int
    session: int

class LTPMessageRetransmissionActor(Actor):
    """
    https://www.rfc-editor.org/rfc/pdfrfc/rfc5326.txt.pdf
    RFC 5326
    Implements LTP based retransmission for satelite communication.
    Each original direct message is now encapsulated as a series of LTP segments in an LTPBlock.
    The remaining LTPMessage's are control messages for sending a checkpoint, report segment and report ack.

    Each LTPBlock has a red part for reliable transfer and green part for unreliable transfer.
    Each LTPBlock must be followed by a checkpoint segment at the end.
    Checkpoints require sender to reply with a report segment of missing data
    When a sender gets a report segment they acknowledge it and retransmit missing data.

    Checkpoints and report segments have timeouts.

    Retransmission timers for timeouts are started when the message is sent
    and not when it's queued. This is so that we don't retransmit messages that are waiting for
    transmission because of bandwidth availability.

    """
    def __init__(self, 
                 config: RetransmissionConfig, 
                 report_segment_base_size: int = 9, 
                 report_ack_size: int = 5, 
                 model_bandwidth = False,                  
                 message_location_tracker: MessageLocationTracker = MessageLocationTracker()):
        super().__init__()
        self._config = config
        self._model_bandwidth = model_bandwidth
        self._message_location_tracker = message_location_tracker

        """
        NB: According to: https://www.rfc-editor.org/rfc/pdfrfc/rfc5326.txt.pdf Sections 3.2.2.
        Suppose we received N data segments.
        Default base size of ack is 9 + 2N bytes. This contains
        - 4 Header Bytes: Control byte (1) + Session ID (1+1 bytes minimally) + Extensions byte (1 byte to indicate 0 extensions in header and trailer).
        - 5 Segment Content Bytes: Report serial (1) + Checkpoint serial (1) + Upper bound (1) + Lower bound (1) + Claim count (1)
            - N claims of received segments, each with 1 byte offset and 1 byte length.
        """
        # number of reported segments to size of report segment
        self._report_segment_size: callable[int, int] = lambda num_claims: (2 * num_claims) + report_segment_base_size

        """
        NB: According to: https://www.rfc-editor.org/rfc/pdfrfc/rfc5326.txt.pdf Sections 3.0 (and 3.1.4)
        Default base size of ack is 5 bytes. This contains
        - 4 Header Bytes: Control byte (1) + Session ID (1+1 bytes minimally) + Extensions byte (1 byte to indicate 0 extensions in header and trailer).
        - 1 content Byte to indicate the report segment number (id)
        """
        self._report_ack_size = report_ack_size
        
        # buffer sent LTP data segments until all are reported as received in report segment
        # UnderlyingMessageID, Link --> LTPSegmentUID --> LTPMessage  
        self._sent_data_segments: dict[LTPSessionID, dict[int, LTPDataSegment]] = {}

        # LTPCheckpointDataSegmentUID --> LTPCheckpointDataSegmentUID, Remaining Retries
        # Used for retransmitting checkpoints on timeout
        self._sender_unreported_checkpoints: dict[int, tuple[LTPDataSegment, int]] = {}

        # keep track of LTPCheckpointDataSegmentUID that sender received report for
        # such that we don't schedule timeouts on a checkpoint for which we already received a report
        self._sender_reported_checkpoints: set[int] = set()

        # all LTP Reports UIDs received by any sender. If ack is lost we need to keep track of this.
        self._sender_received_reports: set[int] = set()

        # LTP sessions canceled identified by underlying message's UID and checkpoint message UID
        self._sender_dropped_transmissions: set[LTPSessionID] = set()

        # buffer received LTP data segments until all are received and acked in report ack
        # UnderlyingMessageID, Link --> LTPSegmentUID --> LTPMessage
        self._received_data_segments: dict[LTPSessionID, dict[int, LTPDataSegment]] = {}

        # LTPCheckpointDataSegmentUID --> LTPCheckpointDataSegmentUID, Remaining Retries
        # used for resending the report if we get a duplicate checkpoint before timeout
        # note only when we send do we start timer and add to unreported checkpoints
        # hence if we get a duplicate checkpoint for a qeueud for transmission report we need this
        self._receiver_report_for_checkpoint: dict[int, LTPReportSegment] = {}

        # LTPReportSegmentUID --> LTPReportSegment, Remaining Retries
        # used for resending reports on tiemout
        self._receiver_unacked_reports: dict[int, tuple[LTPReportSegment, int]] = {}

        # all receiver sent LTP report segment UIDs whose acks were received by receiver. 
        # Such that we don't schedule timeouts on reports for which we already got an ack for.
        self._receiver_acked_reports: set[int] = set()

        # received messages, such that we don't conjoin segments twice on a given ack
        # can in theory check received segments or other data structures that get cleaned up
        # when the message is fully received. But this is cleaner
        self._receiver_received_underlying_messages: set[LTPSessionID] = set()

        # keep track of dropped reports to not resend a dropped report on its timeout
        self._receiver_dropped_reports: set[int] = set()

        # keep track of dropped sessions to not handle acks and data segments on those after dropping
        self._receiver_dropped_sessions: set[LTPSessionID] = set()

        # receiver keeps track of latest ltp session they get
        self._receiver_latest_ltp_session_received: dict[LinkMessageID, int] = {}

        # sender's latest ltp session
        # used for excluding tranmissions from closed sessions
        self._sender_latest_ltp_session: dict[SatID, dict[int, int]] = {}

    def initialize(self) -> list[Event]:
        return []
    
    def __get_send_message_events(self, time: float, source: SatID, destination: SatID, message: LTPSegment) -> list[Event]:
        message = message
        if self._model_bandwidth:
            return [ MessageQueuedEvent(time, source, destination, message) ]
        else:
            return [ MessageSentEvent(time, source, destination, message) ]
    
    def _compute_timeout(self, mobility: MultiConstellation, message: LTPSegment) -> float:
        one_way_delay = mobility.get_delay(message.source, message.destination)
        round_trip_time = 2 * one_way_delay
        return self._config.timeout(round_trip_time)

    def handle_message_sent_event(self, mobility: MultiConstellation, event: MessageSentEvent, message: LTPSegment) -> list[Event]:
        segment_uid = message.uid
        if isinstance(message, LTPDataSegment):

            if message.is_end_of_green_only_block:
                # if its green only, we don't expect a report segment so from sender point of view, message handling is done when sent
                link = Link(event.source, event.destination)
                self._message_location_tracker.remove_message_from_location(message.underlying_message, link)

            if message.uid not in self._sender_reported_checkpoints:
                ltp_session_id = LTPSessionID(event.source, event.destination, message.underlying_message.uid, message.session)
                if segment_uid not in self._sent_data_segments.get(ltp_session_id, {}) and message.data_type == LTPDataType.RED:
                    self._sent_data_segments.setdefault(ltp_session_id, {})[segment_uid] = message
                if message.is_checkpoint:
                    if segment_uid not in self._sender_unreported_checkpoints:
                        self._sender_unreported_checkpoints[segment_uid] = (message, self._config.max_retries)
                    timeout = self._compute_timeout(mobility, message)
                    return [ MessageTransmissionTimeoutEvent( time=event.time + timeout, message=message, source=event.source, destination=event.destination ) ]
                else:
                    return []
            else:
                # its a retransmission of a checkpoint that it already received a report for. Don't schedule another timeout!
                assert message.uid not in self._sender_unreported_checkpoints
                # cleanup as we no longer have scheduled retransmissions for it
                self._sender_reported_checkpoints.remove(message.uid)
                return []
            
        elif isinstance(message, LTPReportSegment):
            if message.uid not in self._receiver_acked_reports:
                if segment_uid not in self._receiver_unacked_reports:
                    self._receiver_unacked_reports[segment_uid] = (message, self._config.max_retries)
                timeout = self._compute_timeout(mobility, message)
                return [ MessageTransmissionTimeoutEvent( time=event.time + timeout, message=message, source=event.source, destination=event.destination ) ] 
            else:
                # its a retransmission of a report that receiver already got an ack for. Don't schedule another timeout!
                assert message.uid not in self._receiver_unacked_reports
                # cannot cleanup _receiver_acked_reports because a report can have more than 1 timeout waiting (e.g. if we received multiple checkpoints)
                return []       
        elif isinstance(message, LTPReportAcknowledgementSegment):
            return []
        else:
            raise RuntimeError(f"Unexpected LTP Message obtained: {message}")

    def handle_data_segment_received_event(self, mobility: MultiConstellation, event: LTPSegmentReceivedEvent, message: LTPDataSegment) -> list[Event]:
        segment_uid = message.uid
        link_message_id = LinkMessageID(event.source, event.destination, message.underlying_message.uid)
        ltp_session_id = LTPSessionID(event.source, event.destination, message.underlying_message.uid, message.session)

        # if receiver aborted receipt and cancelled session we should gracefully ignore this
        if ltp_session_id in self._receiver_dropped_sessions:
            return []
        
        session = max(self._receiver_latest_ltp_session_received.get(link_message_id, -1), message.session)
        self._receiver_latest_ltp_session_received[link_message_id] = session

        if session != message.session:
            if message.is_checkpoint:
                if segment_uid in self._receiver_report_for_checkpoint:
                    # duplicate data segment from previous hop or before rerouting
                    # according to RFC we resend the report segment for it but do nothig more
                    report_segment = self._receiver_report_for_checkpoint.get(segment_uid)
                    return self.__get_send_message_events(event.time, report_segment.source, report_segment.destination, report_segment)
                else:
                    # likely has already been cleaned up, so ignore gracefully
                    assert False, f"Didn't expect to get {message}"
                    return []
            else:
                assert False, f"This should have been aborted {message}"
                return []

        else:
            # latest session so receive if possible and send report segment where applicable
            if ltp_session_id not in self._receiver_received_underlying_messages:
                self._received_data_segments.setdefault(ltp_session_id, {})[segment_uid] = message

                # if this is end of all green block we now process all segments to reconstruct the message
                if message.is_end_of_green_only_block:  
                    return self.__receive_message_from_segments(event, message)

            events = []
            if message.is_checkpoint:
                """
                Receive the underlying message if we now have all segments
                """
                if ltp_session_id not in self._receiver_received_underlying_messages:
                    sent_red_segments_uids = message.checkpoint_data.expected_uids
                    received_segment_uids = [
                        uid for uid, segment in self._received_data_segments.get(ltp_session_id).items() 
                        if segment.data_type == LTPDataType.RED 
                        and segment.session == message.session
                    ]
                    if len(set(sent_red_segments_uids).difference(received_segment_uids)) == 0: 
                        events.extend(self.__receive_message_from_segments(event, message))
                
                # if message has already been received on link, we expect it to have new session if routed onto next hop
                # but if not that could mean either it's stuck in store and forward or delivered and thus we still need to handle
                # duplicate checkpoint segments properly.
                if segment_uid in self._receiver_report_for_checkpoint:
                    # If its a duplicate checkpoint we resend the same report with its own timeout
                    # See https://www.rfc-editor.org/rfc/pdfrfc/rfc5326.txt.pdf 6.8
                    report_segment = self._receiver_report_for_checkpoint.get(segment_uid)
                    events.extend(self.__get_send_message_events(event.time, report_segment.source, report_segment.destination, report_segment))
                else:
                    """
                    The scenario where we get a checkpoint retransmission after the LTP session is done (underlying message received)
                    should not happen. It would mean receiver got an ack for a report it generated for this checkpoint.
                    When sender generated this ack it marked that checkpoint as reported. This retransmission must have been
                    generated after the ack was queued. But then the checkpoint wouldn't be in _sender_unreported_checkpoints so
                    the retransmission couldn't have been generated (as per the timeout handler code)
                    """
                    num_claims = len(received_segment_uids)
                    report_segment = LTPReportSegment(time=event.time,
                                                    source=event.destination, 
                                                    destination=event.source, 
                                                    size=self._report_segment_size(num_claims), 
                                                    underlying_message=message.underlying_message,
                                                    report_checkpoint_uid=segment_uid,
                                                    received_segment_uids=received_segment_uids,
                                                    session=message.session,
                                                    )
                    self._receiver_report_for_checkpoint[segment_uid] = report_segment
                    events.append(LTPSegmentCreatedEvent(time=event.time, message=report_segment))

        return events
    
    def handle_report_segment_received_event(self, mobility: MultiConstellation, event: LTPSegmentReceivedEvent, message: LTPReportSegment) -> list[Event]:
        """
        If its a duplicate report segment
        We only ack it but don't retransmit missing data again
        As per 6.13 in https://www.rfc-editor.org/rfc/pdfrfc/rfc5326.txt.pdf

        As per 6.13 if the session is canceled (underlying message or corresponding checkpoint is dropped) we only ack.
        """
        events = []
        
        report_ack = LTPReportAcknowledgementSegment(source=event.destination, destination=event.source, 
                                                     size=self._report_ack_size, underlying_message=message.underlying_message, 
                                                     acked_report_message_uid=message.uid, session=message.session)
        events.append(LTPSegmentCreatedEvent(event.time, report_ack))

        ltp_checkpoint_id = LTPSessionID(event.destination, event.source, message.report_checkpoint_uid, message.session)
        ltp_session_id = LTPSessionID(event.destination, event.source, message.underlying_message.uid, message.session)
        latest_sender_session = self._sender_latest_ltp_session.get(event.destination, {}).get(message.underlying_message.uid, -1)

        if (message.uid not in self._sender_received_reports 
            # and message.report_checkpoint_uid not in self._sender_dropped_transmissions
            and ltp_checkpoint_id not in self._sender_dropped_transmissions 
            and ltp_session_id not in self._sender_dropped_transmissions
            and message.session == latest_sender_session
            ):
            # Sender sends C1. Receiver replies with R1. Sender times out and resends C1. Sender gets R1 and sends ack. 
            # maybe edge case where ack sent then report resent then ack received and message received.
            # message received induces sent segments deleted; but then report segment comes in but we don't have segments for it
            received_segments = message.received_segment_uids
            checkpoint, _ = self._sender_unreported_checkpoints.get(message.report_checkpoint_uid)
            sent_red_segments = checkpoint.checkpoint_data.expected_uids
            lost_segments = [uid for uid in sent_red_segments if uid not in received_segments]

            segments_to_retransmit = [self._sent_data_segments.get(ltp_session_id).get(uid) for uid in lost_segments]
            assert None not in segments_to_retransmit, f"Trying to retransmit None segment in: {segments_to_retransmit}"

            if segments_to_retransmit:
                segments_to_retransmit[-1] = segments_to_retransmit[-1].copy()
                segments_to_retransmit[-1].is_checkpoint = True
                segments_to_retransmit[-1].checkpoint_data = LTPCheckpointData(expected_uids=sent_red_segments)
            else:
                # all segments were delivered according to the report segment, handling of the underlying message by satelite is done
                link = Link(event.destination, event.source)
                self._message_location_tracker.remove_message_from_location(message.underlying_message, link)
                del self._sent_data_segments[ltp_session_id]
                if event.destination in self._sender_latest_ltp_session:
                    if message.underlying_message.uid in self._sender_latest_ltp_session[event.destination]:
                        del self._sender_latest_ltp_session[event.destination][message.underlying_message.uid]
                    if not self._sender_latest_ltp_session[event.destination]:
                        del self._sender_latest_ltp_session[event.destination]

            for segment in segments_to_retransmit:
                events.extend( self.__get_send_message_events(event.time, event.destination, event.source, segment) )
        
        # even if the above clause doesn't trigger, we must do cleanup
        # It could be because the session updated since all segments were received, 
        # in which case we must cleanup to avoid retransmissions
        self._sender_unreported_checkpoints.pop(message.report_checkpoint_uid, None)
        self._sender_received_reports.add(message.uid)
        self._sender_reported_checkpoints.add(message.report_checkpoint_uid)            
        
        return events
    
    def __receive_message_from_segments(self, event: LTPSegmentReceivedEvent, message: LTPSegment) -> list[Event]:
        """
        Callable either when we receive end of block for an all green data block
        or when we receive an ack for a report saying all red segments are received.
        This method reconstructs the underlying message from the segments received,
        updating the green data size and overall size of the message as neccessary,
        and returns a MessageReceivedEvent for the underlying message.
        """
        ltp_session_id = LTPSessionID(event.source, event.destination, message.underlying_message.uid, message.session)

        underlying_message = message.underlying_message.copy()
        received_green_part_size = sum([segment.size for _, segment in self._received_data_segments.get(ltp_session_id).items() if segment.data_type == LTPDataType.GREEN])
        lost_green_part_size = underlying_message.unreliable_data_size - received_green_part_size
        underlying_message.size -= lost_green_part_size
        underlying_message.unreliable_data_size -= lost_green_part_size

        received_red_part_size = sum([segment.size for _, segment in self._received_data_segments.get(ltp_session_id).items() if segment.data_type == LTPDataType.RED])
        assert received_red_part_size == underlying_message.reliable_data_size

        underlying_message.hops += 1

        if ltp_session_id in self._received_data_segments:
            self._receiver_received_underlying_messages.add(ltp_session_id)
            del self._received_data_segments[ltp_session_id]

        return [MessageReceivedEvent(event.time, event.source, event.destination, underlying_message)]

    def handle_report_ack_received_event(self, mobility: MultiConstellation, event: LTPSegmentReceivedEvent, message: LTPReportAcknowledgementSegment) -> list[Event]:
        events = []     
        link_message_id = LinkMessageID(message.source, message.destination, message.underlying_message.uid)
        if message.session == self._receiver_latest_ltp_session_received.get(link_message_id, -1):
            # otherwise its a stale ack we don't want to act on   
            if message.acked_report_message_uid in self._receiver_unacked_reports:
                # otherwise it's a duplicate ack so we do nothing
                report_segment, _ = self._receiver_unacked_reports.get(message.acked_report_message_uid)

                del self._receiver_unacked_reports[message.acked_report_message_uid]
                self._receiver_acked_reports.add(message.acked_report_message_uid)
                del self._receiver_report_for_checkpoint[report_segment.report_checkpoint_uid]
                del self._receiver_latest_ltp_session_received[link_message_id]
        else:
            # cleanup if its a stale session ack
            self._receiver_unacked_reports.pop(message.acked_report_message_uid, None)
            report_segment = self._receiver_unacked_reports.get(message.acked_report_message_uid, None)
            if report_segment:
                self._receiver_report_for_checkpoint.pop(report_segment.report_checkpoint_uid, None)
            self._receiver_acked_reports.add(message.acked_report_message_uid)

        return events
    
    def handle_message_dropped_event(self, mobility: MultiConstellation, event: LTPSegmentDroppedEvent, message: LTPSegment) -> list[Event]:
        
        if isinstance(message, LTPDataSegment):
            ltp_session_id = LTPSessionID(message.source, message.destination, message.underlying_message.uid, message.session)
            latest_sender_session = self._sender_latest_ltp_session.get(message.source, {}).get(message.underlying_message.uid, -1)
            if ltp_session_id not in self._sender_dropped_transmissions and latest_sender_session == message.session:
                self._sender_dropped_transmissions.add(ltp_session_id)
                link = Link(message.source, message.destination)
                self._message_location_tracker.remove_message_from_location(message.underlying_message, link)
                return [ MessageDroppedEvent(time=event.time, source=message.source, message=message.underlying_message, reason=DropReason.RETRANSMISSION_RETRIES_EXHAUSTED) ]
            
        elif isinstance(message, LTPReportSegment):
            link_message_id = LinkMessageID(message.destination, message.source, message.underlying_message.uid)
            ltp_session_id = LTPSessionID(message.destination, message.source, message.underlying_message.uid, message.session)
            self._receiver_report_for_checkpoint.pop(message.report_checkpoint_uid, None)

            # receiver aborts if they don't get ack. also don't retransmit or abort on a torn down session?
            if ltp_session_id not in self._receiver_dropped_sessions and self._receiver_latest_ltp_session_received.get(link_message_id, -1) == message.session:
                self._receiver_dropped_sessions.add(ltp_session_id)
                del self._receiver_latest_ltp_session_received[link_message_id]
                return [ MessageReceptionCanceledEvent(time=event.time, source=message.destination, destination=message.source, message=message.underlying_message, reason=AbortReason.LTP_SESSION_CANCELATION) ]
        return []
    
    def handle_message_transmission_timeout_event(self, mobility: MultiConstellation, event: MessageTransmissionTimeoutEvent, message: LTPSegment) -> list[Event]:
        if isinstance(message, LTPDataSegment):
            assert message.is_checkpoint
            ltp_session_id = LTPSessionID(event.source, event.destination, message.uid, message.session)
            if message.uid in self._sender_unreported_checkpoints:
                # theres an edge case where something is rerouted but link is down so its in store and forward
                # in this case if we get a timeout the session is still latest, and so we would retransmit it.
                # but we don't want to as session was 'canceled'. Fix: only retransmit if message still at that location
                link = Link(event.source, event.destination)
                if (ltp_session_id not in self._sender_dropped_transmissions 
                    and self._sender_latest_ltp_session.get(event.source, {}).get(message.underlying_message.uid, -1) == message.session
                    and self._message_location_tracker.is_message_at_location(message.underlying_message, link)):
                    # retransmit checkpoint
                    checkpoint, retries_left = self._sender_unreported_checkpoints.get(message.uid)
                    if retries_left > 0:
                        self._sender_unreported_checkpoints[message.uid] = (checkpoint, retries_left - 1)
                        return self.__get_send_message_events(event.time, message.source, message.destination, checkpoint)
                    else:
                        del self._sender_unreported_checkpoints[message.uid]
                        self._sender_dropped_transmissions.add(ltp_session_id)
                        return [ LTPSegmentDroppedEvent(event.time, message.source, message, DropReason.RETRANSMISSION_RETRIES_EXHAUSTED) ]
                else:
                    del self._sender_unreported_checkpoints[message.uid]

        elif isinstance(message, LTPReportSegment):
            underlying_link_message_id = LinkMessageID(message.destination, message.source, message.underlying_message.uid)
            ltp_session_id = LTPSessionID(message.destination, message.source, message.underlying_message.uid, message.session)
            if message.uid in self._receiver_unacked_reports:
                if (message.uid not in self._receiver_dropped_reports 
                    and ltp_session_id not in self._receiver_dropped_sessions 
                    and self._receiver_latest_ltp_session_received.get(underlying_link_message_id, -1) == message.session):
                    # retransmit report
                    report_segment, retries_left = self._receiver_unacked_reports.get(message.uid)
                    if retries_left > 0:
                        self._receiver_unacked_reports[message.uid] = (report_segment, retries_left - 1)
                        return self.__get_send_message_events(event.time, message.source, message.destination, report_segment)
                    else:
                        del self._receiver_unacked_reports[message.uid]
                        self._receiver_dropped_reports.add(message.uid)
                        return [ LTPSegmentDroppedEvent(event.time, message.source, message, DropReason.RETRANSMISSION_RETRIES_EXHAUSTED) ]
                else:
                    del self._receiver_unacked_reports[message.uid]
        return []
    
    def handle_ltp_segment_created_event(self, mobility: MultiConstellation, event: LTPSegmentCreatedEvent, message: LTPSegment):
        underlying_message_uid = message.underlying_message.uid
        if isinstance(message, LTPDataSegment):
            self._sender_latest_ltp_session.setdefault(message.source, {})[underlying_message_uid] = max(message.session, self._sender_latest_ltp_session.get(message.source, {}).get(underlying_message_uid, -1))
        return self.__get_send_message_events(event.time, message.source, message.destination, message)

    def handle_message_reroute_event(self, mobility: MultiConstellation, event: MessageRerouteEvent):
        # (1) drop old LTP session
        # (2) cleanup old LTP session data, assume that if message was received as subequent hops, it was at most 2 hops away
        current_session = self._sender_latest_ltp_session.get(event.source, {}).get(event.message.uid, -1)
        lookback_limit = 2
        # note we assume a link won't flicker up and down within miliseconds - otherwise edge cases may arise here on lookback = 0
        for lookback in range(lookback_limit, -1, -1):
            candidate_ltp_session_id = LTPSessionID(event.source, event.previous_destination, event.message.uid, current_session - lookback)
            if candidate_ltp_session_id in self._sent_data_segments:
                self._sender_dropped_transmissions.add(candidate_ltp_session_id)
                del self._sent_data_segments[candidate_ltp_session_id]
                break
        return []

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        if isinstance(event, MessageSentEvent):
            if isinstance(event.message, LTPSegment):
                return self.handle_message_sent_event(mobility, event, event.message)
            
        if isinstance(event, LTPSegmentReceivedEvent):
            if isinstance(event.message, LTPDataSegment):
                return self.handle_data_segment_received_event(mobility, event, event.message)
            elif isinstance(event.message, LTPReportSegment):
                return self.handle_report_segment_received_event(mobility, event, event.message)
            elif isinstance(event.message, LTPReportAcknowledgementSegment):
                return self.handle_report_ack_received_event(mobility, event, event.message)
            else:
                raise RuntimeError(f"Got unexpected LTP segment received event: {event}")
        
        if isinstance(event, LTPSegmentDroppedEvent):
            if isinstance(event.message, LTPSegment):
                return self.handle_message_dropped_event(mobility, event, event.message)

        if isinstance(event, MessageTransmissionTimeoutEvent):
            return self.handle_message_transmission_timeout_event(mobility, event, event.message)
        
        if isinstance(event, LTPSegmentCreatedEvent):
            return self.handle_ltp_segment_created_event(mobility, event, event.message)
        
        if isinstance(event, MessageRerouteEvent):
            return self.handle_message_reroute_event(mobility, event)
        
        return []
        
