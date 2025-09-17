from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
from collections import defaultdict
from enum import Enum
from typing import Generator, Iterator, Optional, Callable

import numpy as np

from .events import Event, InstantEvent
from .helpers import SatID

def hashable(cls):
    """
    Decorator for Message dataclasses that makes them hashable.
    This uses the source, data, and time fields for hashing.
    The type is also used to distinguish between different message types.
    It also implements __eq__ for comparison.
    """
    def __hash__(self):
        if isinstance(self, DirectMessage):
            return hash((self.source, self.destination, self.data, self.time, self.uid, self.size, type(self)))
        else:
            return hash((self.source, self.data, self.time, self.uid, self.size, type(self)))

    def __eq__(self, other):
        if not isinstance(other, cls):
            return NotImplemented
        if isinstance(self, DirectMessage):
            return self.source == other.source and self.destination == other.destination and self.data == other.data and self.time == other.time and self.uid == other.uid and self.size == other.size and type(self) == type(other)
        else:
            return self.source == other.source and self.data == other.data and self.time == other.time and self.uid == other.uid and self.size == other.size and type(self) == type(other)

    cls.__hash__ = __hash__
    cls.__eq__ = __eq__
    return cls


def broadcast(cls):
    """
    Decorator for BroadcastMessage dataclasses that makes them hashable, and adds the necessary dataclass decorator.
    """
    return hashable(dataclass(kw_only=True)(cls))

"""
NB Source: https://github.com/rossilor95/lcg-python/blob/main/lcg.py
https://en.wikipedia.org/wiki/Linear_congruential_generator
Period is m iff Hull-Dobell Theorem:
    - m is power of 2
    - c is non-zero
    - m and c are coprime
    - a-1 is divisible by all prime factors of m
    - a-1 is divisble by 4 if m is divisible by 4

"""
def lcg_id_generator(seed=0) -> Iterator[int]:
    """
    a: LCG Multiplier
    c: LCG Increment
    m: Modulo, currently 2^32 but TODO consider increasing to 2^64 if you see repetitions in UIDs!
    """
    a = 1664525
    c = 1013904223
    m = 2**32
    while True:
        seed = (a * seed + c) % m
        yield seed

UID_GENERATOR = lcg_id_generator(seed=0)

@hashable
@dataclass(kw_only=True)
class BaseMessage:
    """
    Base class for messages. This class should not be used directly.
    """

    source: SatID
    data: object = None # Data to be sent
    time: float = 0.0 # Time the message was created
    hops: int = 0 # Number of hops the message has taken
    dropped: float = False # Whether the message should be dropped
    size: int = 0 # Number of bytes in message (for bandwidth and retransmission functionality)

    # Unique identifier
    """
    I run into edge case where two different messages have same UID. These can be two checkpoints
    which give race conditions in my LTP transmission functionality. A fix could be
    to store reported checkpoints and other fields as LinkMessageIDs. However, if the same message has the same UID on the same
    link we get to same race. So fix is to cycle periodically through UID.
    Fix: Use LCG Psuedorandom generator
    """
    # NB:
    uid: int = field(init=False, repr=True)

    def __post_init__(self):
        """
        Initialise the message, generating a random uid.
        """
        self.uid = next(UID_GENERATOR)

    def copy(self, keep_uid: bool = True):
        """
        Copy the message.

        Returns:
            Copy of the message.
        """
        r = replace(self)
        if keep_uid:
            r.uid = self.uid
        return r


@hashable
@dataclass(kw_only=True)
class DirectMessage(BaseMessage):
    """
    Message to be sent between nodes.
    """

    destination: SatID

@broadcast
class BroadcastMessage(BaseMessage):
    """
    Message to be broadcast between nodes.
    """

# Start of Message Retransmission Additions
@dataclass(frozen=True)
class LinkLossProbability:
    source: SatID
    destination: SatID
    reset_loss_probability: bool = False
    loss_probability: float = 0.0

    def __post_init__(self):
        if not self.reset_loss_probability and self.loss_probability is None or self.loss_probability < 0:
                raise RuntimeError(f"Got invalid combination of loss probability update event. Got reset flag set to: {self.reset_loss_probability} and probability: {self.loss_probability}")


class LinkLossProbabilityUpdateEvent(InstantEvent):
    """
    Event to update the probability of message loss on given links
    """
    updates: list[LinkLossProbability] = []

    def __init__(self, time: float, updates: list[LinkLossProbability]):
        super().__init__(time, ["updates"])
        self.updates = updates


@hashable
@dataclass(kw_only=True)
class HybridDirectMessage(DirectMessage):
    reliable_data_size: int
    unreliable_data_size: int
    # TODO this can be extended with extended with priority of message and others

    def __post_init__(self):
            super().__post_init__()
            if self.unreliable_data_size + self.reliable_data_size != self.size:
                raise Exception(f"Got mismatch on message sizes. Got reliable part size: {self.reliable_data_size}, unreliable part size: {self.unreliable_data_size} but total size: {self.size}")


@hashable
@dataclass(kw_only=True)
class LTPSegment(DirectMessage):
    underlying_message: HybridDirectMessage
    session: int

class LTPDataType(Enum):
    RED = "RED"
    GREEN = "GREEN"


@dataclass(kw_only=True)
class LTPCheckpointData:
    expected_uids: list[int]

@hashable
@dataclass(kw_only=True)
class LTPDataSegment(LTPSegment):
    data_type: LTPDataType

    is_checkpoint: bool = False
    checkpoint_data: LTPCheckpointData = None

    is_end_of_green_only_block: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.data_type != LTPDataType.RED and self.is_checkpoint:
            raise ValueError(f"Checkpoint segments must be last RED data segment transmitted")

        if self.is_checkpoint and not self.checkpoint_data:
            raise ValueError(f"Checkpoint segments must have checkpoint fields filled in")

        if self.is_end_of_green_only_block and self.data_type == LTPDataType.RED:
            raise ValueError(f"Data segment is indicated to be part of an all green message but is a red segment")

@hashable
@dataclass(kw_only=True)
class LTPReportSegment(LTPSegment):
    report_checkpoint_uid: int
    received_segment_uids: list[int]

@hashable
@dataclass(kw_only=True)
class LTPReportAcknowledgementSegment(LTPSegment):
     acked_report_message_uid: int

@dataclass(frozen=True)
class LinkMessageID:
    source: SatID
    destination: SatID
    uid: int

@dataclass(frozen=True)
class Link:
    source: SatID
    destination: SatID

class MessageLostEvent(InstantEvent):

    source: SatID
    destination: SatID
    message: LTPSegment

    def __init__(self, time: float, source: SatID, destination: SatID, message: BaseMessage):
        super().__init__(time, ["source", "destination", "message"])
        self.source = source
        self.destination = destination
        self.message = message

class LTPSegmentReceivedEvent(Event):
    """
    Event for when LTP segments are received.
    Source and destination refer to the satellites that sent and received the message, respectively.
    Once all LTP messages are received and processed, a MessageReceivedEvent is issued for underlying message
    """

    message: LTPSegment

    def __init__(self, time: float, source: SatID, destination: SatID, message: LTPSegment):
        super().__init__(time, ["source", "destination", "message"])
        self.source = source
        self.destination = destination
        self.message = message


class LossConfig:
    _base_seed: int
    _rngs: dict[Link, Generator]
    _default_loss_probability: float
    _loss_probabilities: dict[Link, float]

    def __init__(self, seed: float=0, default_loss_probability: float=0):
        self._base_seed = seed
        self._default_loss_probability = default_loss_probability
        self._loss_probabilities = {}
        self._rngs = {}

    def set_loss_probability_for_link(self, source: int, destination: int, loss_probability: float) -> None:
        link = Link(source=source, destination=destination)
        self._loss_probabilities[link] = loss_probability

    def reset_loss_probability_for_link(self, source: int, destination: int) -> None:
        link = Link(source=source, destination=destination)
        self._loss_probabilities.pop(link, None)

    def _get_loss_probability_for_link(self, link: Link) -> float:
        return self._loss_probabilities.get(link, self._default_loss_probability)

    def _get_rng_for_link(self, link: Link) -> Generator:
        if link not in self._rngs:
            link_seed = hash((self._base_seed, link.source, link.destination)) & 0xFFFFFFFF
            self._rngs[link] = np.random.default_rng(link_seed)
        return self._rngs[link]

    def is_message_lost(self, source: int, destination: int, size: int) -> bool:
        link = Link(source=source, destination=destination)
        rng = self._get_rng_for_link(link)
        return rng.random() < self._get_loss_probability_for_link(link)



# End of Message Retransmission Additions

# Start of Message Bandwidth Additions

class MessageRerouteEvent(InstantEvent):
    """
    Event for when a message has been queued on the link,
    but the link went down so now we want to reroute it
    by passing it up from link level to network level
    """

    message: BaseMessage
    previous_destination: SatID

    def __init__(self, time: float, source: SatID, previous_destination: SatID, message: BaseMessage):
        super().__init__(time, ["source", "previous_destination", "message"])
        self.source = source
        self.previous_destination = previous_destination
        self.message = message

class MessageQueuedEvent(InstantEvent):
    """
    Event for when a message has been queued on the link,
    waiting to be transmitted with respect to bandwidth capacity
    """

    message: BaseMessage

    def __init__(self, time: float, source: SatID, destination: SatID, message: BaseMessage):
        super().__init__(time, ["source", "destination", "message"])
        self.source = source
        self.destination = destination
        self.message = message

# End of Message Bandwidth Additions

class MessageSentEvent(InstantEvent):
    """
    Event for when a message is sent.
    Source and destination refer to the satellites that sent and received the message, respectively.
    """

    message: BaseMessage

    def __init__(self, time: float, source: SatID, destination: SatID, message: BaseMessage):
        super().__init__(time, ["source", "destination", "message"])
        self.source = source
        self.destination = destination
        self.message = message


class MessageReceivedEvent(Event):
    """
    Event for when a message is received.
    Source and destination refer to the satellites that sent and received the message, respectively.
    """

    message: BaseMessage

    def __init__(self, time: float, source: SatID, destination: SatID, message: BaseMessage):
        super().__init__(time, ["source", "destination", "message"])
        self.source = source
        self.destination = destination
        self.message = message


class MessageDeliveredEvent(InstantEvent):
    """
    Event for when a message is delivered.
    """

    destination: SatID
    message: BaseMessage

    def __init__(self, time: float, destination: SatID, message: BaseMessage):
        super().__init__(time, ["destination", "message"])
        self.destination = destination
        self.message = message

class MessageBroadcastDeliveredEvent(InstantEvent):
    """
    Event for when a message broadcast has been delivered to all nodes in the network.
    """

    message: BroadcastMessage

    def __init__(self, time: float, message: BroadcastMessage):
        super().__init__(time, ["message"])
        self.message = message

class DropReason(Enum):
    DUPLICATE = "DUPLICATE"
    INSUFFICIENT_BUFFER = "INSUFFICIENT_BUFFER"
    MISCONFIGURED = "MISCONFIGURED"
    INDICATED = "INDICATED"
    NO_NEXT_HOP = "NO_NEXT_HOP"
    RETRANSMISSION_RETRIES_EXHAUSTED = "RETRANSMISSION_RETRIES_EXHAUSTED"
    UNKNOWN = "UNKNOWN"

class MessageDroppedEvent(InstantEvent):
    """
    Event for when a message is dropped.
    Source refers to the satellite that dropped the message.
    """

    message: BaseMessage

    def __init__(self, time: float, source: SatID, message: BaseMessage, reason: DropReason = DropReason.UNKNOWN):
        super().__init__(time, ["source", "message", "reason"])
        self.source = source
        self.message = message
        self.reason = reason

class MessageCreatedEvent(InstantEvent):
    """
    Event for when a message is created.
    """

    message: BaseMessage

    def __init__(self, time: float, message: BaseMessage):
        super().__init__(time, ["message"])
        self.message = message

class LTPSegmentCreatedEvent(InstantEvent):
    """
    Event for when a LTP Segment is created.
    """

    message: LTPSegment

    def __init__(self, time: float, message: LTPSegment):
        super().__init__(time, ["message"])
        self.message = message

class AttackMessageDroppedEvent(MessageDroppedEvent):
    """
    Event for when a message is dropped because of an attack.
    """

# Attack strategy type
# Takes an event, and decides whether or not to drop the message
AttackStrategy = Callable[[MessageSentEvent], bool]

def LinkDownAttackStrategy(
        links: set[tuple[SatID, SatID]],
        start_time: float = 0.0,
        probability: float = 1.0,
        seed: Optional[int] = None,
        message_filter: Optional[Callable[[BaseMessage], bool]] = None,
    ) -> AttackStrategy:
    """
    Create an attack strategy that takes down links after a given time.

    Parameters:
        links: Set of links to drop messages for.
        start_time: Time to start dropping messages from.
        probability: Probability of dropping a message.
        seed: Seed to use for the random number generator.
        message_filter: Optional filter function.
                        If provided, only messages that pass the filter will be dropped.

    Returns:
        Attack strategy.
    """

    rng = np.random.default_rng(seed)

    def strategy(event: MessageSentEvent) -> bool:
        """
        Attack strategy that drops messages when a link is down.

        Parameters:
            event: The message sent event.

        Returns:
            True if the message should be dropped, False otherwise.
        """
        return (
            event.time >= start_time
            and (event.source, event.destination) in links
            and rng.random() < probability
            and (message_filter is None or message_filter(event.message))
        )

    return strategy

def NodeDownAttackStrategy(
        nodes: set[SatID],
        start_time: float = 0.0,
        probability: float = 1.0,
        seed: Optional[int] = None,
        message_filter: Optional[Callable[[BaseMessage], bool]] = None,
    ) -> AttackStrategy:
    """
    Create an attack strategy that takes down nodes after a given time.

    Parameters:
        nodes: Set of nodes to drop messages for.
        start_time: Time to start dropping messages from.
        probability: Probability of dropping a message.
        seed: Seed to use for the random number generator.
        message_filter: Optional filter function.
                        If provided, only messages that pass the filter will be dropped.

    Returns:
        Attack strategy.
    """

    rng = np.random.default_rng(seed)

    def strategy(event: MessageSentEvent) -> bool:
        """
        Attack strategy that drops messages when a node is down.

        Parameters:
            event: The message sent event.

        Returns:
            True if the message should be dropped, False otherwise.
        """
        return (
            event.time >= start_time
            and (event.source in nodes or event.destination in nodes)
            and rng.random() < probability
            and (message_filter is None or message_filter(event.message))
        )

    return strategy

def MultipleAttackStrategy(strategies: list[AttackStrategy]) -> AttackStrategy:
    """
    Create an attack strategy that combines multiple attack strategies.

    Parameters:
        strategies: List of attack strategies to combine.

    Returns:
        Combined attack strategy.
    """

    def strategy(event: MessageSentEvent) -> bool:
        """
        Attack strategy that combines multiple attack strategies.

        Parameters:
            event: The message sent event.

        Returns:
            True if the message should be dropped, False otherwise.
        """
        return any(strategy(event) for strategy in strategies)

    return strategy

class RELIABLE_TRANSFER_TYPE(Enum):
    NONE = "NONE"
    LTP = "LTP"

class ReliableTransferConfig(ABC):
    """
    Abstract base class for configurations on how routing actor should send messages
    Either break them down into LTP segments or send the entire message unreliably
    """

    @abstractmethod
    def get_messages(self, time: int, source: SatID, destination: SatID, message: HybridDirectMessage) -> list[DirectMessage]:
        pass


class UnreliableConfig(ReliableTransferConfig):
    """
    Specific transfer config for unreliable transfer
    """
    def get_messages(self, time: int, source: SatID, destination: SatID, message: HybridDirectMessage) -> list[DirectMessage]:
        return [message]

class LTPConfig(ReliableTransferConfig):
    """
    https://ntrs.nasa.gov/api/citations/20240015952/downloads/IEEE_Aeroconf_2025%20final.pdf
    Says performance increases sharply until 8KB Segment Size and then marginally until 64KB.
    But 64KB is more on the unrealistic side (IP fragmentation and otherwise that we don't have).
    Whereas 9KB Jumbo Frames are possible. So we will stick to performance maximising Segment Size for
    regular DTNs at 8KB.
    Suggests 8KB-64KB Segment Size (especially on HDTNs like relay links).
    - Uses high bandwidth link with different MTU (we don't model MTU here)
    - Uses unrealistically high bandwidth though (not the goal of that paper) so need to find other more realistic source for this
    """
    _max_segment_size: int
    _underlying_message_uid_to_latest_ltp_session: dict[int, int] = defaultdict(int)
    def __init__(self, max_segment_size: int=1024 * 8):
        self._max_segment_size = max_segment_size

    def get_messages(self, time: int, source: SatID, destination: SatID, message: HybridDirectMessage) -> list[DirectMessage]:
        session = self._underlying_message_uid_to_latest_ltp_session[message.uid]
        self._underlying_message_uid_to_latest_ltp_session[message.uid] += 1

        red_data_size = message.reliable_data_size
        green_data_size = message.unreliable_data_size

        red_segments = []
        for offset in range(0, red_data_size, self._max_segment_size):
            size = min(self._max_segment_size, red_data_size - offset)
            is_last_red_segment = offset + size == red_data_size
            if is_last_red_segment:
                uids = [segment.uid for segment in red_segments]
                checkpoint_data = LTPCheckpointData(expected_uids=uids)
                last_segment = LTPDataSegment(time=time, source=source, destination=destination, underlying_message=message, data_type=LTPDataType.RED, size=size, is_checkpoint=True, checkpoint_data=checkpoint_data, session=session)
                last_segment.checkpoint_data.expected_uids.append(last_segment.uid)
                red_segments.append(last_segment)
            else:
                red_segments.append( LTPDataSegment(time=time, source=source, destination=destination, underlying_message=message, data_type=LTPDataType.RED, size=size, session=session) )

        green_segments = []
        for offset in range(0, green_data_size, self._max_segment_size):
            size = min(self._max_segment_size, green_data_size - offset)
            is_end_of_green_only_block = red_segments == [] and offset + size == green_data_size
            green_segments.append( LTPDataSegment(time=time, source=source, destination=destination, underlying_message=message, data_type=LTPDataType.GREEN, size=size, is_end_of_green_only_block=is_end_of_green_only_block, session=session) )

        return green_segments + red_segments
