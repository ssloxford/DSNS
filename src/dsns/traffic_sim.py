from typing import Optional
from dsns.message import DirectMessage, HybridDirectMessage, MessageCreatedEvent
from dsns.simulation import Actor
from dsns.events import Event
from dsns.helpers import SatID
import numpy as np
from math import floor

class GenerateMessagesEvent(Event):
    """
    Event to add more messages to the simulation queue.
    """

    def __init__(self, time: float):
        super().__init__(time, [])

class Sampler:
    """
    Base class for sampling strategies.
    """

    def _sample(self) -> float:
        """
        Sample a value from the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def sample_float(self, min: float = None, max: float = None) -> float:
        """
        Sample a value from the distribution.
        """
        value = self._sample()
        if min is not None and value < min:
            value = min
        if max is not None and value > max:
            value = max
        return value

    def sample_int(self, min: int = None, max: int = None) -> int:
        """
        Sample a value from the distribution and convert it to an integer.
        """
        value = int(self._sample())
        if min is not None and value < min:
            value = min
        if max is not None and value > max:
            value = max
        return value

class ConstantSampler(Sampler):
    """
    A simple constant sampler that returns a fixed value.
    """

    def __init__(self, value: float):
        self.value = value

    def _sample(self) -> float:
        return self.value

class UniformSampler(Sampler):
    """
    A simple uniform sampler that generates random numbers within a specified range.
    """

    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def _sample(self) -> float:
        return np.random.uniform(self.min, self.max)

class NormalSampler(Sampler):
    """
    A simple normal sampler that generates random numbers from a normal distribution.
    """

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def _sample(self) -> float:
        return np.random.normal(self.mean, self.std)

class ParetoSampler(Sampler):
    """
    A simple Pareto sampler that generates random numbers from a Pareto distribution.
    """

    def __init__(self, scale: float, shape: float):
        self.scale = scale
        self.shape = shape

    def _sample(self) -> float:
        return np.random.pareto(self.shape) * self.scale


class MultiPointToPointTrafficActor(Actor):
    """
    Actor that simulates point-to-point traffic between multiple pairs of nodes.
    """
    _cutoff: int
    def __init__(self, message_config: list[tuple[str, SatID, SatID, int, float]], update_interval: float = 60, reliable_messages: bool = False, cutoff: Optional[int] = None):
        """
        Parameters:
            message_config: List of tuples containing (name, source, destination, message_size, message_interval).
            update_interval: Time interval to generate new messages in seconds.
            reliable_messages: Whether to use reliable message delivery (i.e. generate HybridDirectMessages).
        """
        super().__init__()
        self.message_config: list[tuple[str, SatID, SatID, int, float]] = message_config
        self.update_interval: float = update_interval
        self.interval_count: int = 0
        self.reliable_messages: bool = reliable_messages
        self._cutoff = cutoff if cutoff is not None else np.inf

    def generate_events(self) -> list[Event]:
        """
        Generate the events for the current time interval.
        """
        events = []

        interval_min = self.interval_count * self.update_interval
        interval_max = (self.interval_count + 1) * self.update_interval

        for name, source, destination, message_size, message_interval in self.message_config:

            index_start = floor((self.interval_count * self.update_interval) / message_interval) - 1
            index_end = floor(((self.interval_count + 1) * self.update_interval) / message_interval) + 1
            while index_start * message_interval < interval_min:
                index_start += 1
            while index_end * message_interval >= interval_max:
                index_end -= 1
            for i in range(index_start, index_end + 1):
                if not self.reliable_messages:
                    message = DirectMessage(
                        source=source,
                        destination=destination,
                        data=f"{name}-{i}",
                        size=int(message_size),
                    )
                else:
                    message = HybridDirectMessage(
                        source=source,
                        destination=destination,
                        data=f"{name}-{i}",
                        size=int(message_size),
                        reliable_data_size=int(message_size),
                        unreliable_data_size=0,
                    )
                events.append(MessageCreatedEvent(
                    time=i * message_interval,
                    message=message,
                ))

        time = (self.interval_count + 1) * self.update_interval
        if time < self._cutoff:
            events.append(GenerateMessagesEvent(
                time=(self.interval_count + 1) * self.update_interval
            ))

        self.interval_count += 1

        return events

    def initialize(self) -> list[Event]:
        return self.generate_events()

    def handle_event(self, _, event: Event) -> list[Event]:
        if isinstance(event, GenerateMessagesEvent):
            return self.generate_events()
        return []


class RandomTrafficActor(Actor):
    """
    Actor that simulates random traffic across the network.
    """

    def __init__(
            self,
            satellites: list[SatID],
            message_interval: float = 1,
            message_size: Sampler = ConstantSampler(1e6 * 8),
            message_source: Sampler = UniformSampler(0, 100),
            message_destination: Sampler = UniformSampler(0, 100),
            update_interval: float = 60,
            reliable_messages: bool = False
        ):
        """
        Parameters:
            satellites: List of satellite IDs to use as sources and destinations.
            message_interval: Time interval between messages in seconds.
            message_size: Sampler for the size of the messages in bytes.
            message_source: Sampler for the source of the messages (e.g. satellite ID).
            message_destination: Sampler for the destination of the messages (e.g. satellite ID).
            update_interval: Time interval to generate new messages in seconds.
            reliable_messages: Whether to use reliable message delivery (i.e. generate HybridDirectMessages).
        """
        super().__init__()
        self.satellites: list[SatID] = satellites
        self.message_interval: float = message_interval
        self.message_size: Sampler = message_size
        self.message_source: Sampler = message_source
        self.message_destination: Sampler = message_destination
        self.update_interval: float = update_interval
        self.interval_count: int = 0
        self.reliable_messages: bool = reliable_messages

    def generate_events(self) -> list[Event]:
        """
        Generate the events for the current time interval.
        """
        events = []

        index_start = floor((self.interval_count * self.update_interval) / self.message_interval)
        index_end = floor(((self.interval_count + 1) * self.update_interval) / self.message_interval)
        for i in range(index_start, index_end):
            source = self.satellites[self.message_source.sample_int(0, len(self.satellites) - 1)]
            destination = self.satellites[self.message_destination.sample_int(0, len(self.satellites) - 1)]
            message_size = self.message_size.sample_float(min=1.0)
            data = f"Message-{i}"

            if not self.reliable_messages:
                message = DirectMessage(
                    source=source,
                    destination=destination,
                    data=data,
                    size=int(message_size),
                )
            else:
                message = HybridDirectMessage(
                    source=source,
                    destination=destination,
                    data=data,
                    size=int(message_size),
                    reliable_data_size=int(message_size),
                    unreliable_data_size=0,
                )
            events.append(MessageCreatedEvent(
                time=i * self.message_interval,
                message=message,
            ))

        events.append(GenerateMessagesEvent(
            time=(self.interval_count + 1) * self.update_interval
        ))

        self.interval_count += 1

        return events

    def initialize(self) -> list[Event]:
        return self.generate_events()

    def handle_event(self, _, event: Event) -> list[Event]:
        if isinstance(event, GenerateMessagesEvent):
            return self.generate_events()
        return []


class PointToPointTrafficActor(Actor):
    """
    Actor that simulates point-to-point traffic between two satellites.
    """

    def __init__(self, source: SatID, destination: SatID, message_interval: float, update_interval: float = 60):
        """
        Parameters:
            source: The satellite sending the messages.
            destination: The satellite receiving the messages.
            message_interval: Time interval between messages in seconds.
            update_interval: Time interval to generate new messages in seconds.
        """
        super().__init__()
        self.source: SatID = source
        self.destination: SatID = destination
        self.message_interval: float = message_interval
        self.update_interval: float = update_interval
        self.message_count: int = 0

    def generate_events(self) -> list[Event]:
        """
        Generate the events for the current time interval.
        """
        events = []
        num_messages = int(self.update_interval / self.message_interval)
        for i in range(self.message_count, self.message_count + num_messages):
            events.append(MessageCreatedEvent(
                time=i * self.message_interval,
                message=DirectMessage(
                    source=self.source,
                    destination=self.destination,
                    data=f"Message {i}: {self.source} -> {self.destination}",
                )
            ))

        events.append(GenerateMessagesEvent(
            time=(self.message_count + num_messages) * self.message_interval
        ))

        self.message_count += num_messages

        return events

    def initialize(self) -> list[Event]:
        return self.generate_events()

    def handle_event(self, _, event: Event) -> list[Event]:
        if isinstance(event, GenerateMessagesEvent):
            return self.generate_events()
        return []