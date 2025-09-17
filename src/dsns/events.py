from abc import ABC, abstractmethod
from typing import Optional

import queue

class Event(ABC):
    """
    Base class for events.
    """

    @abstractmethod
    def __init__(self, time: float, properties: list[str] = []):
        self.time = time
        self.properties = properties

    def __str__(self):
        properties = ", ".join([ f"{property_}={getattr(self, property_)}" for property_ in self.properties ])
        name = getattr(self, "name", self.__class__.__name__)
        return f"{self.time}: {name}({properties})"

    def __lt__(self, other):
        return self.time < other.time


class InstantEvent(Event):
    """
    Base class for instant events.
    These events have been generated at the current simulation time and can therefore be processed immediately.
    """

    def __init__(self, time: float, properties: list[str] = []):
        super().__init__(time, properties)


class EventQueue:
    """
    Priority queue for events.
    """

    def __init__(self):
        self.queue = queue.PriorityQueue()

    def empty(self) -> bool:
        """
        Check if the queue is empty.

        Returns:
            True if the queue is empty, False otherwise.
        """
        return self.queue.empty()

    def add_event(self, event: Event):
        """
        Add an event to the queue.

        Parameters:
            event: The event to add.
        """
        self.queue.put(event)

    def add_events(self, events: list[Event]):
        """
        Add multiple events to the queue.

        Parameters:
            events: The events to add.
        """
        for event in events:
            self.add_event(event)

    def get_next_event(self) -> Optional[Event]:
        """
        Remove and return the next event from the queue.

        Returns:
            The next event, or None if the queue is empty.
        """
        if self.queue.empty():
            return None
        return self.queue.get()

    def get_next_time(self) -> Optional[float]:
        """
        Return the time of the next event in the queue, without removing it.

        Returns:
            The time of the next event, or None if the queue is empty.
        """
        if self.queue.empty():
            return None
        return self.queue.queue[0].time


class LinkUpEvent(Event):
    """
    Event for a link between satellites going up.
    """

    def __init__(self, time: float, sat1: int, sat2: int):
        super().__init__(time, ["sat1", "sat2"])
        self.sat1 = sat1
        self.sat2 = sat2


class LinkDownEvent(Event):
    """
    Event for a link between satellites going down.
    """

    def __init__(self, time: float, sat1: int, sat2: int, sender_down: bool = True, receiver_down: bool = False):
        super().__init__(time, ["sat1", "sat2", "sender_down", "receiver_down"])
        self.sat1 = sat1
        self.sat2 = sat2
        self.sender_down = sender_down
        self.receiver_down = receiver_down
        # TODO: Add flag links going down because of mobility or for other reason (e.g. attack) - but do this after core functionality.



class RenderEvent(Event):
    """
    Event for rendering the current state of the simulation.
    """

    def __init__(self, time: float):
        super().__init__(time, [])