from abc import ABC, abstractmethod
from typing import Optional, Callable

from dsns.message import BaseMessage

from .multiconstellation import MultiConstellation
from .events import Event, InstantEvent, EventQueue
from .helpers import SatID

from tqdm import tqdm


class DataProvider(ABC):
    """
    Base class for data providers (e.g. routing tables, etc.).
    """

    name: str # Name of the data provider
    kind: str # Kind of the data provider

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, mobility: MultiConstellation, time: float) -> list[Event]:
        """
        Initialize the data provider.

        Parameters:
            mobility: The mobility model to initialize the data provider with.
            time: The initial simulation time.

        Returns:
            List of events to add to the event queue.
        """
        pass

    @abstractmethod
    def update(self, mobility: MultiConstellation, time: float):
        """
        Update the data provider. This is called every time the clock updates.

        Parameters:
            mobility: The mobility model to update the data provider with.
            time: The current simulation time.
        """
        pass

    @abstractmethod
    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        """
        Handle an event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        pass


class RoutingDataProvider(DataProvider):
    """
    Base class for routing data providers.
    """

    kind: str = "routing"

    def __init__(self, get_next_hop_override: Optional[Callable[[BaseMessage, SatID, SatID], Optional[SatID]]] = None):
        super().__init__()
        self.get_next_hop_override: Optional[Callable[[BaseMessage, SatID, SatID], Optional[SatID]]] = get_next_hop_override

    def get_next_hop(self, source: SatID, destination: SatID, message: BaseMessage = None) -> Optional[SatID]:
        if self.get_next_hop_override:
            next_hop = self.get_next_hop_override(message, source, destination)
            if next_hop is not None:
                return next_hop
        return self._get_next_hop(source, destination)

    @abstractmethod
    def _get_next_hop(self, source: SatID, destination: SatID) -> Optional[SatID]:
        """
        Get the next hop for a given source and destination.

        Parameters:
            source: Source satellite.
            destination: Destination satellite.

        Returns:
            Next hop satellite, or None if no next hop is available.
        """
        pass

    @abstractmethod
    def get_distance(self, source: SatID, destination: SatID) -> Optional[float]:
        """
        Get the distance (time difference, latency) between a given source and destination.

        Parameters:
            source: Source satellite.
            destination: Destination satellite.

        Returns:
            Distance (time difference) between the source and destination, or None if no distance is available.
        """
        pass

    @abstractmethod
    def get_path_cost(self, source: SatID, destination: SatID) -> Optional[float]:
        pass 

    @abstractmethod
    def get_neighbors(self, source: SatID, max_distance: Optional[float] = None) -> list[SatID]:
        """
        Get the neighbors for a given source, up to a maximum distance (time difference).
        For look-ahead routing, this should provide the neighbors that are reachable within the given time difference,
        including those that are not currently connected.

        Parameters:
            source: Source satellite.
            max_distance: Maximum time difference (distance) to consider.

        Returns:
            List of neighbor satellites.
        """
        pass


class Actor(ABC):
    """
    Base class for simulation actors.

    Actors are objects that can be added to the simulation and can react to events, and generate their own events.
    """

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self) -> list[Event]:
        """
        Initialize the actor.

        Returns:
            List of events to add to the event queue.
        """
        pass

    @abstractmethod
    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        """
        Handle an event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        pass


class Simulation:
    """
    Main event-based simulator.
    """

    def __init__(self, mobility: MultiConstellation, actors: list[Actor] = [], logging_actors: list[Actor] = [], data_providers: list[DataProvider] = [], timestep: Optional[float] = None):
        """
        Setup the simulation.

        Parameters:
            mobility: The mobility model to use.
            actors: List of actors to add to the simulation.
            data_providers: List of data providers to add to the simulation.
            timestep: Timestep for mobility and data provider updates. If None, update on every event.
                      This can signficantly speed up the simulation, but may cause inaccuracies with some data providers.
        """
        self.mobility = mobility
        self.actors = actors
        self.logging_actors = logging_actors
        self.data_providers = data_providers
        self.event_queue = EventQueue()
        self.time = 0.0
        self.timestep = timestep

        self.initialized: bool = False

    def initialize(self, time: float = 0.0):
        """
        Initialize the simulation.

        Parameters:
            time: Initial time for the simulation.
        """

        self.time = time

        if self.initialized:
            raise Exception("Simulation already initialized.")

        self.mobility.update(self.time)

        for logging_actor in self.logging_actors:
            events = logging_actor.initialize()
            self.event_queue.add_events(events)

        for actor in self.actors:
            events = actor.initialize()
            self.event_queue.add_events(events)

        for data_provider in self.data_providers:
            events = data_provider.initialize(self.mobility, self.time)
            self.event_queue.add_events(events)

        self.initialized = True

    def step(self):
        """
        Run the simulation for a single step.
        """
        if not self.initialized:
            raise Exception("Simulation not initialized.")

        event = self.event_queue.get_next_event()
        if event is None:
            raise Exception("No events in event queue.")
        if event.time + (self.timestep or 0.0) < self.time:
            raise Exception(f"Time {event.time} for event {event} is before current time {self.time}.")
        if event.time > self.time:
            if self.timestep is None:
                self.time = event.time

                self.mobility.update(self.time)
                for data_provider in self.data_providers:
                    data_provider.update(self.mobility, self.time)
            else:
                # Floor event.time to the nearest timestep
                time_old = self.time
                self.time = (event.time // self.timestep) * self.timestep
                if self.time < event.time:
                    self.time += self.timestep

                if self.time > time_old:
                    self.mobility.update(self.time)
                    for data_provider in self.data_providers:
                        data_provider.update(self.mobility, self.time)

        self.step_event(event)

    def step_event(self, event: Event):
        """
        Process a single event.
        """
        # ensures logging correctly orders messages wrt instant response events
        for logging_actor in self.logging_actors:
            events = logging_actor.handle_event(self.mobility, event)
            for e in events:
                if isinstance(e, InstantEvent) and e.time <= self.time:
                    self.step_event(e)
                else:
                    self.event_queue.add_event(e)

        for actor in self.actors:
            events = actor.handle_event(self.mobility, event)
            for e in events:
                if isinstance(e, InstantEvent) and e.time <= self.time:
                    self.step_event(e)
                else:
                    self.event_queue.add_event(e)

        for data_provider in self.data_providers:
            events = data_provider.handle_event(self.mobility, event)
            for e in events:
                if isinstance(e, InstantEvent) and e.time <= self.time:
                    self.step_event(e)
                else:
                    self.event_queue.add_event(e)

    def run(self, duration: float, progress: bool = False):
        """
        Run the simulation for a given duration.

        Parameters:
            duration: Duration of the simulation.
        """
        if not self.initialized:
            raise Exception("Simulation not initialized.")

        with tqdm(total=duration, disable=not progress) as pbar:
            while not self.event_queue.empty() and self.time < duration:
                self.step()
                pbar.update(self.time - pbar.n)


class LoggingActor(Actor):
    """
    Actor that logs all events.
    """

    events: list[Event] = []
    event_filter: Callable[[Event], bool]

    def __init__(self, verbose: bool = False, event_filter: Callable[[Event], bool] = lambda _: True):
        """
        Setup the logging actor.

        Parameters:
            verbose: If True, print events to stdout.
            event_filter: Optional filter function to filter events.
        """
        super().__init__()

        self.verbose = verbose
        self.event_filter = event_filter

    def initialize(self) -> list[Event]:
        return []

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        if self.event_filter(event):
            self.events.append(event)

            if self.verbose:
                print(event)

        return []

    def get_events(self) -> list[Event]:
        return self.events


class FixedEventsActor(Actor):
    """
    Actor that generates a fixed set of events.
    """

    def __init__(self, events: list[Event]):
        """
        Setup the fixed events actor.

        Parameters:
            events: List of events to add to the queue.
        """
        super().__init__()

        self.events = events

    def initialize(self) -> list[Event]:
        return self.events

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        return []