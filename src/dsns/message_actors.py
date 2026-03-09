from collections import defaultdict, deque
from typing import Optional, Callable, Union, List
import random

from networkit import Graph
from networkit.distance import Dijkstra
from networkit.dynamics import GraphDifference, GraphEventType
import networkit.graphtools
import numpy as np

try:
    from dsns.key_management import AttackStrategy
except ModuleNotFoundError:
    AttackStrategy = None
from dsns import message
from dsns.message import AttackMessageDroppedEvent, BaseMessage, BroadcastMessage, DirectMessage, DropReason, HybridDirectMessage, LTPSegment, LTPSegmentCreatedEvent, LTPSegmentReceivedEvent, LinkLossProbabilityUpdateEvent, LossConfig, MessageBroadcastDeliveredEvent, MessageCreatedEvent, MessageDeliveredEvent, MessageDroppedEvent, MessageLostEvent, MessageQueuedEvent, MessageReceivedEvent, MessageRerouteEvent, MessageSentEvent, ReliableTransferConfig, UnreliableConfig

from .multiconstellation import MultiConstellation
from .events import Event, LinkUpEvent, LinkDownEvent
from .helpers import SatID
from .simulation import Actor, RoutingDataProvider
from .solvers import GraphSolver, DijkstraSolver, BmsspSolver


class MessageRoutingActor(Actor):
    """
    Actor that implements message routing for each node in the simulation.
    This actor requires a RoutingDataProvider to be present in the data providers.

    Routing can be done using either IP-style best-effort routing, or DTN-style store-and-forward routing.
    For each of these, the store_and_forward parameter should be set and the appropriate RoutingDataProvider should be used.

    Note that this actor does not handle broadcast messages.
    """

    _routing: RoutingDataProvider
    _store_and_forward: bool
    _model_bandwidth: bool
    _attack_strategy: AttackStrategy
    _stored_messages: dict[tuple[SatID, SatID], list[DirectMessage]] = {}
    _delivered_messages: set[DirectMessage] = set() # Set of messages that have been delivered
    _delivered_message_uids: set[int] = set() # messages can differ on #hops if duplicated but still be same message

    _loss_config: LossConfig
    _reliable_transfer_config: ReliableTransferConfig

    _failed_links: set[tuple[SatID, SatID]]


    def __init__(self, routing: RoutingDataProvider, store_and_forward: bool = False, model_bandwidth = False, attack_strategy: Optional[AttackStrategy] = None, loss_config: LossConfig = None, reliable_transfer_config: ReliableTransferConfig = UnreliableConfig()):
        """
        Build the actor.

        Parameters:
            routing: The routing data provider to use.
            store_and_forward: Whether to use store-and-forward routing, instead of best-effort routing.
        """

        super().__init__()

        self._routing = routing
        self._store_and_forward = store_and_forward
        self._model_bandwidth = model_bandwidth
        self._attack_strategy = attack_strategy if attack_strategy is not None else lambda event: False

        self._loss_config = loss_config if loss_config else LossConfig()
        self._reliable_transfer_config = reliable_transfer_config

        self._failed_links = set()

    def initialize(self) -> list[Event]:
        """
        Initialize the actor.

        Returns:
            List of events to add to the event queue.
        """
        return []

    def __get_message_events(self, time: int, source: SatID, destination: SatID, message: BaseMessage):
        if isinstance(message, HybridDirectMessage):
            messages = self._reliable_transfer_config.get_messages(time=time, source=source, destination=destination, message=message)
            return [LTPSegmentCreatedEvent(time, message) for message in messages]
        else:
            if self._model_bandwidth:
                return [MessageQueuedEvent(time, source, destination, message)]
            else:
                return [MessageSentEvent(time, source, destination, message)]

    def _send_message(self, mobility: MultiConstellation, time: float, source: SatID, destination: SatID, message: DirectMessage) -> list[Event]:
        """
        Send a message along the next hop.

        Parameters:
            mobility: The mobility model to handle the event with.
            time: The current simulation time.
            source: Source satellite.
            destination: Destination satellite.
            message: Message to send.

        Returns:
            List of events to add to the event queue.
        """

        next_hop = self._routing.get_next_hop(source, destination, message)
        if next_hop is None:
            return [ MessageDroppedEvent(time, source, message, DropReason.NO_NEXT_HOP) ]
        else:
            # Check if connected
            reroute = getattr(message, "reroute", False)
            if (not mobility.has_link(source, next_hop) or (next_hop, source) in self._failed_links):
                if self._store_and_forward:
                    self._stored_messages.setdefault((source, next_hop), []).append(message)
                    return []
                else:
                    return [ MessageDroppedEvent(time, source, message, DropReason.INSUFFICIENT_BUFFER) ]
            else:
                if reroute:
                    message.reroute = False
                return self.__get_message_events(time, source, next_hop, message)

    def handle_message_sent_event(self, mobility: MultiConstellation, event: MessageSentEvent) -> list[Event]:
        """
        Handle a message sent event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        if self._attack_strategy(event):
            return [ AttackMessageDroppedEvent(event.time, event.source, event.message) ]
        else:
            if self._loss_config.is_message_lost(source=event.source, destination=event.destination, size=event.message.size):
                # TODO: maybe time should be random between sent and received times?
                return [ MessageLostEvent(event.time, event.source, event.destination, event.message) ]

            if isinstance(event.message, LTPSegment):
                message = event.message.copy()
                message.hops += 1
                return [ LTPSegmentReceivedEvent(event.time + mobility.get_delay(event.source, event.destination), event.source, event.destination, message) ]
            else:
                event.message.hops += 1
                return [ MessageReceivedEvent(event.time + mobility.get_delay(event.source, event.destination), event.source, event.destination, event.message) ]

    def handle_message_received_event(self, mobility: MultiConstellation, event: MessageReceivedEvent, message: DirectMessage) -> list[Event]:
        """
        Handle a message received event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        if message.dropped:
            return [ MessageDroppedEvent(event.time, event.destination, message, DropReason.INDICATED) ]
        elif event.destination == message.destination:
            if message not in self._delivered_messages and message.uid not in self._delivered_message_uids:
                self._delivered_messages.add(message)
                self._delivered_message_uids.add(message.uid)
                return [ MessageDeliveredEvent(event.time, event.destination, event.message) ]
            else:
                return [ MessageDroppedEvent(event.time, event.destination, message, DropReason.DUPLICATE) ]
        else:
            return self._send_message(mobility, event.time, event.destination, message.destination, message)

    def handle_message_created_event(self, mobility: MultiConstellation, event: MessageCreatedEvent, message: DirectMessage) -> list[Event]:
        """
        Handle a message created event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        event.message.time = event.time

        if event.message.source == message.destination:
            return [ MessageDeliveredEvent(event.time, message.destination, event.message) ]
        else:
            return self._send_message(mobility, event.time, event.message.source, message.destination, message)

    def handle_link_up_event(self, mobility: MultiConstellation, event: LinkUpEvent) -> list[Event]:
        """
        Handle a link up event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        if self._store_and_forward:
            events = []
            if (event.sat1, event.sat2) in self._stored_messages:
                messages = self._stored_messages[(event.sat1, event.sat2)].copy()
                del self._stored_messages[(event.sat1, event.sat2)]
                for message in messages:
                    events.extend( self.__get_message_events(event.time, event.sat1, event.sat2, message) )

            if (event.sat2, event.sat1) in self._stored_messages:
                messages = self._stored_messages[(event.sat2, event.sat1)].copy()
                del self._stored_messages[(event.sat2, event.sat1)]
                for message in messages:
                    events.extend( self.__get_message_events(event.time, event.sat2, event.sat1, message) )
            return events
        else:
            return []

    def handle_link_loss_probability_update_event(self, mobility: MultiConstellation, event: LinkLossProbabilityUpdateEvent):
        for update in event.updates:
            if update.reset_loss_probability:
                self._loss_config.reset_loss_probability_for_link(source=update.source, destination=update.destination)
            else:
                self._loss_config.set_loss_probability_for_link(source=update.source, destination=update.destination, loss_probability=update.loss_probability)
        return []

    def handle_message_reroute_event(self, mobility: MultiConstellation, event: MessageRerouteEvent, message: DirectMessage):
        return self._send_message(mobility, event.time, event.source, message.destination, message)

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        """
        Handle an event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        if isinstance(event, MessageSentEvent):
            if isinstance(event.message, DirectMessage):
                return self.handle_message_sent_event(mobility, event)
        elif isinstance(event, MessageReceivedEvent):
            if isinstance(event.message, DirectMessage):
                return self.handle_message_received_event(mobility, event, event.message)
        elif isinstance(event, MessageCreatedEvent):
            if isinstance(event.message, DirectMessage):
                return self.handle_message_created_event(mobility, event, event.message)
        elif isinstance(event, LinkDownEvent):
            self._failed_links.add((event.sat1, event.sat2))
            self._failed_links.add((event.sat2, event.sat1))
        elif isinstance(event, LinkUpEvent):
            if (event.sat1, event.sat2) in self._failed_links:
                self._failed_links.discard((event.sat1, event.sat2))
                self._failed_links.discard((event.sat2, event.sat1))
            return self.handle_link_up_event(mobility, event)
        elif isinstance(event, LinkLossProbabilityUpdateEvent):
            return self.handle_link_loss_probability_update_event(mobility, event)
        elif isinstance(event, MessageRerouteEvent):
            if isinstance(event.message, DirectMessage):
                return self.handle_message_reroute_event(mobility, event, event.message)
        return []


class MessageBroadcastActor(Actor):
    """
    Actor that provides message broadcasting for each node in the simulation.

    Routing can be done using either IP-style best-effort routing, or DTN-style store-and-forward routing.
    For each of these, the store_and_forward parameter should be set and the appropriate RoutingDataProvider should be used.

    Note that this actor does not handle unicast messages.
    """

    _routing: RoutingDataProvider
    _store_and_forward: bool
    _model_bandwidth: bool
    _stored_messages: dict[tuple[SatID, SatID], list[BroadcastMessage]] = {}
    _delivered_messages: dict[BroadcastMessage, set[SatID]] = {} # Set of satellites that have received each message

    _forwarding_strategy: str
    _gossip_neighbors: int
    _max_cache_time: None | float | dict[SatID, float] = None
    _attack_strategy: AttackStrategy
    _loss_config: LossConfig

    def __init__(
            self,
            routing: RoutingDataProvider,
            store_and_forward: bool = False,
            model_bandwidth = False,
            forwarding_strategy: str = "flooding",
            gossip_neighbors: int = 1,
            max_cache_time: None | float | dict[SatID, float] = None,
            attack_strategy: Optional[AttackStrategy] = None,
            loss_config: LossConfig = None,
        ):
        """
        Build the actor.

        Parameters:
            routing: The routing data provider to use.
            store_and_forward: Whether to use store-and-forward routing, instead of best-effort routing.
            forwarding_strategy: The broadcast forwarding strategy to use.
                                 "flooding" sends the message to all neighbors.
                                 "gossip" sends the message to a random subset of neighbors.
            gossip_neighbors: Number of neighbors to forward to when using the gossip forwarding strategy.
            max_cache_time: Maximum time to cache messages for, or None to disable caching.
                            If a dict is provided, it should map satellite IDs to cache times.
                            Should not be used when store_and_forward is False.
            attack_strategy: Attack strategy to use.
        """

        super().__init__()

        self._routing = routing
        self._store_and_forward = store_and_forward
        self._model_bandwidth = model_bandwidth
        if forwarding_strategy not in ["flooding", "gossip"]:
            raise ValueError("Invalid forwarding strategy")

        self._forwarding_strategy = forwarding_strategy
        self._gossip_neighbors = gossip_neighbors
        self._max_cache_time = max_cache_time
        self._attack_strategy = attack_strategy if attack_strategy is not None else lambda event: False
        self._loss_config = loss_config if loss_config else LossConfig()


    def initialize(self) -> list[Event]:
        """
        Initialize the actor.

        Returns:
            List of events to add to the event queue.
        """
        return []

    def __get_neighbors(self, source: SatID, previous: Optional[SatID] = None) -> list[SatID]:
        """
        Get the neighbors of a given source, taking into account the forwarding strategy.

        Parameters:
            source: Source satellite.
            previous: Previous satellite that sent the message, or None if this is the first hop.

        Returns:
            List of neighbors.
        """

        if isinstance(self._max_cache_time, dict):
            max_cache_time = self._max_cache_time[source]
        else:
            max_cache_time = self._max_cache_time

        neighbors = self._routing.get_neighbors(source, max_cache_time)
        if previous is not None and previous in neighbors:
            neighbors.remove(previous)

        if self._forwarding_strategy == "flooding":
            return neighbors
        elif self._forwarding_strategy == "gossip":
            return np.random.choice(neighbors, size=self._gossip_neighbors, replace=False).tolist()
        else:
            raise ValueError("Invalid forwarding strategy")

    def __get_message_events(self, time: float, source: SatID, destination: SatID, message: BroadcastMessage) -> list[Event]:
        if self._model_bandwidth:
            return [MessageQueuedEvent(time, source, destination, message)]
        else:
            return [MessageSentEvent(time, source, destination, message)]

    def __forward_message(self, mobility: MultiConstellation, time: float, source: SatID, message: BroadcastMessage, previous: Optional[SatID] = None) -> list[Event]:
        """
        Forward a message to all neighbors.

        Parameters:
            mobility: The mobility model to handle the event with.
            time: The current simulation time.
            source: Source satellite.
            message: Message to send.
            previous: Previous satellite that sent the message, or None if this is the first hop.

        Returns:
            List of events to add to the event queue.
        """
        events = []

        neighbors = self.__get_neighbors(source, previous)

        for neighbor in neighbors:
            # Check if connected
            if not mobility.has_link(source, neighbor):
                if self._store_and_forward:
                    self._stored_messages.setdefault((source, neighbor), []).append(message)
                else:
                    events.append(MessageDroppedEvent(time, source, message, DropReason.INSUFFICIENT_BUFFER))
            else:
                # Send message
                events.extend(self.__get_message_events(time=time, source=source, destination=neighbor, message=message))

        return events

    def handle_message_sent_event(self, mobility: MultiConstellation, event: MessageSentEvent, message: BroadcastMessage) -> list[Event]:
        """
        Handle a message sent event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.
            message: The message to send.

        Returns:
            List of events to add to the event queue.
        """
        if self._attack_strategy(event):
            return [ AttackMessageDroppedEvent(event.time, event.source, event.message) ]
        else:
            if self._loss_config.is_message_lost(source=event.source, destination=event.destination, size=event.message.size):
                return [ MessageLostEvent(event.time, event.source, event.destination, event.message) ]
            else:
                # Copy message to avoid globally modifying the hop count
                message_copy = message.copy()
                message_copy.hops += 1
                return [ MessageReceivedEvent(event.time + mobility.get_delay(event.source, event.destination), event.source, event.destination, event.message) ]

    def handle_message_received_event(self, mobility: MultiConstellation, event: MessageReceivedEvent, message: BroadcastMessage) -> list[Event]:
        """
        Handle a message received event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.
            message: The received message.

        Returns:
            List of events to add to the event queue.
        """

        # If the message has been dropped, ignore it
        if message.dropped:
            return [ MessageDroppedEvent(event.time, event.destination, message, DropReason.INDICATED) ]

        # If the message has already been delivered to this satellite, ignore it
        if event.destination in self._delivered_messages.get(message, set()):
            return []
        events: list[Event] = [ MessageDeliveredEvent(event.time, event.destination, event.message) ]

        # Mark the message as delivered to this satellite
        self._delivered_messages.setdefault(message, set()).add(event.destination)

        # Forward the message to all neighbors (except the previous satellite)
        events.extend(self.__forward_message(mobility, event.time, event.destination, message, event.source))

        return events

    def handle_message_delivered_event(self, mobility: MultiConstellation, event: MessageDeliveredEvent, message: BroadcastMessage) -> list[Event]:
        """
        Handle a message delivered event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.
            message: The received message.

        Returns:
            List of events to add to the event queue.
        """
        # Mark the message as delivered to this satellite
        self._delivered_messages.setdefault(message, set()).add(event.destination)

        # If the message has been delivered to all satellites, generate a message delivered event
        if len(self._delivered_messages.get(message, set())) == len(mobility.satellites):
            return [ MessageBroadcastDeliveredEvent(event.time, message) ]
        else:
            return []

    def handle_message_created_event(self, mobility: MultiConstellation, event: MessageCreatedEvent, message: BroadcastMessage) -> list[Event]:
        """
        Handle a message created event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.
            message: The created message.

        Returns:
            List of events to add to the event queue.
        """
        event.message.time = event.time

        # Mark the message as delivered to the source node
        self._delivered_messages.setdefault(message, set()).add(event.message.source)
        events: list[Event] = [ MessageDeliveredEvent(event.time, event.message.source, event.message) ]

        # Forward the message to all neighbors
        events.extend(self.__forward_message(mobility, event.time, event.message.source, message))

        return events

    def handle_link_up_event(self, mobility: MultiConstellation, event: LinkUpEvent) -> list[Event]:
        """
        Handle a link up event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        if self._store_and_forward:
            events = []

            if (event.sat1, event.sat2) in self._stored_messages:
                messages = self._stored_messages[(event.sat1, event.sat2)]
                del self._stored_messages[(event.sat1, event.sat2)]
                for message in messages:
                    events.extend(self.__get_message_events(time=event.time, source=event.sat1, destination=event.sat2, message=message))

            if (event.sat2, event.sat1) in self._stored_messages:
                messages = self._stored_messages[(event.sat2, event.sat1)]
                del self._stored_messages[(event.sat2, event.sat1)]
                for message in messages:
                    events.extend(self.__get_message_events(time=event.time, source=event.sat2, destination=event.sat1, message=message))

            return events
        else:
            return []

    def handle_link_loss_probability_update_event(self, mobility: MultiConstellation, event: LinkLossProbabilityUpdateEvent):
        for update in event.updates:
            if update.reset_loss_probability:
                self._loss_config.reset_loss_probability_for_link(source=update.source, destination=update.destination)
            else:
                self._loss_config.set_loss_probability_for_link(source=update.source, destination=update.destination, loss_probability=update.loss_probability)
        return []

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        """
        Handle an event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        if isinstance(event, MessageSentEvent):
            if isinstance(event.message, BroadcastMessage):
                return self.handle_message_sent_event(mobility, event, event.message)
        elif isinstance(event, MessageReceivedEvent):
            if isinstance(event.message, BroadcastMessage):
                return self.handle_message_received_event(mobility, event, event.message)
        elif isinstance(event, MessageCreatedEvent):
            if isinstance(event.message, BroadcastMessage):
                return self.handle_message_created_event(mobility, event, event.message)
        elif isinstance(event, MessageDeliveredEvent):
            if isinstance(event.message, BroadcastMessage):
                return self.handle_message_delivered_event(mobility, event, event.message)
        elif isinstance(event, LinkUpEvent):
            return self.handle_link_up_event(mobility, event)
        elif isinstance(event, LinkLossProbabilityUpdateEvent):
            return self.handle_link_loss_probability_update_event(mobility, event)
        return []


class BestEffortRoutingDataProvider(RoutingDataProvider):
    """
    Routing data provider that implements best effort routing.
    """

    time: float # Current time in seconds since the epoch
    __graph: Graph # Graph of the network
    __sat_to_node_id: dict[SatID, int] # Mapping from satellite IDs to node IDs
    __node_to_sat_id: dict[int, SatID] # Mapping from node IDs to satellite IDs

    def __init__(self, get_next_hop_override: Optional[Callable[[BaseMessage, SatID, SatID], Optional[SatID]]] = None):
        super().__init__(get_next_hop_override=get_next_hop_override)

    def initialize(self, mobility: MultiConstellation, time: float) -> list[Event]:
        """
        Initialize the data provider.

        Parameters:
            mobility: The mobility model to initialize the data provider with.
            time: The initial simulation time.

        Returns:
            List of events to add to the event queue.
        """
        self.time = time
        self.__graph = Graph(weighted=True)
        self.__sat_to_node_id = {}
        self.__node_to_sat_id = {}
        for sat in mobility.satellites:
            node_id = self.__graph.addNode()
            self.__sat_to_node_id[sat.sat_id] = node_id
            self.__node_to_sat_id[node_id] = sat.sat_id

        self.update(mobility, 0.0)

        return []

    def update(self, mobility: MultiConstellation, time: float):
        """
        Update the data provider. This is called every time the clock updates.

        Parameters:
            mobility: The mobility model to update the data provider with.
            time: The current simulation time.
        """
        self.__graph.removeAllEdges()
        for sat1, sat2 in mobility.links:
            self.__graph.addEdge(self.__sat_to_node_id[sat1], self.__sat_to_node_id[sat2], mobility.get_delay(sat1, sat2))

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        return []

    def _get_next_hop(self, source: SatID, destination: SatID) -> Optional[SatID]:
        """
        Get the next hop for a given source and destination.

        Parameters:
            source: Source satellite.
            destination: Destination satellite.

        Returns:
            The next hop, or None if no route exists.
        """
        if source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return None

        dijkstra = Dijkstra(self.__graph, self.__sat_to_node_id[source], True, True)
        dijkstra.run()

        path = dijkstra.getPath(self.__sat_to_node_id[destination])

        if len(path) < 2:
            return None

        next_hop = self.__node_to_sat_id[path[1]]

        return next_hop

    def get_distance(self, source: SatID, destination: SatID) -> Optional[float]:
        """
        Get the distance (time difference, latency) between a given source and destination.

        Parameters:
            source: Source satellite.
            destination: Destination satellite.

        Returns:
            Distance (time difference) between the source and destination, or None if no distance is available.
        """
        if source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return None

        dijkstra = Dijkstra(self.__graph, self.__sat_to_node_id[source], True, True)
        dijkstra.run()

        path = dijkstra.getPath(self.__sat_to_node_id[destination])
        if len(path) < 2:
            return None

        return dijkstra.distance(self.__sat_to_node_id[destination])

    def get_path_cost(self, source, destination):
        if source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return None

        dijkstra = Dijkstra(self.__graph, self.__sat_to_node_id[source], True, True)
        dijkstra.run()

        return dijkstra.distance(self.__sat_to_node_id[destination])

    def get_neighbors(self, source: SatID, max_distance: Optional[float] = None) -> list[SatID]:
        """
        Get the neighbors of a given source.

        Parameters:
            source: Source satellite.
            max_distance: Maximum distance to neighbors.

        Returns:
            List of neighbors.
        """
        if source not in self.__sat_to_node_id:
            return []

        node_id = self.__sat_to_node_id[source]
        iter_neighbors = self.__graph.iterNeighborsWeights(node_id)

        return [ self.__node_to_sat_id[neighbor] for (neighbor, weight) in iter_neighbors if max_distance is None or weight <= max_distance ]


class UpdateConnectivityEvent(Event):
    """
    Event for updating the connectivity graph.
    """

    def __init__(self, time: float):
        super().__init__(time)


class LookaheadRoutingDataProvider(RoutingDataProvider):
    """
    Data provider that keeps track of a connectivity graph over time, for use in store-and-forward routing.
    Multiple timesteps are measured at once, to provide information on the next time a link will be up (within a given time bound).

    This also generates events for when links go up and down.
    """

    resolution: float # Time between updates of the connectivity graph
    num_steps: int # Number of timesteps of lookahead provided
    __timestep: float # Timestep at which connectivity updates are scheduled

    time: float # Current time in seconds since the epoch
    __graphs: deque[Graph] = deque() # Graphs for each timestep
    __graph_index_offset: int = 0 # Offset for graph indices
    __sat_to_node_id: dict[SatID, int] # Mapping from satellite IDs to node IDs
    __node_to_sat_id: dict[int, SatID] # Mapping from node IDs to satellite IDs

    def __init__(self, resolution: float, num_steps: int, get_next_hop_override: Optional[Callable[[BaseMessage, SatID, SatID], Optional[SatID]]] = None):
        """
        Initialise the connectivity system.

        Parameters:
            resolution: Time between updates of the connectivity graph.
            num_steps: Number of timesteps of lookahead to store at once.
        """

        self.resolution = resolution
        self.num_steps = num_steps
        self.__timestep = resolution * num_steps

        super().__init__(get_next_hop_override=get_next_hop_override)

    def initialize(self, mobility: MultiConstellation, time: float) -> list[Event]:
        """
        Initialize the connectivity system.

        Parameters:
            mobility: The mobility model to initialize the data provider with.
            time: The initial simulation time.

        Returns:
            List of events to add to the event queue.
        """
        self.time = time

        graph_base = Graph(weighted=True, directed=False)
        self.__sat_to_node_id = {}
        self.__node_to_sat_id = {}
        for sat in mobility.satellites:
            node_id = graph_base.addNode()
            self.__sat_to_node_id[sat.sat_id] = node_id
            self.__node_to_sat_id[node_id] = sat.sat_id

        for _ in range(self.num_steps * 2):
            self.__graphs.append(networkit.graphtools.copyNodes(graph_base))

        events = self.update_connectivity(mobility, initial=True)
        events.append(UpdateConnectivityEvent(self.__timestep))

        return events

    def update(self, mobility: MultiConstellation, time: float):
        """
        Update the connectivity system for a given timestep.

        Parameters:
            mobility: The mobility model to update the data provider with.
            time: The current simulation time.
        """
        self.time = time

    def update_connectivity(self, mobility: MultiConstellation, initial: bool = False) -> list[Event]:
        """
        Update the connectivity graphs.

        Parameters:
            mobility: The mobility model to update the connectivity graph with.
            initial: Whether this is the initial update.

        Returns:
            List of events to add to the event queue.
        """

        events = []

        # Shift everything down by num_steps
        if not initial:
            self.__graph_index_offset += self.num_steps

            self.__graphs.rotate(-self.num_steps)
            for i in range(self.num_steps):
                self.__graphs[i + self.num_steps].removeAllEdges()

        original_time = mobility.time

        loop_range = range(self.num_steps * 2) if initial else range(self.num_steps, self.num_steps * 2)
        for i in loop_range:
            time = (self.__graph_index_offset + i) * self.resolution
            mobility.update(time)

            for sat1, sat2 in mobility.links:
                delay = mobility.get_delay(sat1, sat2)
                self.__graphs[i].setWeight(self.__sat_to_node_id[sat1], self.__sat_to_node_id[sat2], delay)

            # Compare the current graph with the previous graphs, updating the previous graphs as necessary
            # Also generate link up/down events
            for j in range(max(i - self.num_steps, 0), i):
                time_j = (self.__graph_index_offset + j) * self.resolution
                graph_difference = GraphDifference(self.__graphs[j], self.__graphs[i])
                graph_difference.run()

                for graph_edit in graph_difference.getEdits():
                    if graph_edit.type == GraphEventType.EDGE_ADDITION:
                        if j == i - 1:
                            events.append(LinkUpEvent(time, self.__node_to_sat_id[graph_edit.u], self.__node_to_sat_id[graph_edit.v]))
                        # Does an edge already exist? If so, only update the weight if it's lower
                        if self.__graphs[j].hasEdge(graph_edit.u, graph_edit.v):
                            weight_j = self.__graphs[j].weight(graph_edit.u, graph_edit.v)
                            weight_new = time - time_j + graph_edit.w
                            if weight_new < weight_j:
                                self.__graphs[j].setWeight(graph_edit.u, graph_edit.v, time - time_j + graph_edit.w)
                        else:
                            self.__graphs[j].setWeight(graph_edit.u, graph_edit.v, time - time_j + graph_edit.w)
                    elif graph_edit.type == GraphEventType.EDGE_REMOVAL:
                        if j == i - 1:
                            events.append(LinkDownEvent(time, self.__node_to_sat_id[graph_edit.u], self.__node_to_sat_id[graph_edit.v]))
                    elif graph_edit.type == GraphEventType.EDGE_WEIGHT_UPDATE:
                        weight_j = self.__graphs[j].weight(graph_edit.u, graph_edit.v)
                        weight_new = time - time_j + graph_edit.w
                        if weight_j == 0 or weight_new < weight_j:
                            self.__graphs[j].setWeight(graph_edit.u, graph_edit.v, weight_new)
                    else:
                        raise ValueError(f"Unexpected graph edit type: {graph_edit.type}")

        mobility.update(original_time)
        return events

    def handle_update_connectivity_event(self, mobility: MultiConstellation, event: UpdateConnectivityEvent) -> list[Event]:
        """
        Handle an update connectivity event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        events = self.update_connectivity(mobility)
        events.append(UpdateConnectivityEvent(event.time + self.__timestep))

        return events

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        """
        Handle an event.

        Parameters:
            mobility: The mobility model to handle the event with.
            event: The event to handle.

        Returns:
            List of events to add to the event queue.
        """
        if isinstance(event, UpdateConnectivityEvent):
            return self.handle_update_connectivity_event(mobility, event)
        else:
            return []

    def _get_next_hop(self, source: SatID, destination: SatID) -> Optional[SatID]:
        """
        Get the next hop for a given source and destination.

        Parameters:
            source: Source satellite.
            destination: Destination satellite.

        Returns:
            Next hop satellite, or None if no next hop is available.
        """
        if source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return None

        graph_index = (int(self.time // self.resolution) - self.__graph_index_offset) + 1
        # TODO investigate whether the +1 causes issues with sparse networks

        dijkstra = Dijkstra(self.__graphs[graph_index], self.__sat_to_node_id[source], True, True)
        dijkstra.run()

        path = dijkstra.getPath(self.__sat_to_node_id[destination])
        if len(path) < 2:
            return None

        next_hop = self.__node_to_sat_id[path[1]]

        return next_hop

    def get_path_cost(self, source: SatID, destination: SatID):
        if source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return None

        graph_index = (int(self.time // self.resolution) - self.__graph_index_offset) + 1

        dijkstra = Dijkstra(self.__graphs[graph_index], self.__sat_to_node_id[source], True, True)
        dijkstra.run()

        return dijkstra.distance(self.__sat_to_node_id[destination])


    def get_distance(self, source: SatID, destination: SatID) -> Optional[float]:
        """
        Get the distance (time difference, latency) between a given source and destination.

        Parameters:
            source: Source satellite.
            destination: Destination satellite.

        Returns:
            Distance (time difference) between the source and destination, or None if no distance is available.
        """
        if source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return None

        graph_index = (int(self.time // self.resolution) - self.__graph_index_offset) + 1

        dijkstra = Dijkstra(self.__graphs[graph_index], self.__sat_to_node_id[source], True, True)
        dijkstra.run()

        path = dijkstra.getPath(self.__sat_to_node_id[destination])
        if len(path) < 2:
            return None

        return dijkstra.distance(self.__sat_to_node_id[destination])

    def get_neighbors(self, source: SatID, max_distance: Optional[float] = None) -> list[SatID]:
        """
        Get the neighbors of a given source.

        Parameters:
            source: Source satellite.
            max_distance: Maximum distance to neighbors.

        Returns:
            List of neighbors.
        """
        if source not in self.__sat_to_node_id:
            return []

        graph_index = (int(self.time // self.resolution) - self.__graph_index_offset) + 1
        # TODO investigate whether the +1 causes issues with sparse networks

        node_id = self.__sat_to_node_id[source]
        iter_neighbors = self.__graphs[graph_index].iterNeighborsWeights(node_id)

        return [ self.__node_to_sat_id[neighbor] for (neighbor, weight) in iter_neighbors if max_distance is None or weight <= max_distance ]


def novel_cost(mobility: MultiConstellation) -> dict[tuple[SatID, SatID], float]:
    costs = dict()
    MAX_DOPPLER = 10e9
    for u, v in mobility.links:
        pos_u, vel_u = mobility.satellites.by_id(u).position, mobility.satellites.by_id(u).velocity
        pos_v, vel_v = mobility.satellites.by_id(v).position, mobility.satellites.by_id(v).velocity
        cost1 = mobility.get_delay(u, v) + get_doppler_shift(pos_u, vel_u, pos_v, vel_v, ISL_FREQ) / MAX_DOPPLER
        cost2 = mobility.get_delay(v, u) + get_doppler_shift(pos_v, vel_v, pos_u, vel_u, ISL_FREQ) / MAX_DOPPLER
        costs[(u, v)] = cost1
        costs[(v, u)] = cost2
    return costs

class GlobalRoutingDataProvider(RoutingDataProvider):
    solver: GraphSolver
    update_interval: float
    _last_update_time = -float("inf")
    _advanced_cost: bool
    _failed_links: set[tuple[SatID, SatID]] = set()
    _gs_ids: set[SatID] = set()
    _gs_neighbors: dict[SatID, list[tuple[SatID, float]]] = {}
    mobility: Optional[MultiConstellation] = None


    def __init__(self, get_next_hop_override: Optional[Callable[[BaseMessage, SatID, SatID], Optional[SatID]]] = None, solver: Union[GraphSolver, type[GraphSolver]] = BmsspSolver, update_interval: float = 15, advanced_cost=False):
        super().__init__(get_next_hop_override=get_next_hop_override)
        self.solver = solver() if isinstance(solver, type) else solver
        self.update_interval = update_interval
        self._advanced_cost = advanced_cost

    def initialize(self, mobility: MultiConstellation, time: float) -> list[Event]:
        self.update(mobility, time)
        self.mobility = mobility
        return []

    def update(self, mobility: MultiConstellation, time: float, force: bool=False):
        if force or time - self._last_update_time >= self.update_interval:
            if self._advanced_cost:
                costs = novel_cost(mobility)
                self.solver.update(len(mobility.satellites), costs)
            else:
                self.solver.update(mobility)
            self._last_update_time = time

            # self._gs_ids = set()
            # self._gs_neighbors = defaultdict(list)
            # edges_to_remove = set()

            # for sat in mobility.satellites:
            #     if sat.constellation_name == "ground":
            #         self._gs_ids.add(sat.sat_id)

            # if hasattr(self.solver, "graph"):
            #     g = self.solver.graph
            #     for u in self._gs_ids:
            #         if u in g:
            #             for v, w in g[u].items():
            #                 self._gs_neighbors[u].append((v, w))
            #                 edges_to_remove.add((u, v))

            self.solver.remove_edges(self._failed_links)
            # self.solver.remove_edges(self._failed_links | edges_to_remove)

    def _get_best_neighbor(self, neighbors: list[tuple[SatID, float]], target: SatID) -> tuple[SatID, float]:
        if not neighbors:
             return None
        
        if not self.mobility:
             return neighbors[0]

        try:
            target_pos = self.mobility.satellites.by_id(target).position
            
            best_neighbor = neighbors[0]
            min_dist = float('inf')

            for neighbor in neighbors:
                sat_id = neighbor[0]
                pos = self.mobility.satellites.by_id(sat_id).position
                dist = np.linalg.norm(pos - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = neighbor
            return best_neighbor
        except Exception:
            return neighbors[0]

    def _get_next_hop(self, source: SatID, destination: SatID) -> Optional[SatID]:
        if source == destination:
            return None

        # source_new = (source, 0.0)
        # if source in self._gs_ids:
        #     neighbors = self._gs_neighbors.get(source, [])
        #     if len(neighbors) == 0:
        #         return None
        #     best = self._get_best_neighbor(neighbors, destination)
        #     return best[0]

        # destination_new = (destination, 0.0)
        # if destination in self._gs_ids:
        #     neighbors = self._gs_neighbors.get(destination, [])
        #     if len(neighbors) == 0:
        #         return None
        #     destination_new = self._get_best_neighbor(neighbors, source)

        # if source == destination_new[0]:
        #     return destination
        try:
            path = self.solver.get_path(source, destination)
            # path = self.solver.get_path(source_new[0], destination_new[0])
            if path and len(path) > 1:
                return path[1]
            else:
                return None
        except Exception:
            return None

    def get_path_cost(self, source, destination):
        if source == destination:
            return 0

        # source_new = (source, 0.0)

        # if source in self._gs_ids:
        #     neighbors = self._gs_neighbors.get(source, [])
        #     if len(neighbors) == 0:
        #         return None
        #     source_new = self._get_best_neighbor(neighbors, destination)

        # destination_new = (destination, 0.0)
        # if destination in self._gs_ids:
        #     neighbors = self._gs_neighbors.get(destination, [])
        #     if len(neighbors) == 0:
        #         return None
        #     destination_new = self._get_best_neighbor(neighbors, source)

        try:
            return self.solver.get_path_cost(source, destination)
            # return self.solver.get_path_cost(source_new[0], destination_new[0]) + source_new[1] + destination_new[1]
        except Exception:
            return float("inf")
        
    def get_distance(self, source, destination):
        cost = self.get_path_cost(source, destination)
        return cost if cost != float("inf") else None
    
    def get_neighbors(self, source, max_distance = None):
        if hasattr(self.solver, 'graph') and self.solver.graph:
            g = self.solver.graph
            if 0 <= source < len(g):
                neighbors = []
                for v, w in g[source].items():
                    if max_distance is None or w <= max_distance:
                        neighbors.append(v)
                return neighbors
        return []

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        if isinstance(event, LinkDownEvent):
            self._failed_links.add((event.sat1, event.sat2))
            self._failed_links.add((event.sat2, event.sat1))
            self.update(mobility, event.time, force=True)
        elif isinstance(event, LinkUpEvent):
            if (event.sat1, event.sat2) in self._failed_links:
                self._failed_links.discard((event.sat1, event.sat2))
                self._failed_links.discard((event.sat2, event.sat1))
                self.update(mobility, event.time, force=True)
        return []

class GlobalRoutingActor(MessageRoutingActor):
    def __init__(self,  provider: Optional[RoutingDataProvider], solver: Union[GraphSolver, type[GraphSolver]] = BmsspSolver, update_interval: float = 15, advanced_cost=False, store_and_forward = False, model_bandwidth=False, attack_strategy = None, loss_config = None, reliable_transfer_config = UnreliableConfig()):
        if not provider:
            provider = GlobalRoutingDataProvider(solver=solver, update_interval=update_interval, advanced_cost=advanced_cost)
        super().__init__(provider, store_and_forward, model_bandwidth, attack_strategy, loss_config, reliable_transfer_config)


class SourceRoutingDataProvider(RoutingDataProvider):
    solver: GraphSolver
    update_interval: float
    _last_update_time = -float("inf")
    _gs_ids: set[SatID] = set()
    _gs_neighbors: dict[SatID, list[tuple[SatID, float]]] = {}
    mobility: Optional[MultiConstellation] = None

    def __init__(self, get_next_hop_override: Optional[Callable[[BaseMessage, SatID, SatID], Optional[SatID]]] = None, solver: Union[GraphSolver, type[GraphSolver]] = BmsspSolver, update_interval: float = 15):
        super().__init__(get_next_hop_override=get_next_hop_override)
        self.solver = solver() if isinstance(solver, type) else solver
        self.update_interval = update_interval

    def initialize(self, mobility: MultiConstellation, time: float) -> list[Event]:
        self.update(mobility, time)
        return []

    def update(self, mobility: MultiConstellation, time: float):
        self.mobility = mobility
        if  time - self._last_update_time >= self.update_interval:
            self.solver.update(mobility)
            self._last_update_time = time

        
            # self._gs_ids = set()
            # self._gs_neighbors = defaultdict(list)
            # edges_to_remove = set()

            # for sat in mobility.satellites:
            #     if sat.constellation_name == "ground":
            #         self._gs_ids.add(sat.sat_id)

            # if hasattr(self.solver, 'graph'):
            #     g = self.solver.graph
            #     for u in self._gs_ids:
            #         if u in g:
            #             for v, w in g[u].items():
            #                 self._gs_neighbors[u].append((v, w))
            #                 edges_to_remove.add((u, v))
            
            # self.solver.remove_edges(edges_to_remove)

    def _get_best_neighbor(self, source: SatID, target: SatID) -> tuple[SatID, float]:
        if not self.mobility:
            return None

        neighbors = []
        for u, v in self.mobility.links:
            if u == source:
                neighbors.append((v, self.mobility.get_delay(u, v)))
            elif v == source:
                neighbors.append((u, self.mobility.get_delay(v, u)))

        if not neighbors:
             return None

        try:
            target_pos = self.mobility.satellites.by_id(target).position
            
            best_neighbor = neighbors[0]
            min_dist = float('inf')

            for neighbor in neighbors:
                sat_id = neighbor[0]
                pos = self.mobility.satellites.by_id(sat_id).position
                dist = np.linalg.norm(pos - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = neighbor
            return best_neighbor
        except Exception:
            return neighbors[0]

    def _get_path(self, source: SatID, destination: SatID) -> list[SatID]:
        if source == destination:
            return [source]

        # source_new = (source, 0.0)
        # if source in self._gs_ids:
        #     neighbors = self._gs_neighbors.get(source, [])
        #     if len(neighbors) == 0:
        #         return []
        #     # source_new = self._get_best_neighbor(neighbors, destination)
        #     source_new = self._get_best_neighbor(source, destination)

        # destination_new = (destination, 0.0)
        # if destination in self._gs_ids:
        #     neighbors = self._gs_neighbors.get(destination, [])
        #     if len(neighbors) == 0:
        #         return []
        #     # destination_new = self._get_best_neighbor(neighbors, source)
        #     destination_new = self._get_best_neighbor(destination, source)

        try:
            # path = self.solver.get_path(source_new[0], destination_new[0])
            return self.solver.get_path(source, destination)
            # return [source] + path + [destination]
        except Exception:
            return []

    def _get_next_hop(self, source: SatID, destination: SatID) -> Optional[SatID]:
        if source == destination:
            return None

        try:
            path = self._get_path(source, destination)
            if path and len(path) > 1:
                return path[1]
            else:
                return None
        except Exception:
            return None
    
    def get_next_hop(self, source: SatID, destination: SatID, message: BaseMessage = None) -> Optional[SatID]:
        if source == destination:
            return None

        if self.get_next_hop_override:
            next_hop = self.get_next_hop_override(message, source, destination)
            if next_hop is not None:
                return next_hop
        
        if message is None:
            return self._get_next_hop(source, destination)
        
        route = getattr(message, "route", None)
        index = getattr(message, "index", -1)

        if route is None:
            try:
                route = tuple(self._get_path(source, destination))
            except Exception:
                return None
            if len(route) < 2:
                return None
            message.route = route
            message.index = 0
            index = 0

        if index < len(route):
            if index + 1 < len(route):
                next_hop = route[index + 1]
                message.index += 1
                return next_hop
            else:
                return None
        return None

    def get_path_cost(self, source, destination):
        if source == destination:
            return 0

        # source_new = (source, 0.0)

        # if source in self._gs_ids:
        #     neighbors = self._gs_neighbors.get(source, [])
        #     if len(neighbors) == 0:
        #         return None
        #     source_new = self._get_best_neighbor(neighbors, destination)

        # destination_new = (destination, 0.0)
        # if destination in self._gs_ids:
        #     neighbors = self._gs_neighbors.get(destination, [])
        #     if len(neighbors) == 0:
        #         return None
        #     destination_new = self._get_best_neighbor(neighbors, source)

        try:
            # return self.solver.get_path_cost(source_new[0], destination_new[0]) + source_new[1] + destination_new[1]
            return self.solver.get_path_cost(source, destination)
        except Exception:
            return float("inf")
        
    def get_distance(self, source, destination):
        cost = self.get_path_cost(source, destination)
        return cost if cost != float("inf") else None
    
    def get_neighbors(self, source, max_distance = None):
        if hasattr(self.solver, 'graph') and self.solver.graph:
            g = self.solver.graph
            if 0 <= source < len(g):
                neighbors = []
                for v, w in g[source].items():
                    if max_distance is None or w <= max_distance:
                        neighbors.append(v)
                return neighbors
        return []

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        return []

class SourceRoutingActor(MessageRoutingActor):
    def __init__(self,  provider: Optional[RoutingDataProvider] = None, solver: Union[GraphSolver, type[GraphSolver]] = BmsspSolver, update_interval: float = 15, advanced_cost=False, store_and_forward = False, model_bandwidth=False, attack_strategy = None, loss_config = None, reliable_transfer_config = UnreliableConfig()):
        if not provider:
            provider = SourceRoutingDataProvider(solver=solver, update_interval=update_interval)
        super().__init__(provider, store_and_forward, model_bandwidth, attack_strategy, loss_config, reliable_transfer_config)

class ResilientSourceRoutingDataProvider(SourceRoutingDataProvider):
    def __init__(self, get_next_hop_override: Optional[Callable[[BaseMessage, SatID, SatID], Optional[SatID]]] = None, solver: Union[GraphSolver, type[GraphSolver]] = BmsspSolver, update_interval: float = 15):
        super().__init__(get_next_hop_override, solver, update_interval)

class ResilientSourceRoutingActor(MessageRoutingActor):
    _reroute_limit: int = 3
    _local_view_depth: int = 3 # Depth for "5x5 grid" approximation
    
    def __init__(self, 
                 provider: Optional[RoutingDataProvider] = None, 
                 solver: Union[GraphSolver, type[GraphSolver]] = BmsspSolver, 
                 update_interval: float = 15, 
                 store_and_forward = False, 
                 model_bandwidth=False, 
                 attack_strategy = None, 
                 loss_config = None, 
                 reliable_transfer_config = None,
                 reroute_limit: int = 3):
        
        if not provider:
            provider = ResilientSourceRoutingDataProvider(solver=solver, update_interval=update_interval)
            
        super().__init__(provider, store_and_forward, model_bandwidth, attack_strategy, loss_config, reliable_transfer_config)
        self._reroute_limit = reroute_limit

    def _get_local_graph(self, mobility: MultiConstellation, center: SatID, depth: int) -> tuple[Graph, dict[SatID, int], dict[int, SatID]]:
        local_graph = Graph(weighted=True, directed=False)
        sat_to_node = {} # map SatID -> local graph node index
        node_to_sat = {} # map local graph node index -> SatID
        
        queue = deque([(center, 0)])
        visited = {center}
        
        nodes_in_graph = {center} 

        while queue:
            current_sat, d = queue.popleft()
            if d >= depth:
                continue

            network_neighbors = []

            for u, v in mobility.links:
                neighbor = None
                if u == current_sat:
                    neighbor = v
                elif v == current_sat:
                    neighbor = u
                
                if neighbor:
                    if (current_sat, neighbor) in self._failed_links or (neighbor, current_sat) in self._failed_links:
                         continue                
                    network_neighbors.append(neighbor)

            for neighbor in network_neighbors:
                nodes_in_graph.add(neighbor)
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))

        for i, sat in enumerate(nodes_in_graph):
            sat_to_node[sat] = i
            node_to_sat[i] = sat
        
        n = len(nodes_in_graph)
        local_graph.addNodes(n)

        for u, v in mobility.links:
            if (u, v) in self._failed_links or (v, u) in self._failed_links:
                continue

            if u in nodes_in_graph and v in nodes_in_graph:
                u_node = sat_to_node[u]
                v_node = sat_to_node[v]
                if not local_graph.hasEdge(u_node, v_node):
                    queue_u_v = len(self._stored_messages.get((u, v), []))
                    queue_v_u = len(self._stored_messages.get((v, u), []))
                    congestion_penalty = (queue_u_v + queue_v_u) * 0.01

                    dist = mobility.get_delay(u, v)
                    weight = (dist + congestion_penalty)
                    local_graph.addEdge(u_node, v_node, weight)

        return local_graph, sat_to_node, node_to_sat


    def _attempt_reroute(self, mobility: MultiConstellation, time: float, source: SatID, destination: SatID, message: DirectMessage, failure_reason: DropReason) -> List[Event]:
        current_reroutes = getattr(message, "reroute_count", 0)
        if current_reroutes >= self._reroute_limit:
            return [MessageDroppedEvent(time, source, message, DropReason.MAX_HOP_COUNT_EXCEEDED)]

        original_route = getattr(message, "route", None)
        current_index = getattr(message, "index", -1)
        
        if not original_route or current_index < 0:
            return [MessageDroppedEvent(time, source, message, failure_reason)]

        remaining_path = original_route[current_index+1:] 
        if not remaining_path:
             return [MessageDroppedEvent(time, source, message, failure_reason)]

        local_graph, sat_to_node, node_to_sat = self._get_local_graph(mobility, source, depth=3)
        
        dijkstra = Dijkstra(local_graph, sat_to_node[source])
        dijkstra.run()

        distances = dijkstra.getDistances()
        valid_distances = [d for d in distances if d != float('inf')]
        max_local_delay = max(valid_distances) if valid_distances else 0.0
        
        best_target = None
        best_path_local = None
        max_progress_index = -1
        
        for i, sat_id in enumerate(remaining_path):
            if sat_id in sat_to_node:
                dist = dijkstra.distance(sat_to_node[sat_id])
                if dist != float('inf'):
                    if i > max_progress_index:
                        max_progress_index = i
                        best_target = sat_id
                        path_nodes = dijkstra.getPath(sat_to_node[sat_id])
                        best_path_local = [node_to_sat[n] for n in path_nodes]

        if best_target and best_path_local:
            message.reroute_count = current_reroutes + 1

            target_idx_in_original = current_index + 1 + max_progress_index
            
            new_path_segment = best_path_local[1:]
            path_after_target = list(original_route[target_idx_in_original+1:]) 
                        
            full_new_route = list(original_route[:current_index+1]) + new_path_segment + path_after_target
            
            message.route = tuple(full_new_route)
            message.reroute = True
            
            return [MessageRerouteEvent(time + max_local_delay, source, destination, message)]

        return [MessageDroppedEvent(time, source, message, failure_reason)]

    def _send_message(self, mobility: MultiConstellation, time: float, source: SatID, destination: SatID, message: DirectMessage) -> list[Event]:        
        current_index = getattr(message, "index", -1)
        next_hop = self._routing.get_next_hop(source, destination, message)
        
        drop_reason = None
        
        if next_hop is None:
            drop_reason = DropReason.NO_NEXT_HOP
        else:
            if not mobility.has_link(source, next_hop) or (source, next_hop) in self._failed_links:
                drop_reason = DropReason.NO_ROUTE # or similar

        if drop_reason:
             message.index = current_index
             return self._attempt_reroute(mobility, time, source, destination, message, drop_reason)

        if current_index == -1:
            message.index = 0
        else:
            message.index = current_index
        
        events = super()._send_message(mobility, time, source, destination, message)
        
        keep_events = []
        for event in events:
            if isinstance(event, MessageDroppedEvent):
                message.index = current_index
                r_events = self._attempt_reroute(mobility, time, source, destination, message, event.reason)
                if any(isinstance(e, MessageDroppedEvent) for e in r_events):
                    keep_events.append(event)
                else:
                    keep_events.extend(r_events)
            else:
                keep_events.append(event)
                
        return keep_events

    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        events = super().handle_event(mobility, event)

        if isinstance(event, MessageDroppedEvent) and event.reason == DropReason.INSUFFICIENT_BUFFER:
            if not isinstance(event.message, DirectMessage):
                return []
            reroute_events = self._attempt_reroute(mobility, event.time, event.source, event.message.destination, event.message, event.reason)
            
            for e in reroute_events:
                if not isinstance(e, MessageDroppedEvent):
                    events.append(e)

        return events    

class HardwareFailureCheckEvent(Event):
    def __init__(self, time: float):
        super().__init__(time)

class HardwareFailureActor(Actor):
    def __init__(self, failure_rate: float, recovery_rate: float):
        super().__init__()
        self.failure_rate = failure_rate
        self.recovery_rate = recovery_rate
        self.failed_links: set[tuple[SatID, SatID]] = set()

    def initialize(self) -> list[Event]:
        return [HardwareFailureCheckEvent(0)] 
    
    def handle_event(self, mobility: MultiConstellation, event: Event) -> list[Event]:
        if not isinstance(event, HardwareFailureCheckEvent):
            return []
        events = []
        current_time = event.time
        
        recovered = []
        for link in self.failed_links:
            if random.random() < self.recovery_rate: 
                events.append(LinkUpEvent(current_time, link[0], link[1]))
                recovered.append(link)
        
        for link in recovered:
            self.failed_links.remove(link)

        all_links = mobility.links 
        for u, v in all_links:
            sat_u = mobility.satellites.by_id(u)
            sat_v = mobility.satellites.by_id(v)

            if sat_u.constellation_name == "ground" or sat_v.constellation_name == "ground":
                continue
            if sat_u.constellation_name != sat_v.constellation_name:
                continue

            link_key = tuple(sorted((u, v))) 
            
            if link_key not in self.failed_links:
                if random.random() < self.failure_rate:
                    self.failed_links.add(link_key)
                    events.append(LinkDownEvent(current_time, u, v))
        
        events.append(HardwareFailureCheckEvent(current_time + 1.0))
        
        return events



