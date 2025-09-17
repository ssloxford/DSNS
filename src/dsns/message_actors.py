
from collections import deque
from typing import Optional, Callable

from networkit import Graph
from networkit.distance import Dijkstra
from networkit.dynamics import GraphDifference, GraphEventType
import networkit.graphtools
import numpy as np

try:
    from dsns.key_management import AttackStrategy
except ModuleNotFoundError:
    AttackStrategy = None
from dsns.message import AttackMessageDroppedEvent, BaseMessage, BroadcastMessage, DirectMessage, DropReason, HybridDirectMessage, LTPSegment, LTPSegmentCreatedEvent, LTPSegmentReceivedEvent, LinkLossProbabilityUpdateEvent, LossConfig, MessageBroadcastDeliveredEvent, MessageCreatedEvent, MessageDeliveredEvent, MessageDroppedEvent, MessageLostEvent, MessageQueuedEvent, MessageReceivedEvent, MessageRerouteEvent, MessageSentEvent, ReliableTransferConfig, UnreliableConfig

from .multiconstellation import MultiConstellation
from .events import Event, LinkUpEvent, LinkDownEvent
from .helpers import SatID
from .simulation import Actor, RoutingDataProvider


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

    def __send_message(self, mobility: MultiConstellation, time: float, source: SatID, destination: SatID, message: DirectMessage) -> list[Event]:
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
            if not mobility.has_link(source, next_hop):
                if self._store_and_forward:
                    self._stored_messages.setdefault((source, next_hop), []).append(message)
                    return []
                else:
                    return [ MessageDroppedEvent(time, source, message, DropReason.INSUFFICIENT_BUFFER) ]
            else:
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
            return self.__send_message(mobility, event.time, event.destination, message.destination, message)

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
            return self.__send_message(mobility, event.time, event.message.source, message.destination, message)

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
        return self.__send_message(mobility, event.time, event.source, message.destination, message)

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
        elif isinstance(event, LinkUpEvent):
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
