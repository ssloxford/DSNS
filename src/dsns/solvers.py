from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, List, Set, Tuple
from collections import defaultdict
import networkit
from networkit.distance import Dijkstra as NKDijkstra

from dsns.helpers import SatID
from dsns.multiconstellation import MultiConstellation
from dsns.bmssp_solver import BmsspSolver as LibBmsspSolver
# from dsns.bmssp_solver import BmsspSolverV2 as LibBmsspSolverV2
from dsns.comparison_solvers import dijkstra_sssp, dijkstra
from dsns.graph import Graph as LibGraph


class GraphSolver(ABC):
    # Graph is an adjacency dict: u -> {v -> weight}
    graph: Dict[SatID, Dict[SatID, float]]
    # Optional instance of the library's Graph object (lazily built)
    lib_graph: Optional[LibGraph]
    # Cache stores distance results: source -> distances
    cache: Dict[SatID, Tuple[List[float], List[Optional[int]]]]
    n: int

    def __init__(self) -> None:
        super().__init__()
        self.graph = defaultdict(dict)
        self.lib_graph = None
        self.cache = {}
        self.n = 0

    def update(
        self,
        data: Union[MultiConstellation, int],
        costs: Optional[Dict[Tuple[SatID, SatID], float]] = None,
    ) -> None:
        self.cache.clear()
        self.graph = defaultdict(dict)
        self.lib_graph = None

        if isinstance(data, MultiConstellation):
            mobility = data
            self.n = len(mobility.satellites)
            for u, v in mobility.links:
                w_uv = mobility.get_delay(u, v)
                w_vu = mobility.get_delay(v, u)
                self.graph[u][v] = w_uv
                self.graph[v][u] = w_vu

        elif isinstance(data, int):
            if costs is None:
                raise ValueError("If 'n' is provided, 'costs' must also be provided.")
            self.n = data
            for (u, v), c in costs.items():
                self.graph[u][v] = c
                self.graph[v][u] = c
        else:
            raise TypeError(f"Unexpected type for data: {type(data)}")

    def _ensure_lib_graph(self) -> None:
        if self.lib_graph is None:
            self.lib_graph = LibGraph(self.n)
            for u, neighbors in self.graph.items():
                for v, w in neighbors.items():
                    if 0 <= u < self.n and 0 <= v < self.n:
                        self.lib_graph.add_edge(u, v, w)

    @abstractmethod
    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        pass

    @abstractmethod
    def get_path(self, source: SatID, destination: SatID) -> List[SatID]:
        pass

    # FOR BENCHMARKING ONLY, BYPASSES CACHE
    @abstractmethod
    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        pass

    def remove_edges(self, edges: Set[Tuple[SatID, SatID]]) -> None:
        for u, v in edges:
            if u in self.graph and v in self.graph[u]:
                self.graph[u].pop(v, None)
            if v in self.graph and u in self.graph[v]:
                self.graph[v].pop(u, None)
        self.lib_graph = None
        self.cache.clear()

    def _reconstruct_path(self, predecessors: List[Optional[int]], source: int, destination: int) -> List[int]:
        """Helper to reconstruct path from predecessor array."""
        if not (0 <= destination < len(predecessors)):
            return []
            
        path = []
        curr = destination

        while curr is not None:
            path.append(curr)
            if curr == source:
                break

            curr = predecessors[curr]

        if not path or path[-1] != source:
            return []

        return path[::-1]


class BmsspSolver(GraphSolver):
    def __init__(self) -> None:
        super().__init__()


    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        res = self.cache.get(source)
        if res is None:
            self._ensure_lib_graph()
            solver = LibBmsspSolver(self.lib_graph)
            # Returns (distances, predecessors)
            res = solver.solve_sssp(source)
            self.cache[source] = res

        distances, _ = res
        if 0 <= destination < len(distances):
            val = distances[destination]
            return val
        return float("inf")

    def get_path(self, source: SatID, destination: SatID) -> List[SatID]:
        res = self.cache.get(source)
        if res is None:
            self._ensure_lib_graph()
            solver = LibBmsspSolver(self.lib_graph)
            res = solver.solve_sssp(source)
            self.cache[source] = res
            
        distances, predecessors = res
        
        if 0 <= destination < len(distances):
            if distances[destination] == float('inf'):
                return []
        else:
            return []

        return self._reconstruct_path(predecessors, source, destination)

    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        self._ensure_lib_graph()
        solver = LibBmsspSolver(self.lib_graph)
        result = solver.solve(source, destination)
        if result:
            return result[0]
        return float("inf")


class DijkstraSolver(GraphSolver):
    def __init__(self) -> None:
        super().__init__()

    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        res = self.cache.get(source)
        if res is None:
            self._ensure_lib_graph()
            res = dijkstra_sssp(self.lib_graph, source)
            self.cache[source] = res

        distances, _ = res
        if 0 <= destination < len(distances):
            val = distances[destination]
            return val
        return float("inf")

    def get_path(self, source: SatID, destination: SatID) -> List[SatID]:
        res = self.cache.get(source)
        if res is None:
            self._ensure_lib_graph()
            res = dijkstra_sssp(self.lib_graph, source)
            self.cache[source] = res
            
        distances, predecessors = res
        
        if 0 <= destination < len(distances):
            if distances[destination] == float('inf'):
                return []
        else:
            return []

        return self._reconstruct_path(predecessors, source, destination)

    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        self._ensure_lib_graph()
        result = dijkstra(self.lib_graph, source, destination)
        if result:
            return result[0]
        return float("inf")

class NetworkItDijkstraSolver(GraphSolver):
    __graph: Optional[networkit.Graph]
    __sat_to_node_id: Dict[SatID, int]
    __node_to_sat_id: Dict[int, SatID]

    def __init__(self) -> None:
        super().__init__()
        self.__graph = None
        self.__sat_to_node_id = {}
        self.__node_to_sat_id = {}

    def update(
        self,
        data: Union[MultiConstellation, int],
        costs: Optional[Dict[Tuple[SatID, SatID], float]] = None,
    ) -> None:
        # Build the standard self.graph dictionary
        super().update(data, costs)

        # Build the NetworKit graph
        # Note: self.n is populated by the super().update call
        self.__graph = networkit.Graph(self.n, weighted=True, directed=False)
        self.__sat_to_node_id = {}
        self.__node_to_sat_id = {}

        # Keep parity by assigning generic ID mapping
        for i in range(self.n):
            self.__sat_to_node_id[i] = i
            self.__node_to_sat_id[i] = i

        # Populate the networkit graph edges
        # using `u < v` to ensure undirected edges are only added once
        for u, neighbors in self.graph.items():
            for v, w in neighbors.items():
                if u < v and 0 <= u < self.n and 0 <= v < self.n:
                    self.__graph.addEdge(self.__sat_to_node_id[u], self.__sat_to_node_id[v], w)

    def remove_edges(self, edges: Set[Tuple[SatID, SatID]]) -> None:
        # Base graph dict update and cache invalidate
        super().remove_edges(edges)

        # Remove edges from networkit internal graph
        if self.__graph is not None:
            for u, v in edges:
                 if u in self.__sat_to_node_id and v in self.__sat_to_node_id:
                     u_node = self.__sat_to_node_id[u]
                     v_node = self.__sat_to_node_id[v]
                     if self.__graph.hasEdge(u_node, v_node):
                        self.__graph.removeEdge(u_node, v_node)

    def _get_nk_sssp(self, source: SatID) -> NKDijkstra:
        """Helper to get and cache a NetworkIt Dijkstra run for a given source"""
        res = self.cache.get(source)
        if res is None:
            # Run Dijkstra (storePaths=True, storeNodesSortedByDistance=True)
            res = NKDijkstra(self.__graph, self.__sat_to_node_id[source], True, True)
            res.run()
            self.cache[source] = res
        return res

    def get_path_cost(self, source: SatID, destination: SatID) -> float:
        if self.__graph is None or source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return float('inf')

        nk_dijkstra = self._get_nk_sssp(source)
        
        # NetworKit's Dijkstra sets unreachables/same nodes to 0 or large values, check properly
        dist = nk_dijkstra.distance(self.__sat_to_node_id[destination])
        # Very high float values might be returned for unreachable bounds depending on networkit builds 
        if dist > 1e12 or (dist == 0 and source != destination):
            return float('inf')
        return dist

    def get_path(self, source: SatID, destination: SatID) -> List[SatID]:
        if self.__graph is None or source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return []
        
        if source == destination:
            return [source]

        nk_dijkstra = self._get_nk_sssp(source)
        nk_path = nk_dijkstra.getPath(self.__sat_to_node_id[destination])

        # NK getPath returns excluding the source node, e.g., A->B->C gets [B, C]
        if not nk_path and source != destination:
            return []

        # Reconstruct path and map back to sat IDs
        path = [self.__node_to_sat_id[node] for node in nk_path]
        return [source] + path

    def benchmark_solve(self, source: SatID, destination: SatID) -> float:
        """Run Dijkstra directly without caching for accurate benchmarking."""
        if self.__graph is None or source not in self.__sat_to_node_id or destination not in self.__sat_to_node_id:
            return float('inf')

        nk_dijkstra = NKDijkstra(self.__graph, self.__sat_to_node_id[source], True, True)
        nk_dijkstra.run()
        dist = nk_dijkstra.distance(self.__sat_to_node_id[destination])
        if dist > 1e12 or (dist == 0 and source != destination):
            return float('inf')
        return dist
