import os
import msgpack
import hashlib
import time
from typing import Optional, Dict, List, Any
from .graph import Graph, Edge

class GraphCache:
    """
    Handles caching of Graph objects to speed up repeated dataset loading.
    Uses msgpack for fast serialization and file modification time + size for cache validation.
    """
    
    def __init__(self, cache_dir: str = "data/.cache"):
        """
        Initialize the graph cache.
        
        Args:
            cache_dir: Directory to store cached graph files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _serialize_graph(self, graph: Graph) -> bytes:
        """
        Convert Graph object to msgpack-serializable format.
        
        Args:
            graph: Graph object to serialize
            
        Returns:
            Serialized bytes using msgpack
        """
        # Convert adjacency list to a simple format: [vertices, [[to, weight], ...], ...]
        # Ensure all values are native Python types (not numpy types)
        adj_data = []
        for vertex_edges in graph.adj:
            edge_list = [[int(edge.to), float(edge.weight)] for edge in vertex_edges]
            adj_data.append(edge_list)
        
        graph_data = {
            'vertices': int(graph.vertices),
            'adj': adj_data
        }
        
        return msgpack.packb(graph_data, use_bin_type=True)
    
    def _deserialize_graph(self, data: bytes) -> Graph:
        """
        Convert msgpack data back to Graph object.
        
        Args:
            data: Serialized graph data
            
        Returns:
            Reconstructed Graph object
        """
        graph_data = msgpack.unpackb(data, raw=False)
        
        graph = Graph(graph_data['vertices'])
        
        # Reconstruct adjacency list
        for vertex_idx, edge_list in enumerate(graph_data['adj']):
            for to, weight in edge_list:
                graph.adj[vertex_idx].append(Edge(to, weight))
        
        return graph
    
    def _get_cache_key(self, file_path: str, is_directed: Optional[bool] = None) -> str:
        """
        Generate a cache key based on the file path and graph properties.
        
        Args:
            file_path: Path to the original graph file
            is_directed: Whether the graph is directed (None for DIMACS files)
            
        Returns:
            Cache key string
        """
        # Create a hash based on file path and properties
        key_string = f"{file_path}_{is_directed}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cached file."""
        return os.path.join(self.cache_dir, f"{cache_key}.msgpack")
    
    def _get_metadata_path(self, cache_key: str) -> str:
        """Get the path for cache metadata file."""
        return os.path.join(self.cache_dir, f"{cache_key}.meta")
    
    def _is_cache_valid(self, file_path: str, cache_key: str) -> bool:
        """
        Check if the cached version is still valid by comparing file modification time and size.
        
        Args:
            file_path: Path to the original graph file
            cache_key: Cache key for the file
            
        Returns:
            True if cache is valid, False otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        if not os.path.exists(cache_path) or not os.path.exists(metadata_path):
            return False
        
        try:
            # Get current file stats
            file_stat = os.stat(file_path)
            current_mtime = file_stat.st_mtime
            current_size = file_stat.st_size
            
            # Read cached metadata
            with open(metadata_path, 'r') as f:
                lines = f.read().strip().split('\n')
                cached_mtime = float(lines[0])
                cached_size = int(lines[1])
            
            # Cache is valid if modification time and size match
            return (abs(current_mtime - cached_mtime) < 1.0 and 
                    current_size == cached_size)
            
        except (OSError, ValueError, IndexError):
            return False
    
    def _save_metadata(self, file_path: str, cache_key: str):
        """Save metadata about the original file for cache validation."""
        try:
            file_stat = os.stat(file_path)
            metadata_path = self._get_metadata_path(cache_key)
            
            with open(metadata_path, 'w') as f:
                f.write(f"{file_stat.st_mtime}\n")
                f.write(f"{file_stat.st_size}\n")
        except OSError:
            pass  # If we can't save metadata, caching will be disabled for this file
    
    def load_cached_graph(self, file_path: str, is_directed: Optional[bool] = None) -> Optional[Graph]:
        """
        Load a graph from cache if available and valid.
        
        Args:
            file_path: Path to the original graph file
            is_directed: Whether the graph is directed (None for DIMACS files)
            
        Returns:
            Cached Graph object if available and valid, None otherwise
        """
        cache_key = self._get_cache_key(file_path, is_directed)
        
        if not self._is_cache_valid(file_path, cache_key):
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            print(f"Loading graph from cache: {cache_path}")
            start_time = time.time()
            
            with open(cache_path, 'rb') as f:
                data = f.read()
                graph = self._deserialize_graph(data)
            
            load_time = time.time() - start_time
            print(f"Graph loaded from cache in {load_time:.3f} seconds")
            print(f"Graph stats: {graph.vertices} vertices, {sum(len(adj) for adj in graph.adj)} edges")
            
            return graph
            
        except (OSError, msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackException, 
                msgpack.exceptions.UnpackValueError, KeyError, AttributeError) as e:
            print(f"Warning: Failed to load cached graph: {e}")
            # Remove corrupted cache files
            try:
                os.remove(cache_path)
                metadata_path = self._get_metadata_path(cache_key)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
            except OSError:
                pass
            return None
    
    def save_graph_to_cache(self, graph: Graph, file_path: str, is_directed: Optional[bool] = None):
        """
        Save a graph to cache.
        
        Args:
            graph: Graph object to cache
            file_path: Path to the original graph file
            is_directed: Whether the graph is directed (None for DIMACS files)
        """
        cache_key = self._get_cache_key(file_path, is_directed)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            print(f"Saving graph to cache: {cache_path}")
            start_time = time.time()
            
            serialized_data = self._serialize_graph(graph)
            with open(cache_path, 'wb') as f:
                f.write(serialized_data)
            
            # Save metadata for cache validation
            self._save_metadata(file_path, cache_key)
            
            save_time = time.time() - start_time
            cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"Graph cached in {save_time:.3f} seconds ({cache_size_mb:.1f} MB)")
            
        except (OSError, msgpack.exceptions.PackException) as e:
            print(f"Warning: Failed to cache graph: {e}")
