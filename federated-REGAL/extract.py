# extract.py

import numpy as np
from scipy import sparse

class Graph:
    def __init__(self, adj, attributes=None, global_max_degree=None):
        """
        Initializes the Graph object.

        Inputs:
            - adj: Adjacency matrix (sparse CSR matrix).
            - attributes: Attribute matrix (optional, numpy array).
        """
        self.adj = adj.tocsr()  # Ensure adjacency matrix is in CSR format
        self.attributes = attributes  # Attribute matrix (optional)
        self.N = adj.shape[0]  # Number of nodes
        self.degrees = np.array(adj.sum(axis=1)).flatten()
        local_max_degree = int(self.degrees.max()) if self.degrees.size > 0 else 0
        self.max_degree = local_max_degree if global_max_degree is None else int(global_max_degree)

def count_degree_distributions(neighbors, adj):
    """
    Counts the degree distributions of the given neighbors.
    """
    if not neighbors:
        return np.zeros(1, dtype=int)
    
    degrees = np.array(adj[neighbors, :].sum(axis=1)).flatten()
    degree_distribution, _ = np.histogram(degrees, bins=np.arange(degrees.max() + 2))
    return degree_distribution

def get_k_hop_neighbors(node, k, adj):
    """
    Finds the k-hop neighbors of a given node.
    """
    neighbors = set([node])
    current_layer = set([node])
    
    for hop in range(1, k + 1):
        next_layer = set()
        for n in current_layer:
            connected_nodes = adj.getrow(n).indices
            next_layer.update(connected_nodes)
        neighbors.update(next_layer)
        current_layer = next_layer
    
    neighbors.discard(node)
    return list(neighbors)

def extract_node_identity(graph, K, delta, global_max_degree=None, expected_feature_length=None):
    """
    Extracts the node identity features for all nodes in the graph.
    """
    degree_max = int(graph.max_degree if global_max_degree is None else global_max_degree)
    feature_length = int(K * (degree_max + 1))
    if expected_feature_length is not None and feature_length != int(expected_feature_length):
        raise ValueError(
            f"Feature width mismatch: computed {feature_length}, expected {int(expected_feature_length)}"
        )

    node_identities = np.zeros((graph.N, feature_length))
 
    for node in range(graph.N):
        aggregated_distribution = np.zeros(feature_length)
        
        for k in range(1, K + 1):
            k_hop_neighbors = get_k_hop_neighbors(node, k, graph.adj)
            degree_distribution = count_degree_distributions(k_hop_neighbors, graph.adj)
            
            if len(degree_distribution) < (degree_max + 1):
                degree_distribution = np.pad(
                    degree_distribution, 
                    (0, degree_max + 1 - len(degree_distribution)), 
                    'constant'
                )
            else:
                degree_distribution = degree_distribution[:degree_max + 1]
            aggregated_distribution[
                (k-1)*(degree_max + 1) : k*(degree_max + 1)
            ] = (delta ** (k-1)) * degree_distribution
        
        node_identities[node] = aggregated_distribution
    
    return node_identities