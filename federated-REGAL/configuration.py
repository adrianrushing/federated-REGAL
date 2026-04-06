import numpy as np

class RepMethod:
    def __init__(self, 
                 align_info=None, 
                 p=None, 
                 k=10, 
                 max_layer=None, 
                 alpha=0.1, 
                 num_buckets=None, 
                 normalize=True, 
                 gammastruc=1, 
                 gammaattr=1):
        self.p = p  # number of points to sample
        self.k = k  # sample size controller
        self.max_layer = max_layer  # maximum hop distance for neighbor comparison
        self.alpha = alpha  # layer discount factor
        self.num_buckets = num_buckets  # buckets for node feature splitting (log scale base)
        self.normalize = normalize  # toggle normalization of node embeddings
        self.gammastruc = gammastruc  # weight for structural similarity
        self.gammaattr = gammaattr  # weight for attribute similarity

class Graph:
    # Represents an undirected, unweighted graph
    def __init__(self, 
                 adj, 
                 num_buckets=None, 
                 node_labels=None, 
                 edge_labels=None, 
                 graph_label=None, 
                 node_attributes=None, 
                 true_alignments=None):
        self.G_adj = adj  # adjacency matrix of the graph
        self.N = adj.shape[0]  # number of nodes
        self.node_degrees = np.sum(adj, axis=0).astype(int).flatten()
        self.max_degree = np.max(self.node_degrees)  # maximum node degree
        self.num_buckets = num_buckets  # bucket count for splitting features

        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_label = graph_label
        self.node_attributes = node_attributes  # NxA matrix (N: nodes, A: attributes)
        self.kneighbors = None  # dictionary for k-hop neighbors of nodes
        self.true_alignments = true_alignments  # mapping of true alignments for combined graphs


class FederatedStructuralConfig:
    def __init__(
        self,
        global_max_degree,
        k_hops,
        delta,
        basis_size,
        basis_version="v1",
        seed=0,
    ):
        self.global_max_degree = int(global_max_degree)
        self.k_hops = int(k_hops)
        self.delta = float(delta)
        self.basis_size = int(basis_size)
        self.basis_version = basis_version
        self.seed = int(seed)

    @property
    def expected_feature_width(self):
        return self.k_hops * (self.global_max_degree + 1)
