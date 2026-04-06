## Import Section
from sklearn.neighbors import KDTree
from scipy.sparse import *
import numpy as np


def get_top_n_alignments(X,Y,n):
    '''
    Function to produce a KD-Tree and query in order to get the best alignments

    input params: 
    X: Embedding to be aligned
    Y: Embedding to which the alignment should be conducted
    n: Number of top alignments

    return params: 
    d: Array including the distance to each of the nearest nodes
    i: List of indices of the nearest nodes
    '''
    # Create an embedding of Y in a KD-Tree, euclidean distance is used in accordance with the paper
    kd_tree = KDTree(Y, metric = "euclidean")

    # Query that tree to get the alignment distances and indices
    d,i = kd_tree.query(X, k = n)

    return d,i


def get_similarity_matrix(X,Y,n): 
    '''
    Function to calculate tge similarity matrix as per the REGAL Paper
    input params: 
    X: Embedding to be aligned
    Y: Embedding to which the alignment should be conducted
    n: Number of top alignments

    return params: 

    '''
    # Get the top n alignments
    distances,indices = get_top_n_alignments(X,Y,n)

    # Initalize the DOK Matrix
    sparse_align_matrix = dok_matrix((X.shape[0], Y.shape[0]))
    # Loop through all elements to be embedded
    for i in range(X.shape[0]):
        for j in range(n):
            row_index = i
            col_index = indices[i, j]
            # Populate a DOK matrix with similarity scores as defined   
            sparse_align_matrix[row_index, col_index] = np.exp(-distances[i, j])

    return sparse_align_matrix.tocsr()


def split_embeddings(embedding_matrix):
    '''
    Helper Function to split the embeddings if necessary

    input params: 
    embedding_matrix: Matrix including the combined embeddings
    return params: 
    X,Y: Split embeddings
    '''

    split_index= int(embedding_matrix.shape[0] / 2)
    X = embedding_matrix[:split_index]
    Y = embedding_matrix[split_index:]

    return X, Y


def get_federated_pairwise_alignments(client_embeddings, top_n=10):
    '''
    Computes pairwise top-n sparse similarity matrices for all client embedding blocks.

    input params:
    client_embeddings: Dict keyed by client id with value as 2D embedding array.
    top_n: Number of top alignments for each query node.

    return params:
    pairwise_matrices: Dict with keys (client_a, client_b) and sparse similarity matrices.
    '''
    client_ids = sorted(client_embeddings.keys())
    pairwise_matrices = {}

    for i, client_a in enumerate(client_ids):
        for j in range(i + 1, len(client_ids)):
            client_b = client_ids[j]
            X = client_embeddings[client_a]
            Y = client_embeddings[client_b]
            n_local = min(int(top_n), Y.shape[0])
            if n_local <= 0:
                continue
            pairwise_matrices[(client_a, client_b)] = get_similarity_matrix(X, Y, n_local)

    return pairwise_matrices

