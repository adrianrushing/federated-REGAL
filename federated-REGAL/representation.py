import numpy as np

def get_number_of_landmarks(graph, rep_method, k=10, server_override_p=None):
    """
    Given a graph, computes the number of landmark nodes.

    Inputs:
        - graph: A Graph object.
        - rep_method: A Rep_Method object.
        - k: A user-defined parameter that adjusts how many landmark nodes there are.

    Outputs:
    """
    if server_override_p is not None:
        rep_method.p = min(graph.N, int(server_override_p))
        return rep_method.p

    if graph.N <= 1:
        rep_method.p = graph.N
    else:
        rep_method.p = min(graph.N, int(k * np.log2(graph.N)))
    return rep_method.p

def get_random_landmarks(graph, rep_method, seed=None):
    """
    Given a graph and a rep_method, returns a random sample of rep_method.p landmark nodes, chosen without replacement.
    """  
    if seed is None:
        return np.random.choice(graph.N, rep_method.p, replace=False)

    rng = np.random.default_rng(seed)
    return rng.choice(graph.N, rep_method.p, replace=False)


def get_local_landmarks_from_global(global_landmarks, local_start_idx, local_num_nodes):
    """
    Maps server-side global landmark indices to local landmark row indices.

    Inputs:
        - global_landmarks: Iterable of node indices in global space.
        - local_start_idx: Inclusive start index of this client in global ordering.
        - local_num_nodes: Number of local nodes at this client.
    Outputs:
        - local_landmarks: Indices in [0, local_num_nodes).
    """
    local_end_idx = local_start_idx + local_num_nodes
    local_landmarks = [
        int(g_idx - local_start_idx)
        for g_idx in global_landmarks
        if local_start_idx <= g_idx < local_end_idx
    ]
    return np.array(local_landmarks, dtype=int)

def compute_similarity_score(d_u: np.ndarray, d_v: np.ndarray, gamma_s=1, gamma_a=None, f_u=None, f_v=None):
    """
    Helper function. Given two vectors that come from a feature matrix, computes the similarity between them according to equation (1) of the paper.

    Inputs:
        - d_u: First vector for node u.
        - d_v: Second vector for node v.
        - gamma_s: User-defined scalar parameter controlling the effect of the structural identity.
        - gamma_a: User-defined scalar parameter controlling the effect of the attribute identity. TODO: implement.
        - f_u: Attribute vector for node u. TODO: implement.
        - f_v: Attribute vector for node v. TODO: implement.
    """
    # Note: the paper says take norm squared, while the source code they provide says take norm. 
    # Here, I take norm squared to align with the paper.
    return np.exp(-1*gamma_s*np.linalg.norm(d_u - d_v)**2) 

def compute_C_matrix(feature_matrix, landmarks):
    """
    Given a feature matrix and a list of landmark nodes, computes the similarity matrix C between all the n nodes and the p landmark nodes.

    Inputs:
        - feature_matrix: The feature matrix from step 1 of the algorithm.
        - landmarks: The list of landmark nodes of the graph.
    Outputs:
        - C: The n x p similarity matrix.
    """
    C = np.zeros((len(feature_matrix), len(landmarks)))

    for n in range(len(feature_matrix)):
        for j in range(len(landmarks)):
            C[n, j] = compute_similarity_score(d_u=feature_matrix[n], d_v = feature_matrix[landmarks[j]])

    return C


def compute_C_matrix_from_basis(feature_matrix, basis_matrix):
    """
    Computes C using server-provided structural basis vectors rather than node indices.

    Inputs:
        - feature_matrix: n x d matrix of node features.
        - basis_matrix: p x d matrix of shared basis vectors.
    Outputs:
        - C: n x p similarity matrix.
    """
    C = np.zeros((feature_matrix.shape[0], basis_matrix.shape[0]))
    for n in range(feature_matrix.shape[0]):
        for j in range(basis_matrix.shape[0]):
            C[n, j] = compute_similarity_score(d_u=feature_matrix[n], d_v=basis_matrix[j])
    return C


def compute_projection_from_C(C, landmarks, normalize_rows=True, server_projection=None):
    """
    Computes shared-space embeddings from C.

    Inputs:
        - C: n x p similarity matrix.
        - landmarks: Landmark rows in the same indexing domain as C.
        - normalize_rows: Row-normalize output embeddings.
        - server_projection: Optional tuple (U, Sigma) from server to enforce
          a globally shared coordinate system.
    Outputs:
        - Y_twiddle: n x p embedding matrix.
    """
    if server_projection is None:
        W = C[landmarks, :]
        W_dagger = np.linalg.pinv(W)
        U, Sigma, _ = np.linalg.svd(W_dagger)
    else:
        U, Sigma = server_projection

    Sigma = np.diag(Sigma) if Sigma.ndim == 1 else Sigma
    Y_twiddle = C @ U @ np.sqrt(Sigma)

    if normalize_rows:
        denom = np.linalg.norm(Y_twiddle, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        Y_twiddle = Y_twiddle / denom

    return Y_twiddle


def compute_representation_federated(C, landmarks, server_projection=None, normalize_rows=True):
    """
    Federated variant returning one embedding block without assuming a split index.
    """
    return compute_projection_from_C(
        C=C,
        landmarks=landmarks,
        normalize_rows=normalize_rows,
        server_projection=server_projection,
    )

def compute_representation(C, landmarks, n_1):
    """
    Given an n x p similarity matrix C, a list of landmark nodes, and the number of nodes in the first graph, computes the representations of nodes of the two original graphs. See algorithm 2, step 2b, for this function in pseudocode.

    Inputs:
        - C: n x p similarity matrix.
        - landmarks: The list of landmark nodes in the graph.
        - n_1: Number of nodes in the first graph.

    Outputs:
        - Y_twiddle_1: Representations of the nodes of the first graph.
        - Y_twiddle_2: Representations of the nodes of the second graph.
    """
    Y_twiddle = compute_projection_from_C(C=C, landmarks=landmarks, normalize_rows=True)
    
    Y_twiddle_1, Y_twiddle_2 = Y_twiddle[:n_1, :], Y_twiddle[n_1:, :] # Split representations for nodes in G_1, G_2

    return Y_twiddle_1, Y_twiddle_2