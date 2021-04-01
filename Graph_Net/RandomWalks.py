import numpy as np


def random_walk(adjacency_matrix: np.ndarray, start_node: int, length: int, biased: bool, anonymous: bool,
                return_param: float = None, in_out_param: float = None) -> np.ndarray:
    assert length > 0, "Length should be bigger or equal to one."
    assert len(adjacency_matrix.shape) == 2 and adjacency_matrix.shape[0] == adjacency_matrix.shape[1], \
        "adjacency_matrix is not an adjacency matrix."
    assert start_node < adjacency_matrix.shape[0], "Node does not exist in the graph specified by the adjacency matrix."
    assert not biased or (return_param is not None and in_out_param is not None), \
        "A biased random walk needs return parameter and in out parameter."

    walk = np.empty(length)
    index = start_node
    for i in range(length):
        index = np.random.choice(adjacency_matrix[index].shape[0], 1,
                                 p=get_node_distribution(adjacency_matrix=adjacency_matrix, biased=biased,
                                                         current_node=index, last_node=walk[i - 1],
                                                         return_param=return_param, in_out_param=in_out_param))
        walk[i] = index

    if anonymous:
        walk = make_walk_anonymous(walk=walk)

    return walk


def make_walk_anonymous(walk: np.ndarray) -> np.ndarray:
    unique_nodes = np.unique(walk)
    return np.vectorize(lambda x: unique_nodes[x])(walk)


def get_node_distribution(adjacency_matrix: np.ndarray, biased: bool, current_node: int,
                          last_node: int = None, return_param: float = None, in_out_param: float = None) -> np.ndarray:
    if biased:
        return adjacency_matrix[current_node]

    probability_distribution = np.empty(adjacency_matrix[current_node].shape[0])
    for i, value in enumerate(adjacency_matrix[current_node]):
        if value == 0 or (value == 1 and adjacency_matrix[last_node][i] == 1):
            probability_distribution[i] = value
        else:
            probability_distribution[i] = (1 / in_out_param)
    probability_distribution[last_node] = (1 / return_param)

    return probability_distribution
