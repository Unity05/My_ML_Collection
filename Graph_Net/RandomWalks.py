import numpy as np


class RandomWalk:
    def __init__(self, adjacency_matrix: np.ndarray, length: int, biased: bool, anonymous: bool,
                 return_param: float = None, in_out_param: float = None):
        assert self.length > 0, "Length should be bigger or equal to one."
        assert len(self.adjacency_matrix.shape) == 2 and self.adjacency_matrix.shape[0] == self.adjacency_matrix.shape[1], \
            "adjacency_matrix is not an adjacency matrix."
        assert not self.biased or (self.return_param is not None and self.in_out_param is not None), \
            "A biased random walk needs return parameter and in out parameter."

        self.adjacency_matrix = adjacency_matrix
        self.length = length
        self.biased = biased
        self.anonymous = anonymous
        self.return_param = return_param
        self.in_out_param = in_out_param

    def random_walk(self, start_node: int) -> np.ndarray:
        assert start_node < self.adjacency_matrix.shape[0], "Node does not exist in the graph specified by the adjacency matrix."

        walk = np.empty(self.length)
        index = start_node
        for i in range(self.length):
            index = np.random.choice(self.adjacency_matrix[index].shape[0], 1,
                                     p=self.get_node_distribution(current_node=index, last_node=walk[i - 1]))
            walk[i] = index

        if self.anonymous:
            walk = self.make_walk_anonymous(walk=walk)

        return walk

    def get_walks(self, start_nodes: np.ndarray) -> np.ndarray:
        return np.vectorize(self.random_walk, signature='(n)->(m,n)')(start_nodes)

    def make_walk_anonymous(self, walk: np.ndarray) -> np.ndarray:
        unique_nodes = np.unique(walk)
        return np.vectorize(lambda x: unique_nodes[x])(walk)

    """
    TODO:
    reference: https://arxiv.org/pdf/1607.00653.pdf
    """
    def get_node_distribution(self, current_node: int, last_node: int = None) -> np.ndarray:
        if not self.biased:
            return self.adjacency_matrix[current_node]

        probability_distribution = np.empty(self.adjacency_matrix[current_node].shape[0])
        for i, value in enumerate(self.adjacency_matrix[current_node]):
            if value == 0 or (value == 1 and self.adjacency_matrix[last_node][i] == 1):
                probability_distribution[i] = value
            else:
                probability_distribution[i] = (1 / self.in_out_param)
        probability_distribution[last_node] = (1 / self.return_param)

        return probability_distribution
