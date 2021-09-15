import torch
import numpy as np

from collections import defaultdict

from Graph_Net.RandomWalks import RandomWalk
from Random_Stuff.Activations import sigmoid


class RandomWalkNeighbourLoss:
    """
    TODO:
    Reference: http://snap.stanford.edu/class/cs224w-2018/handouts/09-node2vec.pdf
    """

    def __init__(self, graph: np.ndarray, length: int, biased: bool, anonymous: bool,
                 return_param: float = None, in_out_param: float = None):
        assert length > 0, "Length should be bigger or equal to one."
        assert len(graph.shape) == 2 and graph.shape[0] == graph.shape[1], \
            "adjacency_matrix is not an adjacency matrix."

        self.num_nodes = graph.shape[0]
        self.node_p_distribution = np.sum(graph, axis=1) / self.num_nodes
        self.neighbours = RandomWalk(
            adjacency_matrix=graph,
            length=length,
            biased=biased,
            anonymous=anonymous,
            return_param=return_param,
            in_out_param=in_out_param
        ).get_walks(np.arange(self.num_nodes))

    def distance(self, node_embeddings: np.ndarray) -> float:
        return np.sum(np.vectorize(
            lambda x: np.vectorize(self.negative_sampling)(
                np.arange(self.num_nodes),
                x, node_embeddings
            ))(self.neighbours)).item()

    def negative_sampling(self, node: int, neighbour_node: int, node_embeddings: np.ndarray):
        node_emebdding = np.transpose(node_embeddings[node])
        return np.log(sigmoid(np.dot(node_emebdding, node_embeddings[neighbour_node]))) - np.sum(np.vectorize(
            lambda x: np.log(sigmoid(np.dot(node_emebdding, x))))(np.random.choice(
                np.arange(self.num_nodes),
                p=self.node_p_distribution
            )))


class Modularity:
    @staticmethod
    def modularity(edge_index: torch.Tensor, node_degrees: torch.Tensor, community_index: torch.Tensor):
        edge_index.t_()
        m = edge_index.shape[0]
        offset = 0
        modularity = 0
        edge_counter = defaultdict(bool)
        for edge_i in range(m):
            edge = edge_index[edge_i]
            n_expected_edges = (node_degrees[edge[0]] * node_degrees[edge[1]]) / (2 * m)
            if not edge_counter[tuple(edge.tolist())]:
                edge_counter[tuple(edge.tolist())] = True
                offset += n_expected_edges
                modularity += (1 - n_expected_edges)
            else:
                modularity += 1

        n_cummunities = torch.unique(community_index).shape[0]
        community_sums = torch.zeros(size=(n_cummunities, ))
        for degree, community_i in zip(node_degrees, community_index):
            community_sums[community_i.item()] += degree

        for degree, community_i in zip(node_degrees, community_index):
            p = degree / (2 * m)
            for community_j in range(n_cummunities):
                if community_j == community_i:
                    continue
                modularity -= p * community_sums[community_j]

        modularity = (1 / (2 * m)) * (modularity + offset)

        return modularity
