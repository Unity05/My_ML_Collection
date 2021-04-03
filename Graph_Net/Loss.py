import numpy as np

from .RandomWalks import RandomWalk
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
