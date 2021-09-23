import numpy as np
import torch
import torch.nn as nn
import itertools

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


class GraphLogLikelihood(nn.Module):
    """
    LogLikelihood log(P(G|F)) that a given community membership table F generates the ground truth graph G.
    """

    def __init__(self, edge_index: torch.Tensor, n_nodes: int):
        """
        :param edge_index: A tensor that contains the edges / defines the graph structure.
        :param n_nodes: Number of nodes in the graph.
        """

        super(GraphLogLikelihood, self).__init__()
        self.edge_index = edge_index
        non_edge_list = list(itertools.combinations(range(n_nodes), r=2))
        for edge in edge_index.t().tolist():
            non_edge_list.remove(edge)
        self.non_edge_index = torch.tensor(non_edge_list).t()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the log likelihood log(P(G|F)).

        :param input: A tensor representing the community membership table F of shape (#nodes, #communities).
        :return: scalar tensor.
        """

        loss = torch.sum(torch.log(1 - torch.exp((-1) * torch.einsum('ij,ij->i',
                                                                     input[self.edge_index[0]],
                                                                     input[self.edge_index[1]])))) \
               - torch.sum(torch.einsum('ij,ij->i', input[self.non_edge_index[0]], input[self.non_edge_index[1]]))
        return loss
