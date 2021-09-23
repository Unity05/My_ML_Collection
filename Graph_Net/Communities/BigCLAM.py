import torch
import torch.nn as nn

from Graph_Net.Loss import GraphLogLikelihood


class BigCLAM:
    """
    Implementation of BigCLAM described in the paper
    "Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach"
    (https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf)
    """

    def __init__(self, convergence_threshold: float = 0.8):
        """
        :param convergence_threshold: A scalar threshold for when the training should stop.
        """

        self.convergence_threshold = convergence_threshold

    def run(self, edge_index: torch.Tensor, n_nodes: int, n_communities: int,
            device: torch.device, lr: float = 1.0e-3) -> (torch.Tensor, float):
        """
        Computes the community membership table F for a given graph.

        :param edge_index: A tensor that contains the edges / defines the graph structure.
        :param n_nodes: Number of nodes in the graph.
        :param n_communities: Number of possible communities.
        :param device: device to run the computations on.
        :param lr: learning rate for community membership table training.
        :return: the trained community membership table F and the corresponding log likelihood.
        """

        objective = GraphLogLikelihood(
            edge_index=edge_index.to(device),
            n_nodes=n_nodes
        )
        membership = nn.Parameter(torch.empty((n_nodes, n_communities), device=device))
        torch.nn.init.kaiming_normal_(membership)
        optim = torch.optim.SGD(params=membership, lr=lr)

        # Training procedure.
        likelihood_value = 0
        while likelihood_value < self.convergence_threshold:
            likelihood = objective(membership)
            likelihood_value = likelihood.item()
            optim.zero_grad()
            likelihood.backward()
            optim.step()

        return membership.detach(), likelihood_value
