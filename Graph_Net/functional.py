import torch

from collections import defaultdict


def modularity(edge_index: torch.Tensor, node_degrees: torch.Tensor, community_index: torch.Tensor) -> float:
    """
    Computes the modularity of a graph with given communities.
    Compatible with PyTorch Geometric.

    :param edge_index: A tensor that contains the edges / defines the graph structure.
    :param node_degrees: A tensor containing the degree for every node.
    :param community_index: A tensor containing a community index for every node.
    :return: the modularity.
    """

    modularity = 0
    edge_index.t_()
    m = edge_index.shape[0]
    offset = 0
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
    community_sums = torch.zeros(size=(n_cummunities,))
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
