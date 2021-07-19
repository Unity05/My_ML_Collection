import torch
import torch.nn as nn


class GNN(nn.Module):
    def __init__(self, depth: int, embed_dim: int, activation: str = 'ReLU'):
        super(GNN, self).__init__()
        self.depth = depth
        self.layers = [GNNLayer(in_dim=embed_dim, out_dim=embed_dim, activation=activation) for _ in range(depth)]
        self.h = None
        self.adjacency_matrix = None

    def forward(self, x: tuple):
        self.adjacency_matrix, graph = x    # adjust adjacency matrix to batch usage
        # embed matrix of shape [N; Depth; #Nodes; embed_dim]
        self.h = torch.empty((graph.size()[0], self.depth, graph.size()[1], graph.size()[2]))
        for node_index in range(graph.size()[1]):
            self.get_node_embedding(node_index=node_index, remaining_its=(self.depth - 1))

    def get_node_embedding(self, node_index: int, remaining_its: int):
        if remaining_its != 0:
            self.h[:][remaining_its][node_index] = self.layers[remaining_its](self.h[:][remaining_its - 1][self.adjacency_matrix[:, node_index]])


class GNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str = 'ReLU'):
        super(GNNLayer, self).__init__()
        self.message_layer = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.activation_fn = getattr(nn, activation)

    def forward(self, x):
        # x of shape [N; #number_of_nodes; D]
        return self.activation_fn((1 / x.size()[1]) * torch.sum(self.message_layer(x), dim=1))
