import math

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
        # embed matrix of shape [Depth; N; #Nodes; embed_dim]
        self.h = torch.empty((self.depth, graph.size()[0], graph.size()[1], graph.size()[2]))
        self.h[0] = graph
        self.calculate_embedding(remaining_its=(self.depth - 1))

    def calculate_embedding(self, remaining_its: int):
        if remaining_its != 0:
            self.calculate_embedding(remaining_its=(remaining_its - 1))
            self.h[remaining_its] = self.layers[remaining_its](self.h[remaining_its - 1],
                                                               self.adjacency_matrix.transpose(1, 2))


class GNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str = 'ReLU'):
        super(GNNLayer, self).__init__()
        self.message_layer = LinearWithMask(in_features=in_dim, out_features=out_dim)
        self.activation_fn = getattr(nn, activation)()

    def forward(self, x, mask):
        return self.activation_fn((torch.div(1.0, mask.sum(-1))) * torch.sum(self.message_layer(x, mask), dim=1))


class LinearWithMask(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        # simialar to https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearWithMask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Same as in https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear.
        :return:
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        input = input.repeat(mask.size()[1], 1, 1) * mask.permute(1, 2, 0)  # dropout the masked data
        output = input @ self.weight.t() + (self.bias * mask.permute(1, 2, 0))  # dropout respective bias data
        return output
