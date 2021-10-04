from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn


class GNN(nn.Module):
    """
    Simple example GNN with a few GNN Layers stacked sequentially.
    No attention, residual connections, etc.
    """

    def __init__(self, depth: int, embed_dim: int, activation: str = 'ReLU'):
        """
        :param depth: Number of hops that should be included in a node embedding. Equivalent to number of GNN Layers.
        :param embed_dim: Node embedding dimension.
        :param activation: Activation function. Default: ReLU.
        """

        super(GNN, self).__init__()
        self.depth = depth
        self.layers = [GNNLayer(in_dim=embed_dim, out_dim=embed_dim, activation=activation) for _ in range(depth)]
        self.h = None
        self.adjacency_matrix = None

    def forward(self, adjacency_matrix: torch.Tensor, graph: torch.Tensor):
        """
        :param adjacency_matrix: Adjacency matrix of graph.
        :param graph: Initial embeddings of nodes of graph.
        """

        self.adjacency_matrix = adjacency_matrix
        # embed matrix of shape [Depth; N; #Nodes; embed_dim]
        self.h = torch.empty((self.depth, graph.size()[0], graph.size()[1], graph.size()[2]))
        self.h[0] = graph
        self.calculate_embedding(remaining_its=(self.depth - 1))

    def calculate_embedding(self, remaining_its: int):
        """
        Computes next step node embeddings.

        :param remaining_its: Number of steps left to compute.
        """

        if remaining_its != 0:
            self.calculate_embedding(remaining_its=(remaining_its - 1))
            self.h[remaining_its] = self.layers[remaining_its](self.h[remaining_its - 1],
                                                               self.adjacency_matrix.transpose(1, 2))
            # transposing the adjacency matrix should not matter as we now are only interested in undirected graphs.


class GNNLayer(nn.Module):
    """
    GNN Layer.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str = 'ReLU'):
        """
        :param in_dim: Input embedding dimension.
        :param out_dim: Output embedding dimension.
        :param activation: Activation function. Default: ReLU.
        """

        super(GNNLayer, self).__init__()
        self.message_layer = LinearWithMask(in_features=in_dim, out_features=out_dim)
        self.activation_fn = getattr(nn, activation)()

    def forward(self, x, mask):
        """
        :param x: Previous embeddings.
        :param mask: Embedding / node mask.
        :return: New calculated embeddings.
        """

        return self.activation_fn(torch.einsum('i, ij->ij',
                                               torch.div(1.0, mask.sum(-2)),
                                               torch.sum(self.message_layer(x, mask), dim=1)))


class LinearWithMask(nn.Module):
    """
    Linear Layer where only non - masked input data is used for linear transformation.
    This is especially / primarily designed for GNN usage
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        """
        Simialar to https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear.

        :param in_features: size of each input sample (see pytorch)
        :param out_features: size of each output sample (see pytorch)
        :param bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``. (see pytorch)
        :param device: Specific device to run operations on. Default: None.
        :param dtype: Specific dtype. Default: None.
        """

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
        """
        :param input: (N, *, H_in) where * means in GNN context #Nodes and H_in = in_features.
        :param mask: (N, *, *) where * means in GNN context #Nodes
                     and masks all nodes that should not be used for each node.
        :return: (N, *, *, H_out) where * means in GNN context #Nodes and H_out = out_features.
        """

        m = mask.unsqueeze(-1)
        input = input.repeat(mask.size()[1], 1, 1) * m  # dropout the masked data
        output = input @ self.weight.t() + (self.bias * m)  # dropout respective bias data
        return output


class Attention(nn.Module, ABC):
    """
    Abstract Attention class.
    Using template design pattern.
    """

    @abstractmethod
    def calculate_attention_coefficient(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates attention coefficients for every embedding pair.

        :param embeddings: Embedding tensor of shape (N, #Embeds, embed_dim).
        :return: Tensor of shape (N, #Embeds, #Embeds, 1) containing all attention coefficients.
        """
        pass

    def calculate_attention_weight(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates attention weights for every embedding pair.

        :param embeddings: Embedding tensor of shape (N, #Embeds, embed_dim).
        :return: Tensor of shape (N, #Embeds, #Embeds, 1) containing all attention weights.
        """

        return torch.softmax(self.calculate_attention_coefficient(embeddings=embeddings), dim=2)


class LinearAttention(Attention):
    """
    Attention Layer using Linear Layer to compute attention coefficients.
    """

    def __init__(self, embed_dim: int) -> None:
        """
        :param embed_dim: Embedding dimension.
        """

        super(Attention, self).__init__()
        self.linear_layer = nn.Linear(in_features=(2 * embed_dim), out_features=1)

    def calculate_attention_coefficient(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates attention coefficients for every embedding pair.
        The embeddings are first concatenated and then transformed by a linear layer.

        :param embeddings: Embedding tensor of shape (N, #Embeds, embed_dim).
        :return: Tensor of shape (N, #Embeds, #Embeds, 1) containing all attention coefficients.
        """

        embeddings = torch.stack(([embeddings] * embeddings.shape[1]), dim=1)   # (N, #Embeds, #Embeds, embed_dim)
        return self.linear_layer(torch.cat((embeddings, embeddings.transpose(1, 2)), dim=-1))
