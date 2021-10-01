import torch
import torch.nn as nn

from typing import Optional


class GraphRNN(nn.Module):
    """
    Recurrent network for generating graphs as sequence generation.
    Roughly, it is based on a node level GRU and an edge level GRU.
    Related paper: https://arxiv.org/pdf/1802.08773.pdf
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers_node: int, num_layers_edge: int):
        """
        :param input_size: The number of expected features in the input x to the node level rnn.
        :param hidden_size: The number of expected features in the hidden state h and input x to the edge level rnn.
        :param num_layers_node: Number of recurrent layers for the node level rnn.
        :param num_layers_edge: Number of recurrent layers for the edge level rnn.
        """

        super(GraphRNN, self).__init__()

        self.num_layers_node = num_layers_node
        self.hidden_size = hidden_size

        self.node_level = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers_node)
        self.edge_level = nn.GRU(input_size=hidden_size, hidden_size=input_size, num_layers=num_layers_edge)
        self.edge_regression = nn.Sequential(
            nn.Linear(input_size, (input_size // 2)),
            nn.Linear((input_size // 2), 1)
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param input: Tensor (1). SOS tensor.
        :param hidden: Tensor (num_layers_node, N, H_out). Initial hidden state. Defaults to zero if not provided.
        :param target: Tensor (L_S). Containing the target sequence to generate the graph.
        :return: Tensor (L_S). Containing the models predicted sequence for generating the graph.
        """

        if hidden is None:  # Default initial hidden tensor containing zeros.
            hidden = torch.zeros(self.num_layers_node, input.size(1), self.hidden_size)

        prediction = torch.empty_like(target)
        edge_output = input
        current_lowest_node = 0
        node_i = 1
        i = 0
        while True:
            node_output, node_hidden = self.node_level(edge_output, hidden)
            edge_hidden = node_output
            for node_j in range(current_lowest_node, node_i - 1):
                edge_output, edge_hidden = self.edge_level(edge_output, edge_hidden)
                edge_output = self.edge_regression(edge_output)
                if self.training:
                    prediction[i] = edge_output
                    edge_output = target[i]     # teacher forcing; target contains all steps sequentially.
                else:
                    edge_output = torch.bernoulli(edge_output)  # flipping a coin --> 0 / 1.
                    prediction[i] = edge_output
                # Reducing the node space to save computation time.
                # Input data must be BFS adjusted to make this work,
                # since we assume none of the future nodes connect to nodes prior to current_lowest_node.
                if edge_output.item() == 0 and node_j == current_lowest_node:
                    current_lowest_node += 1
            if current_lowest_node >= (node_i - 1):     # EOS.
                break

        return prediction
