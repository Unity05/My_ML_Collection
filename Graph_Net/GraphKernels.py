import numpy as np
from collections import defaultdict


class ColorHashTable:
    def __init__(self, *args):
        self.hash_table = defaultdict(args[0])
        self.next_color = -1

    def get_color(self, color_code: str):
        print(self.hash_table)
        if color_code in self.hash_table:
            return self.hash_table[color_code]
        else:
            self.next_color += 1
            self.hash_table[color_code] = self.next_color
            return self.next_color


class WeisfeilerLehman:

    @staticmethod
    def graph_coloring(graph, n_steps: int):
        n_nodes = graph.shape[0]
        colors = np.empty((n_steps, n_nodes), dtype=np.int)
        hash_table = ColorHashTable(int)
        WeisfeilerLehman.time_coloring(step=(n_steps - 1), graph=graph, n_nodes=n_nodes,
                                       colors=colors, hash_table=hash_table)

    @staticmethod
    def time_coloring(step: int, graph: np.ndarray, n_nodes: int, colors, hash_table):
        if step == 0:
            colors[0, :] = np.ones(n_nodes, dtype=np.int)
            return
        WeisfeilerLehman.time_coloring(step=(step - 1), graph=graph, n_nodes=n_nodes,
                                       colors=colors, hash_table=hash_table)
        for node_i in range(n_nodes):
            colors[step, node_i] = hash_table.get_color(''.join(str(x) for x in colors[(step - 1),
                                                                                       graph[node_i].astype(bool)]))
