from abc import ABC, abstractmethod
import numpy as np


class DFS(ABC):
    """
    Skeleton class for Depth First Search.
    """

    def __init__(self, graph: list):
        self.graph = graph
        self.marked = [False] * len(graph)
        self.init()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def root(self, s: int):
        pass

    @abstractmethod
    def traverse_non_tree_edge(self, v: int, q: int):
        pass

    @abstractmethod
    def traverse_tree_edge(self, v: int, q: int):
        pass

    @abstractmethod
    def backtrace(self, u: int, v: int):
        pass

    @abstractmethod
    def get_return_data(self) -> tuple:
        pass

    def run(self) -> tuple:
        for s, e in enumerate(self.graph):
            if not self.marked[s]:
                self.marked[s] = True
                self.root(s=s)
                self.dfs(u=s, v=s, e=e)
        return self.get_return_data()

    def dfs(self, u: int, v: int, e: list):
        for edge_target in e:
            if self.marked[edge_target]:
                self.traverse_non_tree_edge(v=v, q=edge_target)
            else:
                self.traverse_tree_edge(v=v, q=edge_target)
                self.marked[edge_target] = True
                self.dfs(u=v, v=edge_target, e=self.graph[edge_target])
        self.backtrace(u=u, v=v)


class BFS:
    """
    Basic class for Breadth First Search.
    """

    def __init__(self, graph: list):
        self.graph = graph
        self.init()

    def init(self):
        self.d = np.full(len(self.graph), np.inf)
        self.parent = np.full(len(self.graph), -1)

    def traverse_non_tree_edge(self, u: int, v: int):
        pass

    def get_return_data(self):
        return self.parent, self.d

    def run(self, s: int) -> tuple:
        Q = [s]
        R = []
        length = 0
        while len(Q) > 0:
            for u in Q:
                for edge_target in self.graph[u]:
                    if self.parent[edge_target] == -1:
                        R.append(edge_target)
                        self.d[edge_target] = length + 1
                        self.parent[edge_target] = u
                    else:
                        self.traverse_non_tree_edge(u=u, v=edge_target)
            Q = R
            R = []

        return_data = self.get_return_data()
        self.init()
        return return_data
