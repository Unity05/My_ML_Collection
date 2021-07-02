from abc import ABC, abstractmethod


class DFS(ABC):
    def __init__(self, graph: list[list]):
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

    def dfs(self, u: int, v: int, e: list[int]):
        for edge_target in e:
            if self.marked[edge_target]:
                self.traverse_non_tree_edge(v=v, q=edge_target)
            else:
                self.traverse_tree_edge(v=v, q=edge_target)
                self.marked[edge_target] = True
                self.dfs(u=v, v=edge_target, e=self.graph[edge_target])
        self.backtrace(u=u, v=v)
