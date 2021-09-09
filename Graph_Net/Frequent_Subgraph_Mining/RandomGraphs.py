from abc import ABC, abstractmethod
import numpy as np
import random as rnd


class RandomGraphCreator(ABC):
    """
    Creator class for random graphs.
    """

    def create_graph(self) -> np.array:
        """
        Creates a random graph.

        :return: the random graph.
        """

        # Simply return adjacency matrix for now.
        return self.graph_factorymethod()

    @abstractmethod
    def graph_factorymethod(self) -> np.array:
        pass


class ERGraphCreator(RandomGraphCreator):
    """
    Creator class for Erdős–Rényi random graphs.
    """

    def __init__(self, n: int, p: float):
        """
        :param n: #nodes
        :param p: probability for an edge e
        """

        self.n = n
        self.p = p

    def graph_factorymethod(self) -> np.array:
        """
        Generates an ER random graph.
        More specific: an undirected graph without self - loops with n nodes
        and a probability p for an edge e to be included.

        :return: the ER random graph.
        """

        tri = np.tril(m=np.random.choice(a=[0, 1], size=(self.n, self.n), p=[(1 - self.p), self.p]), k=-1)
        return tri + np.transpose(tri)


class DegreeGraphCreator(RandomGraphCreator):
    """
    Creator class for node degree based random graphs.
    """

    def __init__(self, degree_sequence: list):
        """
        :param degree_sequence: degree for each node
        """

        self.degree_sequence = degree_sequence
        self.n = len(degree_sequence)

    def graph_factorymethod(self) -> np.array:
        """
        Generates a random undirected graph with specified node degrees (double edges are ignored).

        :return: the random graph.
        """
        adjacency_matrix = np.zeros(shape=(self.n, self.n))
        degree_array = np.array(self.degree_sequence)
        nodes = range(self.n)
        edges = set()

        def decrement(i):
            degree_array[i] -= 1

        [[(edges.update([(i, j), (j, i)]), decrement(j))
          for j in rnd.choices(population=nodes, weights=(degree_array > 0))] for i in nodes]

        edges = set(edges)
        adjacency_matrix[list(map(list, zip(*edges)))] = 1

        return adjacency_matrix
