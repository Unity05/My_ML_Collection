from abc import ABC, abstractmethod
import math
import numpy as np
import random as rnd


class RandomGraphCreator(ABC):
    """
    Creator class for random graphs.
    """

    def create_graph(self, **kwargs) -> np.array:
        """
        Creates a random graph.

        :return: the random graph.
        """

        # Simply return adjacency matrix for now.
        return self.graph_factorymethod(**kwargs)

    @abstractmethod
    def graph_factorymethod(self, **kwargs) -> np.array:
        pass


class ERGraph(RandomGraphCreator):
    """
    Creator class for Erdős–Rényi random graphs.
    """

    def __init__(self, n: int, p: float):
        """
        :param n: #nodes
        :param p: probability for an edge e
        """

        super(ERGraph, self).__init__()

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


class DegreeGraph(RandomGraphCreator):
    """
    Creator class for node degree based random graphs.
    """

    def __init__(self, degree_sequence: list):
        """
        :param degree_sequence: degree for each node
        """

        super(DegreeGraph, self).__init__()

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

        for i in nodes:
            for j in rnd.choices(population=nodes, weights=(degree_array > 0)):
                edges.update([(i, j), (j, i)])
                degree_array[i] -= 1

        edges = set(edges)
        adjacency_matrix[list(map(list, zip(*edges)))] = 1

        return adjacency_matrix


class CommunityAffiliationGraph(RandomGraphCreator):
    """
    Creator class for node community affiliation based random graphs.
    """

    def graph_factorymethod(self, **kwargs) -> np.array:
        """
        Generates a random undirected graph with specified communities.

        :param kwargs: should contain:
                                community_lookup: A (0/1) - tensor of shape (#nodes, #communities).
                                                  1 if the node belongs to the community, 0 otherwise.
                                community_p: A tensor of shape (#communities) containing for every community
                                             the probability that two nodes in a community are connected.
        :return: An adjacency matrix containing the edge probabilities.
        """

        edge_probs = np.einsum('ij,kj,j->ikj', kwargs['community_lookup'],
                               kwargs['community_lookup'], (1 - kwargs['community_p']))
        edge_probs = 1 - np.prod(edge_probs, axis=-1, where=(edge_probs != 0), initial=1.0)

        return edge_probs


class SmallWorldModel(RandomGraphCreator):
    """
    Creator class for Small - World Model random graphs.
    """

    def __init__(self, n: int, p: float):
        """
        :param n: #nodes
        :param p: probability for an edge in the regular network
                  to be replaced by an edge leading to a random node.
                  (Higher p means more randomness. Estimated 'optimal' range: 5.0e-3 < p < 1.0e-1.)
        """

        super(SmallWorldModel, self).__init__()

        self.n = n
        self.p = math.sqrt(p)

    def graph_factorymethod(self) -> np.array:
        """
        Generates a random undirected small world model graph.
        If p is well chosen, the random graph is an 'interpolation' between a regular network and a random network
        and has a high clustering coefficient as well as a low diameter or average shortest path length.

        :return: the random small world model graph (adjacency matrix).
        """

        regular_network = np.eye(N=self.n, M=self.n, k=1) \
                          + np.eye(N=self.n, M=self.n, k=2) \
                          + np.eye(N=self.n, M=self.n, k=-(self.n - 1)) \
                          + np.eye(N=self.n, M=self.n, k=-(self.n - 2))
        regular_network = regular_network + np.transpose(regular_network)
        relink_p = np.random.rand(self.n, self.n)
        for i in range(self.n):
            swap_indices = relink_p[i] <= self.p
            regular_network[i, swap_indices] = np.random.permutation(regular_network[i, swap_indices])

        return regular_network


class KroneckerGraph(RandomGraphCreator):
    """
    Creator class for Kronecker random graphs.
    Related paper: https://www.jmlr.org/papers/volume11/leskovec10a/leskovec10a.pdf
    Implementation based on this slide: http://snap.stanford.edu/class/cs224w-2020/slides/14-traditional-generation.pdf#page=57
    """

    def __init__(self, n: int):
        """
        :param n: #nodes
        """

        super(KroneckerGraph, self).__init__()

        self.n = n

    def graph_factorymethod(self, **kwargs) -> np.array:
        """
        Generates a random directed Kronecker graph
        by iteratively 'dropping' an edge into a graph's quadrants / sections.

        :param kwargs: should contain:
                                probability_matrix: (np.array) A probability matrix
                                                    of shape (N x N) for the quadrants / sections.
                                n_edges: (int) #edges.
        :return: (np.array) the random directed Kronecker graph (adjacency matrix).
        """

        adjacency_matrix = np.zeros((self.n, self.n))

        theta = kwargs['probability_matrix']
        N = theta.shape[1]      # N^2 number of 'quadrants'
        assert N**int(math.log(self.n, N)) == self.n, \
            'n must be a power of N for the quadrants / sections to work (as of now).'
        theta = theta / np.sum(theta)   # normalize probabilities
        theta = theta.flatten()     # 1d - probability array

        n_edges = kwargs['n_edges']

        m = int(math.log(self.n, N))     # Generalizing from 2^2 quadrants to N^N sections.

        for _ in range(n_edges):
            x = 0
            y = 0
            for i in range(1, (m + 1)):
                u, v = divmod(np.random.choice(a=theta.size, p=theta), N)
                # Descend into the quadrant (u, v) at level i.
                x += u * N**(m - i)
                y += v * N**(m - i)
            adjacency_matrix[x, y] = 1

        return adjacency_matrix
