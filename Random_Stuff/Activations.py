import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp((-1) * x))


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def softargmax(x: np.ndarray, beta: float) -> np.ndarray:
    return np.matmul(
        softmax(beta * x),
        np.transpose(np.arange(x.shape[0] - 1))
    )
