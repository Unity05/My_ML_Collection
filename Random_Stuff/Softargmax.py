import numpy as np


def softargmax(values: np.ndarray, beta: float) -> np.ndarray:
    return np.matmul(
        np.exp(beta * values) / np.sum(np.exp(beta * values)),
        np.transpose(np.arange(values.shape[0] - 1))
    )
