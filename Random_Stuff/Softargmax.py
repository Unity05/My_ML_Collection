import numpy as np


def softargmax(values: np.ndarray, beta: float) -> np.ndarray:
    return np.sum(np.exp(beta * values) / np.sum(np.exp(beta * values)) * values)

