import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


class GDA:
    def __init__(self, samples: pd.DataFrame):
        self.samples_len = len(samples)
        self.phi = (samples.loc[samples['labels'] == 1].sum().loc['labels'] / self.samples_len)
        samples_label_grouped = samples.groupby('labels')
        samples_vectors_0 = samples_label_grouped.get_group(0).loc[:, ['samples_vectors']]
        samples_vectors_1 = samples_label_grouped.get_group(1).loc[:, ['samples_vectors']]
        self.means = [samples_vectors_0.to_numpy().mean(), samples_vectors_1.to_numpy().mean()]
        self.covariance_matrix = self.calculate_covariance_matrix(samples_vectors_0=samples_vectors_0.values.tolist(),
                                                                  samples_vectors_1=samples_vectors_1.values.tolist())

    def calculate_covariance_matrix(self, samples_vectors_0: np.array, samples_vectors_1: np.array):
        adjusted_samples_vector_0 = np.subtract(np.concatenate(samples_vectors_0, axis=0), self.means[0])
        adjusted_samples_vector_1 = np.subtract(np.concatenate(samples_vectors_1, axis=0), self.means[1])
        return (1/(self.samples_len * 2)) * \
               (np.matmul(adjusted_samples_vector_0.transpose(), adjusted_samples_vector_0) +
                np.matmul(adjusted_samples_vector_1.transpose(), adjusted_samples_vector_1))

    def get_class(self, input_vector: np.array, allow_singular=False):
        return np.argmax([
            multivariate_normal.pdf(x=input_vector, mean=self.means[0],
                                    cov=self.covariance_matrix, allow_singular=allow_singular) * (1 - self.phi),
            multivariate_normal.pdf(x=input_vector, mean=self.means[1],
                                    cov=self.covariance_matrix, allow_singular=allow_singular) * self.phi
        ])

    def get_probability(self, input_vector: np.array, label: int, allow_singular=False):
        return (multivariate_normal.pdf(x=input_vector, mean=self.means[label],
                                        cov=self.covariance_matrix, allow_singular=allow_singular)
                * ((1 - (label % 2)) - ((1 - ((label % 2) * 2)) * self.phi))) \
               / ((multivariate_normal.pdf(x=input_vector, mean=self.means[0],
                                           cov=self.covariance_matrix, allow_singular=allow_singular) * (1 - self.phi))
                  + (multivariate_normal.pdf(x=input_vector, mean=self.means[1],
                                             cov=self.covariance_matrix, allow_singular=allow_singular) * self.phi))
