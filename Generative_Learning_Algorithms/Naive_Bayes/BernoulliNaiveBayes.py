import numpy as np
import pandas as pd


class BernoulliNaiveBayes:
    def __init__(self, samples: pd.DataFrame, la_place_smoothing=True):
        self.vocab_size = samples['sample_vectors'].loc[0].size
        self.samples_len = len(samples)
        self.phi = (samples.loc[samples['labels'] == 1].sum().loc['labels'] / self.samples_len)
        self.phi_word_class = self.__calculate_estimates(samples=samples, la_place_smoothing=la_place_smoothing)

    def get_class(self, input_vector: np.array):
        return np.argmax([
            self.get_probability(input_vector=input_vector, label=0),
            self.get_probability(input_vector=input_vector, label=1)
        ])

    def get_probability(self, input_vector: np.array, label: int):
        phi = ((self.phi**label) * ((1 - self.phi)**(1 - label)))
        p = self.__p_x_given_y(input_vector=input_vector, label=label) * phi
        return p / (p + (self.__p_x_given_y(input_vector=input_vector, label=(1 - label)) * (1 - phi)))

    def __calculate_estimates(self, samples: pd.DataFrame, la_place_smoothing: bool):
        samples_zero = samples.loc[samples['labels'] == 0]
        samples_one = samples.loc[samples['labels'] == 1]
        la_place_smoothing_x = int(la_place_smoothing)
        return np.array([
            ((samples_zero['sample_vectors'].sum() + la_place_smoothing_x)
             / (len(samples_zero) + (la_place_smoothing_x * 2))),
            ((samples_one['sample_vectors'].sum() + la_place_smoothing_x)
             / (len(samples_one) + (la_place_smoothing_x * 2)))
        ])

    def __p_x_given_y(self, input_vector: np.array, label: int):
        probability = 1
        for i in range(self.vocab_size):
            # applying Bernoulli
            probability *= (self.phi_word_class[label][i]**input_vector[i]) \
                           * ((1 - self.phi_word_class[label][i])**(1 - input_vector[i]))
        return probability
