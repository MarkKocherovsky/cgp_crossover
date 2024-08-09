from copy import deepcopy

import numpy as np
from numpy import random


#n_inp: number of inputs, default 1 (univariate), everything else in the dataset will be assumed to be constant
#epsilon: noise
#choice_prop: proportion to perturb
class SAM:
    def __init__(self, epsilon=0.2, choice_prop=1.00):
        self.epsilon = np.abs(epsilon)
        self.choice_prop = choice_prop


class SAM_IN(SAM):
    def __init__(self, dataset, n_inp=1, epsilon=0.2, choice_prop=1.00):
        self.n_inp = n_inp
        self.dataset = dataset
        super().__init__(epsilon, choice_prop)

    def get_std(self):
        if self.dataset.ndim == 1:  #1 vector:
            self.dataset = self.dataset.reshape((-1, 1))
        return np.std(self.dataset[:, :self.n_inp], axis=0)

    def perturb_data(self, std='None'):
        vec = deepcopy(self.dataset)
        try:
            assert vec.ndim == 2
            reshaped = False
        except AssertionError:
            old_shape = vec.shape
            vec = vec.reshape((-1, vec.shape[-1]))
            reshaped = True
        if std == 'None':
            std = self.get_std()

        total_size = vec.shape[0]  #total number of samples
        sample_size = np.round(total_size * self.choice_prop).astype(np.int32)
        samples = random.choice(np.arange(0, total_size, 1, dtype=np.int32), sample_size)
        for s in samples:
            for f in range(self.n_inp):  #does not include constants
                vec[s, f] = vec[s, f] + random.uniform(-1 * self.epsilon * std[f], 1 * self.epsilon * std[f])
        if reshaped:
            vec = vec.reshape(old_shape)
        return vec

    def perturb_constants(self):
        vec = deepcopy(self.dataset)
        try:
            assert vec.ndim == 2
            reshaped = False
        except AssertionError:
            old_shape = vec.shape
            vec = vec.reshape((-1, vec.shape[-1]))
            reshaped = True
        total_size = vec.shape[0]  #total number of samples

        for s in range(total_size):
            for f in range(self.n_inp, vec.shape[1]):  #does not include constants
                vec[s, f] = vec[s, f] + random.uniform(-1 * self.epsilon, 1 * self.epsilon)
        if reshaped:
            vec = vec.reshape(old_shape)
        return vec


class SAM_OUT(SAM):
    def __init__(self, epsilon=1.0):
        super().__init__(epsilon)

    def _get_std(self, predictions, low_bound=1e-2):
        return max(np.std(predictions), low_bound)

    def perturb(self, predictions, num_neighbors=25):
        std = self._get_std(predictions)
        neighborhood = random.normal(0, self.epsilon * std, (num_neighbors, predictions.flatten().shape[-1]))
        #print('Generated Perturbations')
        #print(neighborhood)
        for n in range(num_neighbors):
            neighborhood[n] += predictions.flatten()
        return neighborhood
