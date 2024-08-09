import numpy as np
from numpy import random
from copy import deepcopy

class SAM:
    def __init__(self, epsilon=0.2, choice_prop=1.00):
        self.epsilon = np.abs(epsilon)
        self.choice_prop = choice_prop

class SAM_IN(SAM):
    def __init__(self, dataset, n_inp=1, epsilon=0.2, choice_prop=1.00):
        super().__init__(epsilon, choice_prop)
        self.n_inp = n_inp
        self.dataset = dataset if dataset.ndim == 2 else dataset.reshape(-1, 1)

    def get_std(self):
        return np.std(self.dataset[:, :self.n_inp], axis=0)

    def perturb_data(self, std=None):
        vec = deepcopy(self.dataset)
        reshaped = False
        if vec.ndim != 2:
            old_shape = vec.shape
            vec = vec.reshape(-1, vec.shape[-1])
            reshaped = True

        if std is None:
            std = self.get_std()

        sample_size = np.round(vec.shape[0] * self.choice_prop).astype(np.int32)
        samples = random.choice(vec.shape[0], sample_size, replace=False)
        for s in samples:
            for f in range(self.n_inp):
                vec[s, f] += random.uniform(-self.epsilon * std[f], self.epsilon * std[f])

        if reshaped:
            vec = vec.reshape(old_shape)
        return vec

    def perturb_constants(self):
        vec = deepcopy(self.dataset)
        reshaped = False
        if vec.ndim != 2:
            old_shape = vec.shape
            vec = vec.reshape(-1, vec.shape[-1])
            reshaped = True

        for s in range(vec.shape[0]):
            for f in range(self.n_inp, vec.shape[1]):
                vec[s, f] += random.uniform(-self.epsilon, self.epsilon)

        if reshaped:
            vec = vec.reshape(old_shape)
        return vec

class SAM_OUT(SAM):
    def __init__(self, epsilon=1.0):
        super().__init__(epsilon)

    def get_std(self, predictions, low_bound=1e-2):
        return max(np.std(predictions), low_bound)

    def perturb(self, predictions, num_neighbors=25):
        std = self.get_std(predictions)
        neighborhood = random.normal(0, self.epsilon * std, (num_neighbors, predictions.size))
        return neighborhood + predictions.flatten()

