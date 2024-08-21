from copy import copy
from scipy.stats import pearsonr
import numpy as np


def rmse(preds, reals):
    return np.sqrt(np.mean((preds - reals) ** 2))  #copied from stack overflow


def corr(preds, reals):
    if any(np.isnan(preds)) or any(np.isinf(preds)):
        return np.PINF
    r = pearsonr(preds, reals)[0]
    if np.isnan(r):
        r = 0
    return 1 - r ** 2


def align(preds, reals):
    if not all(np.isfinite(preds)):
        return 1.0, 0.0
    try:
        align = np.round(np.polyfit(preds, reals, 1, rcond=1e-16), decimals=14)
    except:
        return 1.0, 0.0
    a = align[0]
    b = align[1]
    #print(f'align {align}')
    return (a, b)


class Fitness:
    def __init__(self, data, bias, target, pop, func, bank, n_inp, max_d=4, fit_function=corr, arity=2):
        self.data = data.reshape((-1, n_inp))
        self.bias = np.array(bias)
        self.target = target
        self.pop = pop
        self.func = func
        self.fit = fit_function
        self.bank = bank
        self.arity = arity
        self.max_d = max_d
        self.n_inp = n_inp
        self.n_bias = self.bias.shape[-1]
        self.data_bias = np.zeros((self.data.shape[0], self.bias.shape[-1] + n_inp))
        self.data_bias[:, :n_inp] = self.data
        self.data_bias[:, n_inp:] = self.bias

    def run(self, individual):
        preds = np.zeros((len(self.target),))
        for i in range(len(self.data)):
            registers = np.zeros((1 + self.n_inp + self.n_bias + self.max_d,))
            registers[1:self.n_inp + self.n_bias + 1] = self.data_bias[i, :]
            #registers[n_inp+1, n_bias+1] = bias
            for j in range(len(individual)):
                operation = individual[j].astype(int)
                destination = operation[0]
                operator = self.bank[operation[1]]
                sources = operation[2:]
                registers[destination] = operator(copy(registers[sources[0]]), copy(registers[sources[1]]))
            preds[i] = registers[0]
        #print(train_y
        (a, b) = align(preds, self.target)
        preds = preds * a + b
        return self.fit(preds, self.target), a, b

    def predict(self, individual, a, b, n_inp, test):
        preds = np.zeros((len(test),))
        train_x_bias = np.zeros((test.shape[0], self.bias.shape[0] + n_inp))
        train_x_bias[:, :n_inp] = test
        train_x_bias[:, n_inp:] = self.bias
        for i in range(len(test)):
            registers = np.zeros((1 + self.n_inp + self.n_bias + self.max_d,))
            registers[1:self.n_inp + self.n_bias + 1] = train_x_bias[i, :]
            #registers[n_inp+1, n_bias+1] = bias
            for j in range(len(individual)):
                operation = individual[j].astype(int)
                destination = operation[0]
                operator = self.bank[operation[1]]
                sources = operation[2:]
                registers[destination] = operator(registers[sources[0]], registers[sources[1]])
            preds[i] = registers[0]
        return preds * a + b

    def __call__(self, pop=None):
        if pop is None:
            pop = self.pop
        fitnesses = []
        A = []
        B = []
        for ind in pop:
            with np.errstate(invalid='raise'):
                try:
                    f = self.run(ind)
                    v = f[0]
                    a = f[1]
                    b = f[2]
                    fitnesses.append(v)
                    A.append(a)
                    B.append(b)
                except (OverflowError, FloatingPointError):
                    fitnesses.append(np.nan)
                    A.append(1.0)
                    B.append(0.0)
        return np.array(fitnesses), np.array(A), np.array(B)


class FitCollection:
    def __init__(self):
        self.fit_list = [rmse, corr]
        self.name_list = ['RMSE', '1-R^2']
