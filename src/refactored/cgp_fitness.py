# main.py
import numpy as np
from utils import rmse, corr, align, change
from effProg import *  # Ensure the necessary imports from your modules
from cgp_operators import *  # Ensure the necessary imports from your modules

class Fitness:
    def __init__(self):
        self.data = None
        self.target = None
        self.individual = None
        self.fit = None
        self.bank = None
        self.arity = None

    def __call__(self, data, target, individual, fit_function=corr, bank=(add, sub, mul, div), arity=2, opt=0):
        self.data = data
        self.target = target
        self.individual = individual
        self.fit = fit_function
        self.bank = bank
        self.arity = arity
        self._reshape_individual_if_needed(target)
        return self.fitness(data, target, opt)

    def _reshape_individual_if_needed(self, target):
        if not isinstance(self.individual, tuple) and isinstance(self.individual, np.ndarray):
            num_of_outputs = np.atleast_1d(target.ndim)[-1]
            length_of_node = self.arity + num_of_outputs
            output_nodes = np.atleast_1d(self.individual[-num_of_outputs])
            ind_base = np.atleast_1d(self.individual[:-num_of_outputs].reshape((-1, length_of_node)))
            self.individual = (ind_base, output_nodes)

    def run(self, cur_node, inp_nodes):
        try:
            inp_size = inp_nodes.shape[0]
            args = [self.run(self.individual[0][cur_node[j] - inp_size], inp_nodes) if cur_node[j] >= inp_size else inp_nodes[cur_node[j]] for j in range(self.arity)]
            function = self.bank[cur_node[-1]]
            return function(*args)
        except RecursionError:
            print(self.individual)
            raise ValueError(f'Input Node = {cur_node}')
        except IndexError:
            print(cur_node)
            raise IndexError()

    def run_output(self, inp_nodes):
        out_nodes = np.atleast_1d(self.individual[1])
        inp_nodes = np.array(inp_nodes)
        outs = np.zeros(out_nodes.shape)
        for i in range(outs.shape[0]):
            if out_nodes[i] < len(inp_nodes):
                outs[i] = inp_nodes[out_nodes[i]]
            else:
                try:
                    outs[i] = self.run(self.individual[0][out_nodes[i] - inp_nodes.shape[0]], inp_nodes)
                except IndexError:
                    self._print_debug_info(out_nodes, inp_nodes, i)
                    raise IndexError(f'Asked for ind[{out_nodes[i] - inp_nodes.shape[0]}], len(ind) = {self.individual[0].shape}\ni = {i}, out_nodes = {out_nodes}')
        return outs

    def _print_debug_info(self, out_nodes, inp_nodes, i):
        print('Index error')
        print(self.individual)
        print(inp_nodes)
        print(self.individual[0])
        print(inp_nodes.shape)

    def fitness(self, data, targ, opt=0):
        data, targ = np.atleast_1d(data), np.atleast_1d(targ)
        out_x = np.array([self._evaluate_individual(data[x]) for x in range(data.shape[0])])
        try:
            a, b = align(out_x, targ)
        except (OverflowError, FloatingPointError):
            return np.nan, 1.0, 0.0
        new_x = out_x * a + b
        return (new_x, a, b) if opt == 1 else (self.fit(new_x, self.target), a, b)

    def _evaluate_individual(self, data_point):
        with np.errstate(invalid='raise'):
            try:
                return self.run_output([data_point] if len(data_point.shape) <= 1 else data_point)
            except (OverflowError, FloatingPointError):
                return np.nan

class FitCollection:
    def __init__(self):
        self.fit_list = [rmse, corr]
        self.name_list = ['RMSE', '1-R^2']

class MutationImpact:
    def __init__(self, neutral_limit=0.1):
        self.drift_list = []
        self.drift_cum = np.array([0, 0, 0])

    def __call__(self, fitnesses, max_p, option='TwoParent', children=4):
        drift = np.array([0, 0, 0])
        if option == 'TwoParent':
            self._evaluate_two_parent(fitnesses, max_p, drift)
        elif option == 'OneParent':
            self._evaluate_one_parent(fitnesses, max_p, children, drift)
        self.drift_cum += drift

