import warnings

from scipy.stats import pearsonr

from cgp_operators import *


def rmse(preds, reals):
    return np.sqrt(np.mean((preds - reals) ** 2))


def corr(preds, reals):
    if any(np.isnan(preds)) or any(np.isinf(preds)):
        return np.PINF
    r = pearsonr(preds, reals)[0]
    return 1 - r ** 2 if not np.isnan(r) else 1


def align(preds, reals):
    if not all(np.isfinite(preds)):
        return 1.0, 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            align_values = np.round(np.polyfit(preds, reals, 1, rcond=1e-16), decimals=14)
    except:
        return 1.0, 0.0
    return align_values[0], align_values[1]


def change(new, old):
    return (new - old) / old


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

        if not isinstance(self.individual, tuple) and isinstance(self.individual, np.ndarray):
            num_of_outputs = np.atleast_1d(target.ndim)[-1]
            length_of_node = self.arity + num_of_outputs
            output_nodes = np.atleast_1d(self.individual[-num_of_outputs])
            ind_base = np.atleast_1d(self.individual[:-num_of_outputs].reshape((-1, length_of_node)))
            self.individual = (ind_base, output_nodes)

        return self.fitness(data, target, opt)

    def run(self, cur_node, inp_nodes):
        try:
            inp_size = inp_nodes.shape[0]
            args = [inp_nodes[cur_node[j]] if cur_node[j] < inp_size else self.run(
                self.individual[0][cur_node[j] - inp_size], inp_nodes) for j in range(self.arity)]
            return self.bank[cur_node[-1]](*args)
        except RecursionError:
            print(self.individual)
            raise ValueError(f'Input Node = {cur_node}')
        except IndexError:
            print(cur_node)
            raise IndexError()

    def run_output(self, inp_nodes):
        out_nodes = np.atleast_1d(self.individual[1])
        inp_nodes = np.array(inp_nodes)
        outs = np.atleast_1d(np.zeros(out_nodes.shape))

        for i in range(outs.shape[0]):
            if out_nodes[i] < len(inp_nodes):
                outs[i] = inp_nodes[out_nodes[i]]
            else:
                try:
                    outs[i] = self.run(self.individual[0][out_nodes[i] - inp_nodes.shape[0]], inp_nodes)
                except IndexError:
                    print('index error')
                    print(self.individual)
                    print(inp_nodes)
                    print(self.individual[0])
                    print(inp_nodes.shape)
                    raise IndexError(
                        f'Asked for ind[{out_nodes[i] - inp_nodes.shape[0]}], len(ind) = {self.individual[0].shape}\ni = {i}, out_nodes = {out_nodes}, ind = {self.individual[0]}')

        return outs

    def fitness(self, data, targ, opt=0):
        data = np.atleast_1d(data)
        targ = np.atleast_1d(targ)
        out_x = np.zeros(data.shape[0])

        for x in range(data.shape[0]):
            in_val = [data[x]] if len(data.shape) <= 1 else data[x, :]
            with np.errstate(invalid='raise'):
                try:
                    out_x[x] = self.run_output(in_val)
                except (OverflowError, FloatingPointError):
                    out_x[x] = np.nan

        with np.errstate(invalid='raise'):
            try:
                a, b = align(out_x, targ)
            except (OverflowError, FloatingPointError):
                return np.nan, 1.0, 0.0

        new_x = out_x * a + b
        if opt == 1:
            return new_x, a, b
        return self.fit(new_x, self.target), a, b


class FitCollection:
    def __init__(self):
        self.fit_list = [rmse, corr]
        self.name_list = ['RMSE', '1-R^2']


# Add xover impact
class DriftImpact:
    def __init__(self, neutral_limit=0.1):
        self.mut_list = []
        self.mut_cum = np.array([0, 0, 0])
        self.xov_list = []
        self.xov_cum = np.array([0, 0, 0])
        self.neutral_limit = neutral_limit

    def __call__(self, fitnesses, max_p, xov_parents, mut_parents, option='TwoParent', children=4, opt=0):
        drift_mut = np.array([0, 0, 0])
        drift_xov = np.array([0, 0, 0])
        drift_per_parent_mut = []
        drift_per_parent_xov = []

        if option == 'TwoParent':
            for i in xov_parents:
                p = min(fitnesses[i], fitnesses[i + 1])
                c = min(fitnesses[i + max_p], fitnesses[i + 1 + max_p])
                drift_xov, drift_per_parent_xov = self.get_drift_category(c, drift_xov, drift_per_parent_xov, p)
            for i in mut_parents:
                p = fitnesses[i]
                c = fitnesses[i + max_p]
                drift_mut, drift_per_parent_mut = self.get_drift_category(c, drift_mut, drift_per_parent_mut, p)
        elif option == 'OneParent':  # only uses mutation
            for i in range(max_p):
                p = fitnesses[i]
                c = min(fitnesses[i * children: i * children + children])
                drift_mut, drift_per_parent_mut = self.get_drift_category(c, drift_mut, drift_per_parent_mut, p)

        self.mut_cum += drift_mut
        self.xov_cum += drift_xov
        self.mut_list.append(drift_mut.copy())

        if opt == 1:
            return drift_per_parent_mut, drift_per_parent_xov

    def get_drift_category(self, c, drift, drift_per_parent, p):
        change_val = change(c, p)
        if change_val > self.neutral_limit:
            drift[0] += 1
            drift_per_parent.append(0)
        elif change_val < -self.neutral_limit:
            drift[2] += 1
            drift_per_parent.append(2)
        else:
            drift[1] += 1
            drift_per_parent.append(1)

        return drift, drift_per_parent

    def returnLists(self, option=0):
        if option == 0:
            return self.mut_cum, self.mut_list, self.xov_cum, self.xov_list
        elif option == 1:
            return self.mut_cum / np.sum(self.mut_cum), np.divide(self.mut_list,
                                                                  np.sum(self.mut_list, axis=1,
                                                                         keepdims=True)), self.xov_cum / np.sum(
                self.xov_cum), np.divide(self.xov_list, np.sum(self.xov_list, axis=1, keepdims=True))
