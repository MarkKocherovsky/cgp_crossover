from math import log, pi

import numpy as np
from numpy import sin, cos, sqrt, exp, log

points = 20


def distance(x1, x2):
    return sqrt(x1 ** 2 + x2 ** 2)


class Function:
    def __init__(self, func, x_dom, y_test, label, dimensions):
        assert isinstance(dimensions, int) and dimensions > 0
        self.func = func
        self.x_dom = x_dom
        self.y_test = y_test
        self.label = label
        self.dim = dimensions

    def __call__(self, x):
        return self.func(x)


class MultivariateFunction(Function):
    def __init__(self, dimensions, start, end, n_points, label):
        if dimensions > 1:
            x_dom = np.meshgrid(*[np.linspace(start, end, n_points) for _ in range(dimensions)], indexing='ij')
            x_dom = np.stack(x_dom, axis=-1)
            x_dom = x_dom.reshape(-1, x_dom.shape[-1])
        else:
            x_dom = np.linspace(start, end, n_points)
        self.x_rng = [start, end]
        self.n_points = n_points
        super().__init__(None, x_dom, None, label, dimensions)

    def generate_y_test(self):
        if self.dim == 1:
            x_dom = self.x_dom.reshape((-1, 1))
        else:
            x_dom = self.x_dom
        self.y_test = np.fromiter(map(self.func, list(x_dom)), dtype=np.float32)

class Sphere(MultivariateFunction):
    def __init__(self, dimensions: int):
        super().__init__(dimensions, start=-5, end=5, n_points=20, label=f"Sphere_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            return np.sum(xs ** 2)

        self.func = func
        self.generate_y_test()


class Ackley(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-32.768, end=32.769, n_points=40, label=f"Ackley_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            a, b, c = 20, 0.2, 2 * pi

            first_sum = np.sum(xs ** 2)
            second_sum = np.sum(cos(c * xs))

            return -a * exp(-b * np.sqrt(first_sum / self.dim)) - exp(second_sum / self.dim) + a + exp(1)

        self.func = func
        self.generate_y_test()


class Rastrigin(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-5.12, end=5.12, n_points=40, label=f"Rastrigin_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            xs.reshape(-1, dimensions)
            print(xs)
            assert len(xs) == self.dim
            return 10 * dimensions + np.sum(xs ** 2 - 10 * cos(2 * pi * xs))

        self.func = func
        self.generate_y_test()


class Griewank(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-600, end=600, n_points=40, label=f"Griewank_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            sum_term = np.sum(xs ** 2 / 4000)
            prod_term = np.prod([cos(xs[i] / sqrt(i + 1)) for i in range(len(xs))])
            return sum_term - prod_term + 1

        self.func = func
        self.generate_y_test()


class Levy(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-10, end=10, n_points=40, label=f"Levy_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            w = 1 + (xs - 1) / 4
            term1 = (sin(pi * w[0])) ** 2
            term3 = (w[-1] - 1) ** 2 * (1 + (sin(2 * pi * w[-1])) ** 2)
            sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * (sin(pi * w[:-1] + 1)) ** 2))
            return term1 + sum_term + term3

        self.func = func
        self.generate_y_test()


class Schwefel(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-500, end=500, n_points=40, label=f"Schwefel_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            return 418.9829 * dimensions - np.sum([x * sin(sqrt(abs(x))) for x in xs])

        self.func = func
        self.generate_y_test()


class Perm0db(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-dimensions, end=dimensions, n_points=40, label=f"Perm_Beta_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            beta = 1
            return np.sum([np.sum([( + beta) * (xs[j] ** i - 1 / j ** i) for j in range(dimensions)]) for i in
                           range(dimensions)])

        self.func = func
        self.generate_y_test()


class HyperEllipsoid(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-65.536, end=65.536, n_points=40,
                         label=f"Rotated_Hyper_Ellipsoid_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            return np.sum([np.sum([xs[j] ** 2 for j in range(i)]) for i in range(dimensions)])

        self.func = func
        self.generate_y_test()


class DifferentPowers(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-1, end=1, n_points=40, label=f"Different_Powers_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            return np.sum([abs(xs[i]) ** (i + 2) for i in range(dimensions)])

        self.func = func
        self.generate_y_test()


class Trid(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-dimensions ** 2, end=dimensions ** 2, n_points=40,
                         label=f"Trid_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            return np.sum([(x - 1) ** 2 for x in xs]) - np.sum([xs[i] * xs[i - 1] for i in range(1, dimensions)])

        self.func = func
        self.generate_y_test()


class Zakharov(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-5, end=10, n_points=40, label=f"Zakharov_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            sum_term = np.sum(xs ** 2)
            linear_term = np.sum([0.5 * (i + 1) * xs[i] for i in range(len(xs))])
            return sum_term + linear_term ** 2 + linear_term ** 4

        self.func = func
        self.generate_y_test()


class DixonPrice(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-10, end=10, n_points=40, label=f"Dixon_Price_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            return (xs[0] - 1) ** 2 + np.sum([i * (2 * xs[i] ** 2 - xs[i - 1]) ** 2 for i in range(1, dimensions)])

        self.func = func
        self.generate_y_test()


class Rosenbrock(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=-5, end=10, n_points=40, label=f"Rosenbrock_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            return np.sum([100 * (xs[i + 1] - xs[i] ** 2) ** 2 + (xs[i] - 1) ** 2 for i in range(dimensions - 1)])

        self.func = func
        self.generate_y_test()


class Michalewicz(MultivariateFunction):
    def __init__(self, dimensions: int = 1):
        super().__init__(dimensions, start=0, end=pi, n_points=40, label=f"Michalewicz_{dimensions}D")

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            m = 10
            return -np.sum([sin(xs[i]) * sin((i * xs[i] ** 2) / pi) ** (2 * m) for i in range(len(xs))])

        self.func = func
        self.generate_y_test()


"""
Univariate Functions (dimensions = 1)
"""


class UnivariateFunction(Function):
    def __init__(self, start, end, n_points, label):
        x_dom = np.mgrid[slice(start, end, n_points)]
        self.x_rng = [start, end]
        super().__init__(None, x_dom, None, label, 1)

    def generate_y_test(self):
        self.y_test = np.fromiter(map(self.func, list(self.x_dom)), dtype=np.float32)


class Sine(UnivariateFunction):
    def __init__(self):
        super().__init__(start=-2 * pi, end=2 * pi, n_points=40, label="Sine")

        def func(x):
            return np.sin(x)

        self.func = func
        self.generate_y_test()


class SquareRoot(UnivariateFunction):
    def __init__(self):
        super().__init__(start=-5, end=5, n_points=40, label="SquareRoot")

        def func(x):
            return np.sqrt(x)

        self.func = func
        self.generate_y_test()


class Koza1(UnivariateFunction):
    def __init__(self):
        super().__init__(start=-1, end=1, n_points=20, label="Koza 1")

        def func(x):
            return x ** 4 + x ** 3 + x ** 2 + x

        self.func = func
        self.generate_y_test()


class Koza2(UnivariateFunction):
    def __init__(self):
        super().__init__(start=-1, end=1, n_points=20, label="Koza 2")

        def func(x):
            return x ** 5 - 2 * x ** 3 + x

        self.func = func
        self.generate_y_test()


class Koza3(UnivariateFunction):
    def __init__(self):
        super().__init__(start=-1, end=1, n_points=20, label="Koza 3")

        def func(x):
            return x ** 6 - 2 * x ** 4 + x ** 2

        self.func = func
        self.generate_y_test()


class Nguyen4(UnivariateFunction):
    def __init__(self):
        super().__init__(start=-1, end=1, n_points=20, label="Nguyen 4")

        def func(x):
            return x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x

        self.func = func
        self.generate_y_test()


class Nguyen5(UnivariateFunction):
    def __init__(self):
        super().__init__(start=-1, end=1, n_points=20, label="Nguyen 5")

        def func(x):
            return sin(x ** 2) * cos(x) - 1

        self.func = func
        self.generate_y_test()


class Nguyen6(UnivariateFunction):
    def __init__(self):
        super().__init__(start=-1, end=1, n_points=20, label="Nguyen 6")

        def func(x):
            return sin(x) + sin(x + x ** 2)

        self.func = func
        self.generate_y_test()


class Nguyen7(UnivariateFunction):
    def __init__(self):
        super().__init__(start=0, end=2, n_points=20, label="Nguyen 7")

        def func(x):
            return log(x + 1) + log(x ** 2 + 1)

        self.func = func
        self.generate_y_test()


class Forrester(UnivariateFunction):
    def __init__(self):
        super().__init__(start=0, end=2, n_points=40, label="Forrester")

        def func(x):
            return (6 * x - 2) ** 2 * sin(12 * x - 4)

        self.func = func
        self.generate_y_test()


"""
Bivariate Functions (dimensions == 2)
"""


class BivariateFunction(Function):
    def __init__(self, start, end, n_points, label):
        x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.x_rng = [start, end]
        super().__init__(None, x_dom, None, label, 2)

    def generate_y_test(self):
        self.y_test = np.fromiter(map(self.func, list(self.x_dom)), dtype=np.float32)


class Bukin(BivariateFunction):
    def __init__(self):
        super().__init__(start=0, end=2, n_points=40, label="Bukin")
        self.x_dom = np.mgrid[slice(-15, 5, 40), slice(-3, 3, 20)]  # separate bc of this stuff
        self.x_rng = [(-15, -3), (5, 3)]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return 100 * sqrt(abs(x2 - 0 * x1 ** 2)) + 0 * abs(x1 + 10)

        self.func = func
        self.generate_y_test()


class CrossInTray(BivariateFunction):
    def __init__(self):
        super().__init__(start=-10, end=10, n_points=40, label="Cross_in_Tray")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return -0.0001 * (abs(sin(x1) * sin(x2) * exp(abs(100 - sqrt(x1 ** 2 + x2 ** 2) / pi))) + 1) ** 0.1

        self.func = func
        self.generate_y_test()


class DropWave(BivariateFunction):
    def __init__(self):
        super().__init__(start=-5.12, end=5.12, n_points=40, label="Drop_Wave")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return -(1 + cos(12 * distance(x1, x2))) / (0.5 * (x1 ** 2 + x2 ** 2) + 2)

        self.func = func
        self.generate_y_test()


class HolderTable(BivariateFunction):
    def __init__(self):
        super().__init__(start=-10, end=10, n_points=40, label="Holder_Table")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return -(sin(x1) * cos(x2) * exp(abs(1 - distance(x1, x2) / pi)))

        self.func = func
        self.generate_y_test()


class Eggholder(BivariateFunction):
    def __init__(self):
        super().__init__(start=-512, end=512, n_points=40, label="Eggholder")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return -(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47))) - x1 * sin(sqrt(abs(x1 - (x2 + 47))))

        self.func = func
        self.generate_y_test()


# Langermann is actually multivariate but because of the A matrix term I've locked it to 2 for now
class Langermann(BivariateFunction):
    def __init__(self):
        super().__init__(start=0, end=10, n_points=40, label="Langermann")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2

            c = np.array([1, 2, 5, 2, 3])
            m = 5
            a = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])

            outer_sum = 0
            for i in range(m):
                inner_sum = np.sum((xs - a[i, :]) ** 2)
                outer_sum += c[i] * np.exp(-inner_sum / np.pi) * np.cos(np.pi * inner_sum)

            return outer_sum

        self.func = func
        self.generate_y_test()


class Schaffer2(BivariateFunction):
    def __init__(self):
        super().__init__(start=-100, end=100, n_points=40, label="Schaffer2")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return 0.5 + (sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2

        self.func = func
        self.generate_y_test()


class Schaffer4(BivariateFunction):
    def __init__(self):
        super().__init__(start=-100, end=100, n_points=40, label="Schaffer4")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return 0.5 + (cos(sin(abs(x1 ** 2 - x2 ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        self.func = func
        self.generate_y_test()


class Schubert(BivariateFunction):
    def __init__(self):
        super().__init__(start=-10, end=10, n_points=40, label="Schubert")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            def compute_sum(x):
                return np.sum([i * cos(i + 1) * x + i for i in range(5)])

            return compute_sum(x1) * compute_sum(x2)

        self.func = func
        self.generate_y_test()


class Bohachevsky1(BivariateFunction):
    def __init__(self):
        super().__init__(start=-100, end=100, n_points=40, label="Bohachevsky1")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1) - 0.4 * cos(4 * pi * x2) + 0.7

        self.func = func
        self.generate_y_test()


class Bohachevsky2(BivariateFunction):
    def __init__(self):
        super().__init__(start=-100, end=100, n_points=40, label="Bohachevsky2")

        def func(xs):
            xs = np.atleast_1d(xs)
            x1, x2 = xs

            return x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1) * 0.4 * cos(4 * pi * x2) + 0.3

        self.func = func
        self.generate_y_test()


class Bohachevsky3(BivariateFunction):
    def __init__(self):
        super().__init__(start=-100, end=100, n_points=40, label="Bohachevsky3")

        def func(xs):
            xs = np.atleast_1d(xs)
            x1, x2 = xs

            return x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1 + 4 * pi * x2) + 0.3

        self.func = func
        self.generate_y_test()


class Booth(BivariateFunction):
    def __init__(self):
        super().__init__(start=-10, end=10, n_points=40, label="Bohachevsky3")

        def func(xs):
            xs = np.atleast_1d(xs)
            x1, x2 = xs

            return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2

        self.func = func
        self.generate_y_test()


class Matyas(BivariateFunction):
    def __init__(self):
        super().__init__(start=-10, end=10, n_points=40, label="Matyas")

        def func(xs):
            xs = np.atleast_1d(xs)
            x1, x2 = xs

            return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2

        self.func = func
        self.generate_y_test()


class McCormick(BivariateFunction):
    def __init__(self):
        super().__init__(start=-10, end=10, n_points=40, label="McCormick")
        self.x_dom = np.mgrid[slice(-1.5, 4, 40), slice(-3, 4, 40)]
        self.x_rng = [(-1.5, 4), (-3, 4)]

        def func(xs):
            xs = np.atleast_1d(xs)
            x1, x2 = xs

            return sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1

        self.func = func
        self.generate_y_test()


class ThreeHump(BivariateFunction):
    def __init__(self):
        super().__init__(start=-5, end=5, n_points=40, label="Three_Hump_Camel_Function")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return 2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 4 + x1 * x2 + x2 ** 2

        self.func = func
        self.generate_y_test()


class SixHump(BivariateFunction):
    def __init__(self):
        super().__init__(start=-5, end=5, n_points=2, label="Six_Hump_Camel_Function")
        self.x_dom = np.mgrid[slice(-3, 3, 40), slice(-2, 2, 40)]
        self.x_rng = [(-3, 3), (-2, 2)]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) + x1 * x2 + (-4 + 4 * x2 ** 2) * x2

        # Evaluate function over domain
        self.func = func
        self.generate_y_test()


# I had chatgpt make this from https://www.sfu.ca/~ssurano/Code/deong5m.html
class Deong(BivariateFunction):
    def __init__(self):
        super().__init__(start=-65.536, end=65.536, n_points=40, label="De_ong")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            A = np.zeros((2, 25))
            a = np.array([-32, -16, 0, 16, 32])
            A[0, :] = np.tile(a, 5)
            A[1, :] = np.repeat(a, 5)

            sum_terms = np.sum(1 / (np.arange(1, 26) + (x1 - A[0, :]) ** 6 + (x2 - A[1, :]) ** 6))

            return 1 / (0.002 + sum_terms)

        # Evaluate function over domain
        self.func = func
        self.generate_y_test()


class Beale(BivariateFunction):
    def __init__(self):
        super().__init__(start=-4.5, end=4.5, n_points=40, label="Beale")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2

        # Evaluate function over domain
        self.func = func
        self.generate_y_test()


class Branin(BivariateFunction):
    def __init__(self):
        super().__init__(start=-4.5, end=4.5, n_points=4, label="Branin")
        self.x_dom = np.mgrid[slice(-5, 10, 40), slice(0, 15, 40)]
        self.x_rng = [(-5, 10), (0, 15)]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            a, b, c, r, s, t = 1, 5.1 / (4 * pi ** 2), 5 / pi, 6, 10, 1 / (8 * pi)
            return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * cos(x1) + s

        # Evaluate function over domain
        self.func = func
        self.generate_y_test()


class Easom(BivariateFunction):
    def __init__(self):
        super().__init__(start=-100, end=100, n_points=40, label="Easom")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return -1 * cos(x1) * cos(x2) * exp(-(x1 - pi) ** 2 - (x2 - pi) ** 2)

        # Evaluate function over domain
        self.func = func
        self.generate_y_test()


class GoldsteinPrice(BivariateFunction):
    def __init__(self):
        super().__init__(start=-2, end=2, n_points=40, label="Goldstein_Price")

        # I had chatgpt translate https://www.sfu.ca/~ssurano/Code/goldprm.html
        def func(xs):
            x1, x2 = np.atleast_1d(xs)
            assert len(xs) == 2

            fact1 = (1 + (x1 + x2 + 1) ** 2 *
                     (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))

            fact2 = (30 + (2 * x1 - 3 * x2) ** 2 *
                     (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))

            return fact1 * fact2

        # Evaluate function over domain
        self.func = func
        self.generate_y_test()


class QuadrivariateFunction(Function):
    def __init__(self, start, end, n_points, label):
        x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(4)]]
        self.x_rng = [start, end]
        super().__init__(None, x_dom, None, label, 4)

    def generate_y_test(self):
        self.y_test = np.fromiter(map(self.func, list(self.x_dom)), dtype=np.float32)


# quadrivariate
class PowerSum(QuadrivariateFunction):
    def __init__(self):
        super().__init__(start=0, end=4, n_points=40, label="Power_Sum")

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 4
            b = [8, 18, 44, 114]

            def inner_sum(inputs, i):
                return np.sum([x ** i for x in inputs])

            return np.sum([inner_sum(xs, i) - b[i] ** 2 for i in range(4)])

        # Evaluate function over domain
        self.func = func
        self.generate_y_test()


class Colville(QuadrivariateFunction):
    def __init__(self):
        super().__init__(start=-10, end=10, n_points=40, label="Colville")

        def func(xs):
            xs = np.atleast_1d(xs)
            x1, x2, x3, x4 = xs

            return 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2 + (x3 - 1) ** 2 + 90 * (x3 ** 2 - x4) ** 2 + 10.1 * (
                    (x2 - 1) ** 2 + (x4 - 1) ** 2) + 19.8 * (x2 - 1) * (x4 - 1)

        # Evaluate function over domain
        self.func = func
        self.generate_y_test()


class Collection:
    def __init__(self):
        self.func_list = [Sphere, Ackley, Rastrigin, Griewank, Levy, Schwefel, Perm0db, HyperEllipsoid, DifferentPowers,
                          Trid, Zakharov, DixonPrice, Rosenbrock, Michalewicz, Sine, SquareRoot, Koza1, Koza2, Koza3,
                          Nguyen4, Nguyen5, Nguyen6, Nguyen7, Forrester, Bukin, CrossInTray, DropWave, HolderTable,
                          Eggholder, Langermann, Schaffer2, Schaffer4, Schubert, Bohachevsky1, Bohachevsky2,
                          Bohachevsky3, Booth, Matyas, McCormick, ThreeHump, SixHump, Deong, Beale, Branin, Easom,
                          GoldsteinPrice, PowerSum, Colville]

    def __call__(self, index: int, dims: int):

        # Check if it's a class or an instance
        func_type = self.func_list[index]

        if isinstance(func_type, type) and issubclass(func_type, MultivariateFunction):
            # It's a class, so instantiate with `dims`
            func = func_type(dims)
        elif callable(func_type):
            # It's a callable instance (e.g., a function or obect with a __call__ method)
            func = func_type()
        else:
            raise TypeError("The function at the provided index is neither a callable nor a class")

        return func, func.label, func.dim
