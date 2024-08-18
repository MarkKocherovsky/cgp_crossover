from math import log, pi

import numpy as np
from numpy import sin, cos, sqrt, exp, log
from numpy.random import random_sample

points = 20


def distance(x1, x2):
    return sqrt(x1 ** 2 + x2 ** 2)


class Function:
    def __init__(self, func, x_dom, y_test, label, dimensions):
        self.func = func
        self.x_dom = x_dom
        self.y_test = y_test
        self.label = label
        assert isinstance(dimensions, int)
        assert dimensions > 0
        self.dim = dimensions

    def __call__(self, x):
        return self.func(x)


"""
Multivariate Functions (0 < dimensions < n)
"""


class sphere(Function):
    def __init__(self, dimensions: int = 1):
        start = -5
        end = 5.01
        n_points = 20j
        x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        label = f"Sphere_{dimensions}D"

        """
        xs: list of x_values (x1, x2, x3,...,xn)
        """

        def func(xs: np.ndarray):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim
            return np.sum(xs ** 2)

        y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
        super().__init__(func, x_dom, y_test, label, dimensions)


class ackley(Function):
    def __init__(self, dimensions: int = 1):
        start = -32.768
        end = 32.769
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Ackley_{dimensions}D"
        self.x_rng = [-32.768, 32.768]

        # refactored by chatgpt
        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim

            a, b, c = 20, 0.2, 2 * pi

            def compute_component(inputs, sub_func):
                return np.sum(sub_func(inputs))

            first_component = lambda inputs: inputs ** 2
            second_component = lambda inputs: cos(c * inputs)

            first_sum = compute_component(xs, first_component)
            second_sum = compute_component(xs, second_component)

            return -a * exp(-b * first_sum) - exp(1 / self.dim * second_sum) + a + exp(1)

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


class rastrigin(Function):
    def __init__(self, dimensions: int = 1):
        start = -5.12
        end = 5.12
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Rastrigin_{dimensions}D"
        self.x_rng = [-5.12, 5.12]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim

            def compute_component(inputs, sub_func):
                return np.sum(sub_func(inputs))

            first_component = lambda inputs: inputs ** 2 - 10 * cos(2 * pi * inputs)

            first_sum = compute_component(xs, first_component)

            return 10 * dimensions + first_sum

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


class griewank(Function):
    def __init__(self, dimensions: int = 1):
        start = -600
        end = 600.01
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Griewank_{dimensions}D"
        self.x_rng = [-600, 600]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == self.dim

            def compute_sum(inputs, sub_func):
                return np.sum(sub_func(inputs))

            def compute_prod(inputs, sub_func):
                return np.prod(sub_func(inputs))

            first_component = lambda inputs: inputs ** 2 / 4000
            second_component = lambda inputs: list(map(lambda i, x: cos(x / sqrt(i + 1)), range(dimensions), inputs))

            first_sum = compute_sum(xs, first_component)
            second_sum = compute_prod(xs, second_component)

            return 10 * first_sum - second_sum + 1

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


# refactored from https://www.sfu.ca/~ssurjano/levy.html using chatgpt
class levy(Function):
    def __init__(self, dimensions: int = 1):
        start = -10
        end = 10.01
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Levy_{dimensions}D"
        self.x_rng = [-10, 10]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == dimensions

            # Compute w
            w = 1 + (np.array(xs) - 1) / 4

            # Compute components
            term1 = (sin(pi * w[0])) ** 2
            term3 = (w[-1] - 1) ** 2 * (1 + (sin(2 * pi * w[-1])) ** 2)
            sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * (sin(pi * w[:-1] + 1)) ** 2))

            return term1 + sum_term + term3

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


class schwefel(Function):
    def __init__(self, dimensions: int = 1):
        start = -500
        end = 500
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Schwefel_{dimensions}D"
        self.x_rng = [-500, 500]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == dimensions

            def compute_term(inputs):
                return np.sum([x * sin(sqrt(abs(x))) for x in inputs])

            return 418.9829 * dimensions - compute_term(xs)

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


class perm_0db(Function):
    def __init__(self, dimensions: int = 1):
        start = -dimensions
        end = dimensions
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Perm_Beta_{dimensions}D"
        self.x_rng = [-dimensions, dimensions]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == dimensions

            beta = 1

            def inner_sum(inputs, i):
                return np.sum([(j + beta) * (inputs[j] ** i - 1 / j ** i) for j in range(dimensions)])

            def outer_sum(inputs):
                return np.sum([inner_sum(inputs, i) for i in range(dimensions)])

            return outer_sum(xs)

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


class hyper_ellipsoid(Function):
    def __init__(self, dimensions: int = 1):
        start = -65.536
        end = 65.536
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Rotated_Hyper_Ellipsoid{dimensions}D"
        self.x_rng = [-65.536, 65.536]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == dimensions

            def inner_sum(inputs, i):
                return np.sum([inputs[j] ** 2 for j in range(i)])

            def outer_sum(inputs):
                return np.sum([inner_sum(inputs, i) for i in range(dimensions)])

            return outer_sum(xs)

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


class different_powers(Function):
    def __init__(self, dimensions: int = 1):
        start = -1
        end = 1
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Different_Powers_{dimensions}D"
        self.x_rng = [-1, 1]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == dimensions

            return np.sum([abs(xs[i]) ** (i + 2) for i in range(dimensions)])

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


class trid(Function):
    def __init__(self, dimensions: int = 1):
        start = -1 * int(dimensions ** 2)
        end = int(dimensions ** 2)
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(dimensions)]]
        self.label = f"Trid_{dimensions}D"
        self.x_rng = [-int(dimensions ** 2), int(dimensions ** 2)]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == dimensions

            def first_sum(inputs):
                return np.sum([(x - 1) ** 2 for x in inputs])

            def second_sum(inputs):
                return np.sum([inputs[i] * inputs[i - 1] for i in range(1, dimensions)])

            return first_sum(xs) - second_sum(xs)

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, dimensions)


"""
Univariate Functions (dimensions = 1)
"""


class Sine(Function):
    def __init__(self):
        x_dom = np.arange(-2 * pi, 2 * pi, 4 * pi / points)
        label = "Sine"

        def func(x):
            return np.sin(x)

        y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
        super().__init__(func, x_dom, y_test, label, int(1))


class SquareRoot(Function):
    def __init__(self, dimensions):
        x_dom = np.arange(0, 10.1, 10 / points)
        label = f"SquareRoot"

        def func(x):
            return np.sqrt(x)

        y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
        super().__init__(func, x_dom, y_test, label, int(1))


class koza1(Function):
    def __init__(self):
        self.x_dom = (1.001 - (-1)) * random_sample((20,)) + (-1)
        self.label = "Koza 1"
        self.x_rng = [-1, 1]

        def func(x):
            return x ** 4 + x ** 3 + x ** 2 + x

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, int(1))


class koza2(Function):
    def __init__(self):
        self.x_dom = (1.001 - (-1)) * random_sample((20,)) + (-1)
        self.x_rng = [-1, 1]
        self.label = "Koza 2"

        def func(x):
            return x ** 5 - 2 * x ** 3 + x

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, int(1))


class koza3(Function):
    def __init__(self):
        self.x_dom = (1.001 - (-1)) * random_sample((20,)) + (-1)
        self.x_rng = [-1, 1]
        self.label = "Koza 3"

        def func(x):
            return x ** 6 - 2 * x ** 4 + x ** 2

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, int(1))


class nguyen4(Function):
    def __init__(self):
        self.x_dom = (1.001 - (-1)) * random_sample((20,)) + (-1)
        self.x_rng = [-1, 1]
        self.label = "Nguyen 4"

        def func(x):
            return x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, int(1))


class nguyen5(Function):
    def __init__(self):
        self.x_dom = (1.001 - (-1)) * random_sample((20,)) + (-1)
        self.label = "Nguyen 5"
        self.x_rng = [-1, 1]

        def func(x):
            return sin(x ** 2) * cos(x) - 1

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, int(1))


class nguyen6(Function):
    def __init__(self):
        self.x_dom = (1.001 - (-1)) * random_sample((20,)) + (-1)
        self.label = "Nguyen 6"
        self.x_rng = [-1, 1]

        def func(x):
            return sin(x) + sin(x + x ** 2)

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, int(1))


class nguyen7(Function):
    def __init__(self):
        self.x_dom = (2.001 - 0) * random_sample((20,)) + 0
        self.label = "Nguyen 7"
        self.x_rng = [0, 2]

        def func(x):
            return log(x + 1) + log(x ** 2 + 1)

        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
        super().__init__(func, self.x_dom, self.y_test, self.label, int(1))


"""
Bivariate Functions (dimensions == 2)
"""


class bukin(Function):
    def __init__(self):
        self.x_dom = np.mgrid[slice(-15, 5.01, 40j), slice(-3, 3.01, 20j)]
        self.label = "Bukin"
        self.x_rng = [(-15, -3), (5, 3)]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return 100 * sqrt(abs(x2 - 0.01 * x1 ** 2)) + 0.01 * abs(x1 + 10)

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class cross_in_tray(Function):
    def __init__(self):
        start = -10
        end = 10.01
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Cross_in_Tray"
        self.x_rng = [-10, 10]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return -0.0001 * (abs(sin(x1) * sin(x2) * exp(abs(100 - sqrt(x1 ** 2 + x2 ** 2) / pi))) + 1) ** 0.1

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class drop_wave(Function):
    def __init__(self):
        start = -5.12
        end = 5.13
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Drop_Wave"
        self.x_rng = [-5.12, 5.12]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return -(1 + cos(12 * distance(x1, x2))) / (0.5 * (x1 ** 2 + x2 ** 2) + 2)

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class holder_table(Function):
    def __init__(self):
        start = -10
        end = 10.01
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Holder_Table"
        self.x_rng = [-10, 10]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return -(sin(x1) * cos(x2) * exp(abs(1 - distance(x1, x2) / pi)))

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class eggholder(Function):
    def __init__(self):
        start = -512
        end = 512.01
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Eggholder"
        self.x_rng = [-512, 512]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs
            return -(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47))) - x1 * sin(sqrt(abs(x1 - (x2 + 47))))

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


# Langermann is actually multivariate but because of the A matrix term I've locked it to 2 for now
class langermann(Function):
    def __init__(self):
        start = 0
        end = 10.01
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Langermann"
        self.x_rng = [0, 10]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            c = np.array([1, 2, 5, 2, 3])
            m = 5
            A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])

            outer_sum = 0
            for i in range(m):
                inner_sum = np.sum((xs - A[i, :]) ** 2)
                outer_sum += c[i] * np.exp(-inner_sum / np.pi) * np.cos(np.pi * inner_sum)

            return outer_sum

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class schaffer2(Function):
    def __init__(self):
        start = -100
        end = 100
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Schaffer_N2"
        self.x_rng = [-100, 100]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return 0.5 + (sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class schaffer4(Function):
    def __init__(self):
        start = -100
        end = 100
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Schaffer_N4"
        self.x_rng = [-100, 100]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return 0.5 + (cos(sin(abs(x1 ** 2 - x2 ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class shubert(Function):
    def __init__(self):
        start = -10
        end = 10
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Schubert"
        self.x_rng = [-10, 10]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            def compute_sum(x):
                return np.sum([i * cos(i + 1) * x + i for i in range(5)])

            return compute_sum(x1) * compute_sum(x2)

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class bohachevsky1(Function):
    def __init__(self):
        start = -100
        end = 100
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Bohachevsky_N1"
        self.x_rng = [-100, 100]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1) - 0.4 * cos(4 * pi * x2) + 0.7

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class bohachevsky2(Function):
    def __init__(self):
        start = -100
        end = 100
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Bohachevsky_N2"
        self.x_rng = [-100, 100]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1) * 0.4 * cos(4 * pi * x2) + 0.3

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class bohachevsky3(Function):
    def __init__(self):
        start = -100
        end = 100
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Bohachevsky_N3"
        self.x_rng = [-100, 100]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1 + 4 * pi * x2) + 0.3

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class booth(Function):
    def __init__(self):
        start = -10
        end = 10
        n_points = 40j
        self.x_dom = np.mgrid[[slice(start, end, n_points) for _ in range(2)]]
        self.label = "Booth"
        self.x_rng = [-10, 10]

        def func(xs):
            xs = np.atleast_1d(xs)
            assert len(xs) == 2
            x1, x2 = xs

            return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2

        # Evaluate function over domain
        self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)

        # Call the parent class constructor
        super().__init__(func, self.x_dom, self.y_test, self.label, int(2))


class Collection:
    def __init__(self):
        self.func_list = [koza1(), koza2(), koza3(), nguyen4(), nguyen5(), nguyen6(), nguyen7(), ackley(), rastrigin(),
                          griewank(), levy()]
        self.name_list = [f.label for f in self.func_list]
