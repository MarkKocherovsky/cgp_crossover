import numpy as np
from numpy import sin, cos, log, sqrt, exp, pi
from sklearn.model_selection import train_test_split


def distance(xs):
    return np.sqrt(np.sum(np.array(xs) ** 2))


class Function:
    def __init__(self, name: str, dimensions: int, x_domain: list | np.ndarray, function, points: int,
                 test_train_fraction: float = None):
        self.name = name
        self.dimensions = dimensions
        self.x_domain = np.array(x_domain)
        self.function = function
        self.train_x, self.test_x, self.train_y, self.test_y = self.get_points(points, test_train_fraction)

    def get_points(self, points, test_train_fraction):
        if self.x_domain.shape == (2,):
            # Single range applied across all dimensions
            low, high = self.x_domain
            x_points = np.random.uniform(low, high, size=(points, self.dimensions))
        elif self.x_domain.ndim == 1:
            # 1D domain: apply range uniformly across all dimensions
            x_points = np.random.uniform(
                self.x_domain[0], self.x_domain[1], size=(points, self.dimensions)
            )
        else:
            # Multi-dimensional domain
            assert self.dimensions == self.x_domain.shape[0], (
                "If the x_domain is different across dimensions, "
                "exactly n dimensions must be specified."
            )
            x_points = np.array([
                [np.random.uniform(low, high) for low, high in self.x_domain]
                for _ in range(points)
            ])

        # Evaluate function for each point
        y_points = np.array([self.function(*x) for x in x_points])

        # Train/test split
        if test_train_fraction is not None:
            train_x, test_x, train_y, test_y = train_test_split(
                x_points, y_points, test_size=test_train_fraction
            )
            return train_x, test_x, train_y, test_y
        else:
            # Return all data as training data
            return x_points, None, y_points, None

    def return_points(self):
        return self.train_x, self.test_x, self.train_y, self.test_y

    def __call__(self, x_points):
        x_points = np.atleast_2d(x_points)  # Ensure 2D input for multiple points
        assert x_points.shape[1] == self.dimensions, (
            f"Input data should have {self.dimensions} dimensions, "
            f"but {x_points.shape[1]} dimensions were given."
        )
        return np.array([self.function(*x) for x in x_points])


class OneDimensionalFunction(Function):  # functions that will always be 1-dimensional
    def __init__(self, name: str, x_domain: list | np.ndarray, function, points: int,
                 test_train_fraction: float = None):
        super().__init__(name, 1, x_domain, function, points, test_train_fraction)


class TwoDimensionalFunction(Function):  # functions that will always be 2-dimensional
    def __init__(self, name: str, x_domain: list | np.ndarray, function, points: int,
                 test_train_fraction: float = None):
        super().__init__(name, 2, x_domain, function, points, test_train_fraction)


koza_1 = OneDimensionalFunction(
    name='Koza 1',
    x_domain=[-1, 1],
    function=lambda x: x ** 4 + x ** 3 + x ** 2 + x,
    points=20
)

koza_2 = OneDimensionalFunction(
    name='Koza 2',
    x_domain=[-1, 1],
    function=lambda x: x ** 5 - 2 * x ** 3 + x,
    points=20
)

koza_3 = OneDimensionalFunction(
    name='Koza 3',
    x_domain=[-1, 1],
    function=lambda x: x ** 6 - 2 * x ** 4 + x ** 2,
    points=20
)

nguyen_4 = OneDimensionalFunction(
    name='Nguyen 4',
    x_domain=[-1, 1],
    function=lambda x: x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x,
    points=20
)

nguyen_5 = OneDimensionalFunction(
    name='Nguyen 5',
    x_domain=[-1, 1],
    function=lambda x: sin(x ** 2) * cos(x) - 1,
    points=20
)

nguyen_6 = OneDimensionalFunction(
    name='Nguyen 6',
    x_domain=[-1, 1],
    function=lambda x: sin(x) + sin(x + x ** 2),
    points=20
)

nguyen_7 = OneDimensionalFunction(
    name='Nguyen 7',
    x_domain=[0, 2],
    function=lambda x: log(x + 1) + log(x ** 2 + 1),
    points=20
)

forrester = OneDimensionalFunction(
    name='Forrester',
    x_domain=[0, 2],
    function=lambda x: (6 * x - 2) ** 2 * sin(12 * x - 4),
    points=20
)

bukin = TwoDimensionalFunction(
    name='Bukin',
    x_domain=[0, 2],
    function=lambda x1, x2: 100 * sqrt(abs(x2 - 0 * x1 ** 2)) + 0 * abs(x1 + 10),
    points=40
)

cross_in_tray = TwoDimensionalFunction(
    name='Cross-in-Tray',
    x_domain=[-10, 10],
    function=lambda x1, x2: -0.0001 * (
            abs(sin(x1) * sin(x2) * exp(abs(100 - sqrt(x1 ** 2 + x2 ** 2) / pi))) + 1) ** 0.1,
    points=40
)

drop_wave = TwoDimensionalFunction(
    name='Drop Wave',
    x_domain=[-5.12, 5.12],
    function=lambda x1, x2: -(1 + cos(12 * distance((x1, x2)))) / (0.5 * (x1 ** 2 + x2 ** 2) + 2),
    points=40
)

holder_table = TwoDimensionalFunction(
    name='Holder Table',
    x_domain=[-10, 10],
    function=lambda x1, x2: -(sin(x1) * cos(x2) * exp(abs(1 - distance((x1, x2)) / pi))),
    points=40
)

eggholder = TwoDimensionalFunction(
    name='Eggholder',
    x_domain=[-512, 512],
    function=lambda x1, x2: -(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47))) - x1 * sin(sqrt(abs(x1 - (x2 + 47)))),
    points=40
)

schaffer2 = TwoDimensionalFunction(
    name='Schaffer 2',
    x_domain=[-100, 100],
    function=lambda x1, x2: 0.5 + (sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2,
    points=40
)

schaffer4 = TwoDimensionalFunction(
    name='Schaffer 4',
    x_domain=[-100, 100],
    function=lambda x1, x2: 0.5 + (cos(sin(abs(x1 ** 2 - x2 ** 2))) ** 2 - 0.5) / (
            1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2,
    points=40
)

schubert = TwoDimensionalFunction(
    name='Schubert',
    x_domain=[-10, 10],
    function=lambda x1, x2: (lambda compute_sum: compute_sum(x1) * compute_sum(x2))(
        lambda x: np.sum([i * np.cos(i + 1) * x + i for i in range(5)])
    ),
    points=40
)

bohachevsky1 = TwoDimensionalFunction(
    name='Bohachevsky 1',
    x_domain=[-100, 100],
    function=lambda x1, x2: x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1) - 0.4 * cos(4 * pi * x2) + 0.7,
    points=40
)

bohachevsky2 = TwoDimensionalFunction(
    name='Bohachevsky 2',
    x_domain=[-100, 100],
    function=lambda x1, x2: x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1) * 0.4 * cos(4 * pi * x2) + 0.3,
    points=40
)

bohachevsky3 = TwoDimensionalFunction(
    name='Bohachevsky 3',
    x_domain=[-100, 100],
    function=lambda x1, x2: x1 ** 2 + 2 * x2 ** 2 - 0.3 * cos(3 * pi * x1 + 4 * pi * x2) + 0.3,
    points=40
)

booth = TwoDimensionalFunction(
    name='Booth',
    x_domain=[-10, 10],
    function=lambda x1, x2: (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2,
    points=40
)

matyas = TwoDimensionalFunction(
    name='Matyas',
    x_domain=[-10, 10],
    function=lambda x1, x2: 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2,
    points=40
)

mccormick = TwoDimensionalFunction(
    name='McCormick',
    x_domain=[[-1.5, 4], [-3, 4]],
    function=lambda x1, x2: sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1,
    points=40
)

threehumpcamel = TwoDimensionalFunction(
    name='Three-Hump Camel',
    x_domain=[-5, 5],
    function=lambda x1, x2: 2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 4 + x1 * x2 + x2 ** 2,
    points=40
)

sixhumpcamel = TwoDimensionalFunction(
    name='Six-Hump Camel',
    x_domain=[[-3, 3], [-2, 2]],
    function=lambda x1, x2: (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) + x1 * x2 + (-4 + 4 * x2 ** 2) * x2,
    points=40
)

de_ong = TwoDimensionalFunction(
    name='De Ong',
    x_domain=[-65.536, 65.536],
    function=lambda x1, x2: (
        lambda A: 1 / (0.002 + np.sum(1 / (np.arange(1, 26) + (x1 - A[0, :]) ** 6 + (x2 - A[1, :]) ** 6))))(
        np.vstack([
            np.tile([-32, -16, 0, 16, 32], 5),
            np.repeat([-32, -16, 0, 16, 32], 5)
        ])
    ),
    points=40
)
beale = TwoDimensionalFunction(
    name='Beale',
    x_domain=[-4.5, 4.5],
    function=lambda x1, x2: (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2,
    points=40
)

branin = TwoDimensionalFunction(
    name='Branin',
    x_domain=[(-5, 10), (0, 15)],
    function=lambda x1, x2: (
        lambda a=1, b=5.1 / (4 * pi ** 2), c=5 / pi, r=6, s=10, t=1 / (8 * pi):
        a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * cos(x1) + s
    )(),
    points=40
)

easom = TwoDimensionalFunction(
    name='Easom',
    x_domain=[-10, 10],
    function=lambda x1, x2: 1 * cos(x1) * cos(x2) * exp(-(x1 - pi) ** 2 - (x2 - pi) ** 2),
    points=40
)

goldstein_price = TwoDimensionalFunction(
    name='Goldstein-Price',
    x_domain=[-2, 2],
    function=lambda x1, x2: (
            (1 + (x1 + x2 + 1) ** 2 *
             (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)) *
            (30 + (2 * x1 - 3 * x2) ** 2 *
             (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
    ),
    points=40
)

power_sum = Function(
    name='Power Sum',
    x_domain=[0, 4],
    dimensions=4,
    function=lambda x1, x2, x3, x4: np.sum([
        np.sum([x ** i for x in [x1, x2, x3, x4]]) - b ** 2
        for i, b in enumerate([8, 18, 44, 114])
    ]),
    points=40
)

coleville = Function(
    name='Coleville',
    x_domain=[-10, 10],
    dimensions=4,
    function=lambda x1, x2, x3, x4: 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2 + (x3 - 1) ** 2 + 90 * (
            x3 ** 2 - x4) ** 2 + 10.1 * (
                                            (x2 - 1) ** 2 + (x4 - 1) ** 2) + 19.8 * (x2 - 1) * (x4 - 1),
    points=40
)


def sphere(n_dim):
    sphere_function = Function(
        name='Sphere',
        x_domain=[-5.12, 5.12],
        dimensions=n_dim,
        function=lambda *xs: np.sum(np.atleast_1d(xs) ** 2),
        points=20
    )
    return sphere_function


def ackley(n_dim):
    ackley_function = Function(
        name='Ackley',
        x_domain=[-32.768, 32.768],
        dimensions=n_dim,
        function=lambda *xs: (
                -20 * np.exp(-0.2 * np.sqrt(np.sum(np.array(xs) ** 2) / n_dim))
                - np.exp(np.sum(np.cos(2 * np.pi * np.array(xs))) / n_dim)
                + 20 + np.exp(1)
        ),

        points=40
    )
    return ackley_function


def rastrigin(n_dim):
    rastrigin_function = Function(
        name='Rastrign',
        x_domain=[-5.12, 5.12],
        dimensions=n_dim,
        function=lambda *xs: 10 * n_dim + np.sum(np.array(xs) ** 2 - 10 * cos(2 * pi * np.array(xs))),
        points=40
    )
    return rastrigin_function


def griewank(n_dim):
    griewank_function = Function(
        name='Rastrign',
        x_domain=[-600, 600],
        dimensions=n_dim,
        function=lambda *xs: (
                np.sum(np.array(xs) ** 2) / 4000 - np.prod(
            [cos(np.atleast_1d(xs)[i] / sqrt(i + 1)) for i in range(n_dim)]) + 1
        ),
        points=40
    )
    return griewank_function


def levy(n_dim):
    levy_function = Function(
        name='Levy',
        x_domain=[-10, 10],
        dimensions=n_dim,
        function=lambda *xs: (
                (sin(pi * (1 + (np.atleast_1d(xs)[0] - 1) / 4))) ** 2 +
                np.sum(((1 + (np.atleast_1d(xs)[:-1] - 1) / 4) - 1) ** 2 *
                       (1 + 10 * (sin(pi * (1 + (np.atleast_1d(xs)[:-1] - 1) / 4) + 1)) ** 2)) +
                ((1 + (np.atleast_1d(xs)[-1] - 1) / 4) - 1) ** 2 *
                (1 + (sin(2 * pi * (1 + (np.atleast_1d(xs)[-1] - 1) / 4))) ** 2)
        ),
        points=40
    )
    return levy_function


def schwefel(n_dim):
    schwefel_function = Function(
        name='Levy',
        x_domain=[-500, 500],
        dimensions=n_dim,
        function=lambda *xs: 418.9829 * n_dim - np.sum([x * sin(sqrt(abs(x))) for x in np.atleast_1d(xs)]),
        points=40
    )
    return schwefel_function


def perm(n_dim):
    perm_function = Function(
        name='Perm',
        x_domain=[-n_dim, n_dim],
        dimensions=n_dim,
        function=lambda *xs: np.sum(
            [np.sum([(j + 0.5) * (np.atleast_1d(xs)[j] ** i - 1 / j ** i) for j in range(1, n_dim)]) for i in
             range(1, n_dim)]),
        points=40
    )
    return perm_function


def hyper_ellipsoid(n_dim):
    hyper_ellipsoid_function = Function(
        name='Hyper Ellipsoid',
        x_domain=[-65.536, 65.536],
        dimensions=n_dim,
        function=lambda *xs: np.sum([np.sum([np.atleast_1d(xs)[j] ** 2 for j in range(i)]) for i in range(n_dim)]),
        points=40
    )
    return hyper_ellipsoid_function


def different_powers(n_dim):
    different_powers_function = Function(
        name='Different Powers',
        x_domain=[-1, 1],
        dimensions=n_dim,
        function=lambda *xs: np.sum([abs(np.atleast_1d(xs)[i]) ** (i + 2) for i in range(n_dim)]),
        points=40
    )
    return different_powers_function


def trid(n_dim):
    trid_function = Function(
        name='Trid',
        x_domain=[-n_dim ** 2, n_dim ** 2],
        dimensions=n_dim,
        function=lambda *xs: np.sum([(x - 1) ** 2 for x in np.atleast_1d(xs)]) - np.sum(
            [np.atleast_1d(xs)[i] * np.atleast_1d(xs)[i - 1] for i in range(1, n_dim)]),
        points=40
    )
    return trid_function


def zakharov(n_dim):
    zakharov_function = Function(
        name='Zakharov',
        x_domain=[-5, 10],
        dimensions=n_dim,
        function=lambda *xs: (
                np.sum(np.array(xs) ** 2) +
                np.sum([0.5 * (i + 1) * np.atleast_1d(xs)[i] for i in range(n_dim)]) ** 2 +
                np.sum([0.5 * (i + 1) * np.atleast_1d(xs)[i] for i in range(n_dim)]) ** 4
        ),
        points=40
    )
    return zakharov_function


def dixon_price(n_dim):
    dixon_price_function = Function(
        name='Dixon-Price',
        x_domain=[-10, 10],
        dimensions=n_dim,
        function=lambda *xs: (np.atleast_1d(xs)[0] - 1) ** 2 + np.sum(
            [i * (2 * np.atleast_1d(xs)[i] ** 2 - np.atleast_1d(xs)[i - 1]) ** 2 for i in range(1, n_dim)]),
        points=40
    )
    return dixon_price_function


def rosenbrock(n_dim):
    assert n_dim > 1, 'The Rosenbrock Function requires at least two dimensions.'
    rosenbrock_function = Function(
        name='Rosenbrock',
        x_domain=[-5, 10],
        dimensions=n_dim,
        function=lambda *xs: np.sum(
            [100 * (np.atleast_1d(xs)[i + 1] - np.atleast_1d(xs)[i] ** 2) ** 2 + (np.atleast_1d(xs)[i] - 1) ** 2 for i
             in range(n_dim - 1)]),
        points=40
    )
    return rosenbrock_function


def michaelewiz(n_dim):
    michaelewiz_function = Function(
        name='Michaelewiz',
        x_domain=[0, pi],
        dimensions=n_dim,
        function=lambda *xs: -np.sum(
            [sin(np.atleast_1d(xs)[i]) * sin((i * np.atleast_1d(xs)[i] ** 2) / pi) ** (2 * 10) for i in range(n_dim)]),
        points=40
    )
    return michaelewiz_function


class Collection:
    def __init__(self):
        self.function_list = {
            'Ackley': ackley,
            'Beale': beale,
            'Bohachevsky 1': bohachevsky1,
            'Bohachevsky 2': bohachevsky2,
            'Bohachevsky 3': bohachevsky3,
            'Booth': booth,
            'Branin': branin,
            'Bukin': bukin,
            'Coleville': coleville,
            'Cross-in-Tray': cross_in_tray,
            'De Ong': de_ong,
            'Different Powers': different_powers,
            'Dixon-Price': dixon_price,
            'Drop Wave': drop_wave,
            'Easom': easom,
            'Eggholder': eggholder,
            'Forrester': forrester,
            'Goldstein-Price': goldstein_price,
            'Griewank': griewank,
            'Holder Table': holder_table,
            'Hyper Ellipsoid': hyper_ellipsoid,
            'Koza 1': koza_1,
            'Koza 2': koza_2,
            'Koza 3': koza_3,
            'Levy': levy,
            'Matyas': matyas,
            'McCormick': mccormick,
            'Michaelewiz': michaelewiz,
            'Nguyen 4': nguyen_4,
            'Nguyen 5': nguyen_5,
            'Nguyen 6': nguyen_6,
            'Nguyen 7': nguyen_7,
            'Perm': perm,
            'Power Sum': power_sum,
            'Rastrigin': rastrigin,
            'Rosenbrock': rosenbrock,
            'Schaffer 2': schaffer2,
            'Schaffer 4': schaffer4,
            'Schubert': schubert,
            'Schwefel': schwefel,
            'Six-Hump Camel': sixhumpcamel,
            'Sphere': sphere,
            'Three-Hump Camel': threehumpcamel,
            'Trid': trid,
            'Zakharov': zakharov
        }

    def __call__(self, function_name, n_dims=1):
        assert function_name in self.function_list, f'{function_name} not a valid function.'
        try:
            return self.function_list.get(function_name(n_dims))
        except TypeError:
            return self.function_list.get(function_name)
