import numpy as np
from numpy import random, sin, cos, tan, sqrt, exp, log, abs, floor, ceil
from numpy.random import random_sample
from math import log, pi
points = 20

class Function:
	def __init__(self, func, x_dom, y_test, label):
		self.func = func
		self.x_dom = x_dom
		self.y_test = y_test
		self.label = label
	def __call__(self, x):
		return self.func(x)
		

class Sphere(Function):
	def __init__(self):
		x_dom = np.arange(-5.0, 5.01, 10.0/points)
		label = "Sphere"
		def func(x):
			return x**2
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)


class Sine(Function):
	def __init__(self):
		x_dom = np.arange(-2*pi, 2*pi, 4*pi/points)
		label = "Sine"
		def func(x):
			return np.sin(x)
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)

class SquareRoot(Function):
	def __init__(self):
		x_dom = np.arange(0, 10.1, 10/points)
		label = "SquareRoot"
		def func(x):
			return np.sqrt(x)
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)
		
class koza1(Function):
	def __init__(self):
		x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		label = "Koza 1"
		def func(x):
			return x**4+x**3+x**2+x
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)
		
class koza2(Function):
	def __init__(self):
		x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		label = "Koza 2"
		def func(x):
			return x**5-2*x**3+x
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)

class koza3(Function):
	def __init__(self):
		x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		label = "Koza 3"
		def func(x):
			return x**6-2*x**4+x**2
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)
		
class nguyen4(Function):
	def __init__(self):
		x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		label = "Nguyen 4"
		def func(x):
			return x**6+x**5+x**4+x**3+x**2+x
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)
		
class nguyen5(Function):
	def __init__(self):
		x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		label = "Nguyen 5"
		def func(x):
			return sin(x**2)*cos(x)-1
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)
		
class nguyen6(Function):
	def __init__(self):
		x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		label = "Nguyen 6"
		def func(x):
			return sin(x)+sin(x+x**2)
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)
		
class nguyen7(Function):
	def __init__(self):
		x_dom = (2.001-(0))*random_sample((20,))+(0)
		label = "Nguyen 7"
		def func(x):
			return log(x+1)+log(x**2+1)
		y_test = np.fromiter(map(func, list(x_dom)), dtype=np.float32)
		super().__init__(func, x_dom, y_test, label)

class Collection():
	def __init__(self):
		self.func_list = [koza1(), koza2(), koza3(), nguyen4(), nguyen5(), nguyen6(), nguyen7()]
		self.name_list = [f.label for f in self.func_list]

c = Collection()
f = c.func_list
[x for x in f]
