import numpy as np
from numpy import random, sin, cos, tan, sqrt, exp, log, abs, floor, ceil
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
		
class Collection():
	def __init__(self):
		self.func_list = [Sphere(), Sine(), SquareRoot()]
		self.name_list = [f.label for f in self.func_list]
