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
		x_rng = [-5.0, 5.0]
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
		self.x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		self.label = "Koza 1"
		self.x_rng = [-1, 1]
		def func(x):
			return x**4+x**3+x**2+x
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)
		
class koza2(Function):
	def __init__(self):
		self.x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		self.x_rng = [-1, 1]
		self.label = "Koza 2"
		def func(x):
			return x**5-2*x**3+x
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)

class koza3(Function):
	def __init__(self):
		self.x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		self.x_rng = [-1, 1]
		self.label = "Koza 3"
		def func(x):
			return x**6-2*x**4+x**2
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)
		
class nguyen4(Function):
	def __init__(self):
		self.x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		self.x_rng = [-1, 1]
		self.label = "Nguyen 4"
		def func(x):
			return x**6+x**5+x**4+x**3+x**2+x
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)
		
class nguyen5(Function):
	def __init__(self):
		self.x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		self.label = "Nguyen 5"
		self.x_rng = [-1, 1]
		def func(x):
			return sin(x**2)*cos(x)-1
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)
		
class nguyen6(Function):
	def __init__(self):
		self.x_dom = (1.001-(-1))*random_sample((20,))+(-1)
		self.label = "Nguyen 6"
		self.x_rng = [-1, 1]
		def func(x):
			return sin(x)+sin(x+x**2)
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)
		
class nguyen7(Function):
	def __init__(self):
		self.x_dom = (2.001-(0))*random_sample((20,))+(0)
		self.label = "Nguyen 7"
		self.x_rng = [0, 2]
		def func(x):
			return log(x+1)+log(x**2+1)
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)
class ackley(Function):
	def __init__(self):
		self.x_dom = (32.7681-(-32.768))*random_sample((40,))+(-32.768)
		self.label = "Ackley"
		self.x_rng = [-32.768, 32.768]
		def func(x):
			a = 20
			b = 0.2
			c = 2*pi
			return -1*a*exp(-1*b*sqrt(x**2))-exp(cos(c*x))+a+exp(1)
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)

class rastrigin(Function):
	def __init__(self):
		self.x_dom = (5.12001-(-5.12))*random.sample((40,))+(-5.12)
		self.label = "Rastrigin"
		self.x_rng = [-5.12, 5.12]
		def func(x):
			return 10+(x**2-10*cos(2*pi*x)) 
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)

class griewank(Function):
	def __init__(self):
		self.x_dom = (600.0001-(-600))*random.sample((40,))+(-600)
		self.label = "Griewank"
		self.x_rng = [-60, 60]
		def func(x):
			return x**2/4000-cos(x)+1
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)

class levy(Function):
	def __init__(self):
		self.x_dom = (10.0001-(-10))*random.sample((40,))+(-10)
		self.label = 'Levy'
		self.x_rng = [-10, 10]
		def func(x):
			def w(x):
				return 1+(x-1)/4
			return sin(pi*w(x))**2+(w(x)-1)**2*(1+10*sin(pi*w(x)+1)**2)+(w(x)-1)**2*(1+sin(2*pi*w(x))**2)
		self.y_test = np.fromiter(map(func, list(self.x_dom)), dtype=np.float32)
		super().__init__(func, self.x_dom, self.y_test, self.label)


class Collection():
	def __init__(self):
		self.func_list = [koza1(), koza2(), koza3(), nguyen4(), nguyen5(), nguyen6(), nguyen7(), ackley(), rastrigin(), griewank(), levy()]
		self.name_list = [f.label for f in self.func_list]

c = Collection()
f = c.func_list
[x for x in f]
