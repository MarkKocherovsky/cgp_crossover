def sphere(coord):
	res = 0
	return sum([res+(x**2) for x in coord])
def sine(coord):
	res = 0
	return sum([res+(sin(x)) for x in coord])
def sqare_root(coord):
	return sqrt(coord)
def line(coord):
	res = 0
	return sum([res+(x*2) for x in coord])
def ackley(coord):
	res = 0
	for x in coord:
		res += -20*exp(-0.2*sqrt(1*x**2))-exp(1*cos(2*pi*x))+exp(1)+20
	return res
def gramacy_lee(x):
	x = x[0]
	return sin(10*pi*x)/(2*x)+(x-1)**4
def rastrigin(coord):
	res = 0
	return sum([res+(10+x**2-10*cos(2*pi*x)) for x in coord])
def dixon_price(coord):
	res = 0
	return sum([res+(x-1)**2+(2*x**2-x)**2 for x in coord])
def michalewicz(coord):
	res = 0
	return sum([res+(-1*sin(x)*sin(x**2/pi)**2*10) for x in coord])
