import numpy
from effProg import *
from cgp_operators import *
from scipy.stats import pearsonr
from numpy import random

#fit_bank = [rmse, corr]
#fit_names = ["RMSE", "1-R^2"]
#f = int(argv[7])
#fit = fit_bank[f]
#fit_name  = fit_names[f]
#print(fit_name)

def rmse(preds, reals):
	return np.sqrt(np.mean((preds-reals)**2)) #copied from stack overflow

def corr(preds, reals):
	if any(np.isnan(preds)) or any(np.isinf(preds)):
		return np.PINF
	r = pearsonr(preds, reals)[0]
	if np.isnan(r):
		r = 0
	return (1-r**2)
def align(ind, out, preds, reals):
	if not all(np.isfinite(preds)):
		return 1.0, 0.0
	try:
		align = np.round(np.polyfit(preds, reals, 1, rcond=1e-16), decimals = 14)
	except:
		return 1.0, 0.0
	a = align[0]
	b = align[1]
	return (a,b)

def change(new, old):
	return (new-old)/old

class Fitness:
	def __init__(self):
		self.data = None
		self.target = None
		self.individual = None
		self.fit = None
		self.bank = None
		self.arity = None
	def __call__(self, data, target, individual, fit_function = corr, bank = (add, sub, mul, div), arity = 2, opt = 0):
		self.data = data
		self.target = target
		self.individual = individual
		self.fit = fit_function
		self.bank = bank
		self.arity = arity
		return self.fitness(data, target, opt)

	def run(self, cur_node, inp_nodes):
		try:
			inp_size = inp_nodes.shape[0]
			args = []
			ind = self.individual[0]
			for j in range(self.arity):
				if cur_node[j] < inp_size:
					#print(inp_nodes)
					args.append(inp_nodes[cur_node[j]])
				else:
					args.append(self.run(ind[cur_node[j]-inp_size], inp_nodes))
			function = self.bank[cur_node[-1]]
			return function(args[0], args[1]) # so far 2d only
		except RecursionError:
			print(self.individual)
			raise ValueError(f'Input Node = {cur_node}')
		except IndexError:
			print(cur_node)
			raise IndexError()
				
	def run_output(self, inp_nodes):
		out_nodes = self.individual[1]
		ind = self.individual[0]
		inp_nodes = np.array(inp_nodes)
		outs = np.zeros(out_nodes.shape,)
		for i in np.arange(0, outs.shape[0], 1, np.int32):
			if out_nodes[i] < (len(inp_nodes)):
				outs[i] = inp_nodes[out_nodes[i]]
			else:
				outs[i] = self.run(ind[out_nodes[i]-inp_nodes.shape[0]], inp_nodes)
		return outs

	def fitness(self, data, targ, opt = 0):
		ind_base = self.individual[0]
		output_nodes = self.individual[1]
		data = np.array(data)
		out_x = np.zeros(data.shape[0])
		for x in range(data.shape[0]):
			if len(data.shape) <= 1:
				in_val = [data[x]]
			else:
				in_val = data[x, :]
			with np.errstate(invalid='raise'):
				try:
					out_x[x] = self.run_output(in_val)
				except (OverflowError, FloatingPointError):
					out_x[x] = np.nan	
				#except ValueError:
				#	print('ValueError')
				#	print(self.run_output(in_val))
		with np.errstate(invalid='raise'):
			try:
				(a,b) = align(ind_base, output_nodes, out_x, targ)
			except (OverflowError, FloatingPointError):
				return np.nan, 1.0, 0.0
		new_x = out_x*a+b
		if opt == 1:
			return new_x, a, b
		return self.fit(new_x, self.target), a, b

class FitCollection():
	def __init__(self):
		self.fit_list = [rmse, corr]
		self.name_list = ['RMSE', '1-R^2']

class MutationImpact():
	def __init__(self, neutral_limit = 0.1):
		self.drift_list = []
		self.drift_cum = np.array([0, 0, 0]) #del, neutral, beneficial
	def __call__(self, fitnesses, max_p, option = 'TwoParent', children = 4):
		drift = np.array([0, 0, 0])
		if option == 'TwoParent': #two parents have two children
			for i in range(0, max_p, 2):
				p = min(fitnesses[i], fitnesses[i+1])
				c = min(fitnesses[i+max_p], fitnesses[i+1+max_p])
				if change(c, p) > 0.1:
					drift[0] +=1
				elif change(c,p) < -0.1:
					drift[2] += 1
				else:
					drift[1] += 1
		elif option == 'OneParent':
			for i in range(0, max_p):
				p = fitnesses[i]
				c = []
				for x in range(i*children, i*children+children):
					c.append(fitnesses[x])
				c = min(c)
				if change(c, p) > 0.1: #yes it's not clean shut up
					drift[0] +=1
				elif change(c,p) < -0.1:
					drift[2] += 1
				else:
					drift[1] += 1
		self.drift_cum += np.copy(drift)
		self.drift_list.append(np.copy(self.drift_cum))
	def return_lists(self, option = 0):
		if option == 0:
			return self.drift_cum, self.drift_list
		elif option == 1:
			return self.drift_cum/np.sum(self.drift_cum), np.divide(self.drift_list, np.sum(self.drift_list, axis = 1).flatten()[:, np.newaxis])

			
