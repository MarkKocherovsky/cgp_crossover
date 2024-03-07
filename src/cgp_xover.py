import numpy as np
from numpy import random
from subgraph import *

def xover_1x(p1, p2): #one point crossover
	ind1 = p1[0]
	ind2 = p2[0]
	out1 = p1[1]
	out2 = p2[1]
	s = ind1.shape #preserve real shape
	ind1 = ind1.flatten()
	ind2 = ind2.flatten()
	i = random.randint(1, ind1.shape[0]-1)
	front_1 = ind1[:i].copy()
	back_1 = ind1[i:].copy()
	front_2 = ind2[:i].copy()
	back_2 = ind2[i:].copy()
	ind1 = np.concatenate((front_1, back_2))
	ind2 = np.concatenate((front_2, back_1))
	
	if len(out1.shape) <=1 and out1.shape[0] == 1:
		t = out1.copy()
		out1 = out2.copy()
		out2 = t
	else:
		i = random.randint(0, out2.shape[0])
		front_1 = out1[:i].copy()
		back_1 = out1[i:].copy()
		front_2 = out2[:i].copy()
		back_2 = out2[i:].copy()
		out1 = np.concatenate((front_1, back_2))
		out2 = np.concatenate((front_2, back_1))
	return [(ind1.reshape(s), out1), (ind2.reshape(s), out2)]
def xover_nodex(p1, p2): #one point crossover
	ind1 = p1[0]
	ind2 = p2[0]
	out1 = p1[1]
	out2 = p2[1]
	s = ind1.shape #preserve real shape
	i = random.randint(1, ind1.shape[0]-1)
	front_1 = ind1[:i, :].copy()
	back_1 = ind1[i:, :].copy()
	front_2 = ind2[:i, :].copy()
	back_2 = ind2[i:, :].copy()
	ind1 = np.concatenate((front_1, back_2), axis = 0)
	ind2 = np.concatenate((front_2, back_1), axis = 0)
	
	if len(out1.shape) <=1 and out1.shape[0] == 1:
		t = out1.copy()
		out1 = out2.copy()
		out2 = t
	else:
		i = random.randint(0, out2.shape[0])
		front_1 = out1[:i].copy()
		back_1 = out1[i:].copy()
		front_2 = out2[:i].copy()
		back_2 = out2[i:].copy()
		out1 = np.concatenate((front_1, back_2))
		out2 = np.concatenate((front_2, back_1))
	return [(ind1.reshape(s), out1), (ind2.reshape(s), out2)]

def xover_2x(p1, p2): #two point crossover
	ind1 = p1[0]
	ind2 = p2[0]
	out1 = p1[1]
	out2 = p2[1]

	s = ind1.shape #preserve real shape
	ind1 = ind1.flatten()
	ind2 = ind2.flatten()
	i = random.randint(1, ind1.shape[0]-1)
	j = random.randint(i, ind1.shape[0]-1)
	front_1 = ind1[:i].copy()
	mid_1 = ind1[i:j].copy()
	back_1 = ind1[j:].copy()

	front_2 = ind2[:i].copy()
	mid_2 = ind2[i:j].copy()
	back_2 = ind2[j:].copy()

	ind1 = np.concatenate((front_1, mid_2, back_1))
	ind2 = np.concatenate((front_2, mid_1, back_2))
	
	if len(out1.shape) <=1 and out1.shape[0] == 1: #same as one point bc so far we have one output node anyways
		t = out1.copy()
		out1 = out2.copy()
		out2 = t
	else:
		i = random.randint(0, out2.shape[0])
		front_1 = out1[:i].copy()
		back_1 = out1[i:].copy()
		front_2 = out2[:i].copy()
		back_2 = out2[i:].copy()
		out1 = np.concatenate((front_1, back_2))
		out2 = np.concatenate((front_2, back_1))
	return [(ind1.reshape(s), out1), (ind2.reshape(s), out2)]

def xover_sgx(P1, P2, inputs = 1):
	children = []
	children.append(SubgraphCrossover(P1, P2, inputs))
	children.append(SubgraphCrossover(P2, P1, inputs))
	return children

def xover(parents, method = 'None', p_xov = 0.5):
	children = []
	methods = {'None': 'None', 'OnePoint': xover_1x, 'TwoPoint': xover_2x, 'Subgraph': xover_sgx, 'Node': xover_nodex}
	try:
		xover_method = methods[method]
	except:
		xover_method = 'None'
	retention = []
	for i in range(0, len(parents), 2):
		if xover_method == 'None' or random.random() < p_xov:
			children.append(parents[i])
			children.append(parents[i+1])
		else:
			c1, c2 = xover_method(parents[i], parents[i+1])
			children.append(c1)
			children.append(c2)
			retention.append(i)
	return children, np.array(retention).astype(np.int32)
