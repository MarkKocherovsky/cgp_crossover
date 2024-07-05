import numpy as np
from numpy import random
from subgraph import *

def xover_1x(p1, p2, max_n = 64, first_body_node = 11, fixed_length = True, bank_len = 4): #one point crossover
	ind1 = p1[0]
	ind2 = p2[0]
	out1 = p1[1]
	out2 = p2[1]
	s1 = ind1.shape[1] #preserve real shape
	if fixed_length:
		s2 = s1
		ind1 = ind1.flatten()
		ind2 = ind2.flatten()
	else:
		s2 = ind2.shape[1]
	try:
		i = random.randint(1, ind1.shape[0]-1)
	except ValueError:
		i = 1
	if fixed_length:
		j = i
	else:
		try:
			j = random.randint(1, ind2.shape[0]-1)
		except ValueError:
			j = 1
	front_1 = ind1[:i].copy()
	back_1 = ind1[i:].copy()
	front_2 = ind2[:j].copy()
	back_2 = ind2[j:].copy()
	if fixed_length:
		ind1 = np.concatenate((front_1, back_2))
		ind2 = np.concatenate((front_2, back_1))
	else:
		ind1 = np.concatenate((front_1, back_2), axis = 0)
		ind2 = np.concatenate((front_2, back_1), axis = 0)
	
	if len(out1.shape) <=1 and out1.shape[0] == 1:
		t = out1.copy()
		out1 = out2.copy()
		out2 = t
	else:
		i = random.randint(0, out1.shape[0])
		if fixed_length:
			j = i
		else:
			j = random.randint(0, out2.shape[0])
			print(i, j)
		front_1 = out1[:i].copy()
		back_1 = out1[i:].copy()
		front_2 = out2[:j].copy()
		back_2 = out2[j:].copy()
		out1 = np.concatenate((front_1, back_2))
		out2 = np.concatenate((front_2, back_1))
	whr_o1 = np.where(out1 >= ind1.shape[0])
	whr_o2 = np.where(out2 >= ind2.shape[0])
	for w in whr_o1:
		out1[w] = ind1.shape[0]+first_body_node-1
	for w in whr_o2:
		out2[w] = ind2.shape[0]+first_body_node-1
	
	ind1 = ind1.reshape(-1, s1)
	ind2 = ind2.reshape(-1, s2)
	if ind1.shape[0] > max_n: #keep to maximimum rule size!
		idxs = np.array(range(0, max_n))
		to_del = random.choice(idxs, ((ind1.shape[0]-max_n),), replace=False)
		ind1 = np.delete(ind1, to_del, axis = 0)
	if ind2.shape[0] > max_n:
		idxs = np.array(range(0, max_n))
		to_del = random.choice(idxs, ((ind2.shape[0]-max_n),), replace=False)
		ind2 = np.delete(ind2, to_del, axis = 0)
	whr1 = np.array(list(zip(*np.where(ind1[:, :-1] >= ind1.shape[0]))))
	whr2 = np.array(list(zip(*np.where(ind2[:, :-1] >= ind2.shape[0]))))
	for w in whr1:
		try:
			ind1[w, w.shape[0]] = random.randint(0, w[0], (w.shape[0],))
		except:
			ind1[w, w.shape[0]] = 0
	for w in whr2:
		try:
			ind2[w, w.shape[0]] = random.randint(0, w[0], (w.shape[0],))
		except:
			ind2[w, w.shape[0]] = 0
	ar1 = np.array(list(zip(*np.where(ind1[:, -1] >= bank_len))))
	ar2 = np.array(list(zip(*np.where(ind2[:, -1] >= bank_len))))
	for a in ar1:
		ind1[a[0], -1] = random.randint(0, bank_len)
	for a in ar2:
		ind2[a[0], -1] = random.randint(0, bank_len)
	return [(ind1, out1), (ind2, out2)]
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

def xover(parents, method = 'None', p_xov = 0.5, max_n = 64, first_body_node = 11, fixed_length = True, bank_len = 4):
	children = []
	methods = {'None': 'None', 'OnePoint': xover_1x, 'TwoPoint': xover_2x, 'Subgraph': xover_sgx, 'Node': xover_nodex}
	try:
		xover_method = methods[method]
	except:
		xover_method = 'None'
	retention = []
	parent_distro = np.zeros(len(parents))
	for i in range(0, len(parents), 2):
		if method == 'None' or random.random() < p_xov:
			children.append(parents[i])
			children.append(parents[i+1])
		elif method == 'OnePoint':
			c1, c2 = xover_method(parents[i], parents[i+1], max_n = max_n, fixed_length = fixed_length, first_body_node = first_body_node, bank_len = bank_len)
			children.append(c1)
			children.append(c2)
			retention.append(i)
		else:
			c1, c2 = xover_method(parents[i], parents[i+1])
			children.append(c1)
			children.append(c2)
			retention.append(i)
	return children, np.array(retention).astype(np.int32)
