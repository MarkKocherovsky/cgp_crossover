import numpy as np
from numpy import random

def generate_parents(max_p, max_n, bank, first_body_node = 11, outputs = 1, arity = 2):
	parents = []
	for p in range(0, max_p):
		ind_base = np.zeros((max_n, arity+1), np.int32)
		for i in range(0, max_n):
			#print(i < inputs+bias)
			for j in range(0, arity): #input
				if i < (first_body_node):
					ind_base[i,j] = random.randint(0, first_body_node)
				else:
					ind_base[i,j] = random.randint(0, i+first_body_node)
			ind_base[i, -1] = random.randint(0, len(bank))
			output_nodes = random.randint(0, max_n+(first_body_node), (outputs,), np.int32)
		if max_p ==1:
			return (ind_base.copy(), output_nodes.copy())
		else:
			parents.append((ind_base.copy(), output_nodes.copy()))
	return parents
