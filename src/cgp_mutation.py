import numpy as np
from numpy import random
from copy import deepcopy

#Subjects: Individuals to go through the loop

def mutate_1_plus_4(individual, len_bank = 4, arity = 2, in_size = 11): #temporary to make cgp(1+4) work
	ind = individual[0]
	out = individual[1]
	i = int((ind.shape[0]+out.shape[0])*random.random_sample())
	if i >= ind.shape[0]: #output node
		i = i-ind.shape[0]
		out[i] = random.randint(0, ind.shape[0]+in_size)
	else: #body node
		j = int(ind.shape[1]*random.random_sample())
		#print(i,j)
		if j < arity:
			ind[i, j] = random.randint(0, i+in_size)
		else:
			ind[i,j] = random.randint(0, len_bank)
	return ind, out

def basic_mutation(subjects, arity = 2, in_size = 11, p_mut = 0.025, bank_len = 4):
	mut_id = np.random.randint(0, len(subjects), (1,))
	#mutants = parents[mut_id]
	for m in range(len(subjects)):
		if random.random() >= p_mut:
			continue
		mutant = deepcopy(subjects[m])
		ind = mutant[0].copy()
		out = mutant[1].copy()
		i = int((ind.shape[0]+out.shape[0])*random.random_sample())
		if i >= ind.shape[0]: #output node
			i = i-ind.shape[0]
			out[i] = random.randint(0, ind.shape[0]+in_size)
		else: #body node
			j = int(ind.shape[1]*random.random_sample())
			#print(i,j)
			if j < arity:
				ind[i, j] = random.randint(0, i+in_size)
			else:
				ind[i,j] = random.randint(0, bank_len)
		subjects[m] = (ind, out)
	return subjects
