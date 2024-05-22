import numpy as np
from numpy import random
from copy import deepcopy
from cgp_parents import generate_single_instruction
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

def macromicro_mutation(subjects, arity = 2, in_size = 11, p_mut = 0.025, bank_len = 4, n_max = 64):
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
			if len(ind) < 2: #too small
				mutation = random.randint(1, 3)
			elif len(ind) >= n_max: #too big
				mutation = random.randint(0, 2)
			else: #any mutation
				mutation = random.randint(0, 3)
			if mutation == 0: #remove instruction, TODO 7 May: deleting instructions without changing other instructions causes recursion error
				j = random.randint(0, len(ind))
				ind = np.delete(ind, j, axis = 0)
				#print(f'Instruction Deleted at {j}')
			elif mutation == 1: #point mutation
				j = int(ind.shape[1]*random.random_sample())
				#print(i,j)
				if j < arity:
					#print(f'{i} {j} {i+in_size}')
					ind[i, j] = random.randint(0, i+in_size)
				else:
					ind[i,j] = random.randint(0, bank_len)
			elif mutation == 2: #Add instruciton
				j = random.randint(0, len(ind))
				new_inst = generate_single_instruction(j)
				#print(f'New Instruction {new_inst} inserted at {j} (treated as {j+in_size}')
				np.insert(ind, j, new_inst, axis = 0)
			else:
				raise ValueError(f'src::cgp_mutation::macromicro_mutation: asked for mutation == {mutation}')
			for k in range(len(ind)): #this is kludgy but readable, prevent recursion
				for l in range(0, ind[k, :-1].shape[0]):
					if ind[k, l] == k+in_size:
						#print(f'Node {k}: {ind[k]} | position {l}')
						#print(f'replacing {ind[k,l]} in {k}, {l}')
						ind[k, l] = random.randint(0, k+in_size)
						#print(ind[k])
						#print("------")
		subjects[m] = (ind, out)
	return subjects 
