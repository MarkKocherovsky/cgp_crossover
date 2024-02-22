import numpy as np
from numpy import random
from lgp_parents import *
def macromicro_mutation(individuals, max_c, max_r, max_d, bank, inputs = 1, n_bias = 10, arity = 2, p_mut = 1/40): #either removes, changes, or adds an instruction
	children = []
	newInstructionGenerator = lgpParentGenerator(1, max_r, max_d, bank)
	while len(children) < max_c:
		for p in range(0, len(individuals)): #go through each parent
			parent = individuals[p].copy()
			if random.random() <= p_mut:
				if len(parent < 2):
					mutation = random.randint(1, 3)
				else:
					mutation = random.randint(0, 3)
				
				if mutation == 0: #remove instfuction
					inst = random.randint(0, len(parent))
					children.append(np.delete(parent, inst, axis = 0))
				elif mutation == 1: #change instruction
					try:
						inst = random.randint(0, len(parent))
						part = random.randint(0, len(parent[inst]))
					except:
						print(parent)
					possible_destinations, possible_sources = newInstructionGenerator.get_registers()
					if part == 0: #destination
						parent[inst, part] = random.choice(possible_destinations) #destination index
					elif part == 1: #operator
						parent[inst, part] = random.randint(0, len(bank))
					else: #source
						parent[inst, part] = random.choice(possible_sources)
					children.append(parent)
				elif mutation == 2:
					try:
						inst = random.randint(0, len(parent))
					except ValueError:
						inst = 0
					new_inst = newInstructionGenerator() #easiest just to generate a new individual with one instruction
					np.insert(parent, inst, new_inst, axis = 0) #parent.insert(inst, new_inst)
					children.append(parent)
			else:
				children.append(parent)
			if len(children) >= max_c:
				break
	return children
