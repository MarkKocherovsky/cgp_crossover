import numpy as np
from numpy import random
#Selection
#pop: population
#f_list: fitnesses
#n_con: number of contestants
def tournament_elitism(pop, f_list, max_p, n_con = 4):
	new_p = []
	idx = np.array(range(0, len(pop)))
	#keep best ind
	best_f_i = np.argmin(f_list)
	new_p.append(pop[best_f_i])
	while len(new_p) < max_p:
		c_id = random.choice(idx, (n_con,), replace = False) #get contestants id
		f_c = f_list[c_id]
		winner = np.argmin(f_c)
		w_id = c_id[winner]
		new_p.append(pop[w_id])
	return new_p