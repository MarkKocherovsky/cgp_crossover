import numpy as np
from numpy import random

def fight(contestants, c_fitnesses):
	winner = np.argmin(c_fitnesses)
	return contestants[winner], winner

def lgp_tournament_elitism_selection(pop, fitnesses, max_p, n_tour = 4):
	n_tour = int(len(pop)/10)
	if n_tour <=1:
		n_tour = 2
	new_parents = []
	idxs = []
	#new_fitnesses = []
	#print(fitnesses)
	#print(np.argmin(fitnesses)
	best_fit_id = np.argmin(fitnesses)
	best_fit = np.argmin(fitnesses)
	#print(pop[best_fit_id])
	new_parents.append(pop[best_fit_id])
	while len(new_parents) < max_p:
		contestant_indices = random.choice(range(len(pop)), n_tour, replace = False)
		#print(pop)
		contestants = []
		for i in contestant_indices:
			contestants.append(pop[i])
		c_fitnesses = fitnesses[contestant_indices]
		candidate, winner = fight(contestants, c_fitnesses)
		#if winner not in idxs:
		new_parents.append(candidate)	
	return new_parents
