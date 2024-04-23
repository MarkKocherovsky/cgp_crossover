import numpy as np
from numpy import random
import math
from effProg import cgp_active_nodes
from effProg import effProg


# Fitness_scores is a list of scores where fitness[i] corresponds to pop[i].
# Takes the {proportion} of the population that is most fit and duplicates it to reach {max_parents}
def truncation_elitism(population, fitness_scores, max_parents, proportion=0.4, minimize=True):
    if (minimize):
        fitness_scores = np.array([i * -1 for i in fitness_scores])

    combined = list(zip(fitness_scores, population))
    combined.sort(key=lambda p: p[0], reverse=True)
    _, parents = list(map(list, zip(*combined)))

    truncated = parents[:math.ceil(len(population) * proportion)]
    parents = []
    for i in range(max_parents):
        parents.append(truncated[i % len(truncated)])
    return parents


# # The best two out of the three contestants are chosen in layer two
# def cgp_double_tournament(population, fitness_scores, max_parents, t1_size=4, t2_size=3, minimize=True):
#     size_scores = np.zeros(len(population))
#     for i in range(len(population)):
#         size_scores[i] = cgp_active_nodes(
#             population[i][0], population[i][1], opt=0)

#     if (minimize):
#         fitness_scores = np.array([i * -1 for i in fitness_scores])
#         size_scores = np.array([i * -1 for i in size_scores])

#     parents = []
#     pop_ids = np.array(range(len(population)))
#     while len(parents) < max_parents:
#         # Layer 1
#         finalist_ids = []
#         for i in range(t2_size):
#             contestant_ids = random.choice(pop_ids, (t1_size,), replace=False)
#             temp = np.argmin(fitness_scores[contestant_ids])
#             winner_id = contestant_ids[temp]
#             finalist_ids.append(winner_id)

#         # Layer 2
#         firstplace = np.argmin(size_scores[finalist_ids])
#         del finalist_ids[firstplace]
#         secondplace = np.argmin(size_scores[finalist_ids])
#         parents.append(population[firstplace])
#         if len(parents) < max_parents:
#             parents.append(population[secondplace])
#     return parents


# def lgp_double_tournament(population, fitness_scores, max_parents, t1_size=4, t2_size=3, minimize=True):
#     size_scores = np.zeros(len(population))
#     for i in range(len(population)):
#         size_scores[i] = len(effProg(4, population[i]))

#     if (minimize):
#         fitness_scores = np.array([i * -1 for i in fitness_scores])
#         size_scores = np.array([i * -1 for i in size_scores])

#     parents = []
#     pop_ids = np.array(range(len(population)))
#     while len(parents) < max_parents:
#         # Layer 1
#         finalist_ids = []
#         for i in range(t2_size):
#             contestant_ids = random.choice(pop_ids, (t1_size,), replace=False)
#             temp = np.argmin(fitness_scores[contestant_ids])
#             winner_id = contestant_ids[temp]
#             finalist_ids.append(winner_id)

#         # Layer 2
#         firstplace = np.argmin(size_scores[finalist_ids])
#         del finalist_ids[firstplace]
#         secondplace = np.argmin(size_scores[finalist_ids])
#         parents.append(population[firstplace])
#         if len(parents) < max_parents:
#             parents.append(population[secondplace])
#     return parents


def roulette_wheel(population, fitness_scores, max_parents, minimize=True):
    if (minimize):
        fitness_scores = np.array([1/(i+1) for i in fitness_scores])

    parents = []
    total_fitness = sum(fitness_scores)
    for n in range(max_parents):
        spin = random.random_sample() * total_fitness
        points = 0
        for i in range(len(population)):
            points += fitness_scores[i]
            if points >= spin:
                parents.append(population[i])
                break
    return parents


# 1 <= Pressure <= 2
def linear_ranked(population, fitness_scores, max_parents, pressure=1.5, minimize=True):
    if (minimize):
        fitness_scores = np.array([i * -1 for i in fitness_scores])

    # Sort by rank in ascending order
    ranked_pop = list(zip(population, fitness_scores))
    ranked_pop, _ = zip(*sorted(ranked_pop, key=lambda p: p[1]))

    # Generate selection probabilities
    ranking_scores = []
    N = len(population)
    for i in range(len(population)):
        prob = (1/N) * (pressure - (2*pressure - 2) * (i / (N-1)))
        ranking_scores.append(prob)

    # Selection
    parents = []
    for n in range(max_parents):
        spin = random.random_sample()  # Total probability is 1
        points = 0
        for i in range(len(population)):
            points += ranking_scores[i]
            if points >= spin:
                parents.append(ranked_pop[i])
                break
    return parents


# Testcase_scores is a 2-layered list of scores where test[i][j] corresponds
# to the score of pop[i] on test[j].
def lexicase(population, testcase_scores, max_parents, epsilon=0.1, minimize=True):
    if (minimize):
        temp = np.zeros((len(testcase_scores), len(testcase_scores[0])))
        for i in range(len(testcase_scores)):
            for j in range(len(testcase_scores[0])):
                temp[i][j] = testcase_scores[i][j] * -1
        testcase_scores = temp

    parents = []
    testcase_scores = testcase_scores.T  # Convert to pop[j] on test[i]
    print(testcase_scores)
    while (len(parents) < max_parents):
        # Assemble test cases in random order
        testing_order = list(range(len(testcase_scores)))
        random.shuffle(testing_order)

        # Selection
        survivors = list(population)
        survivor_scores = np.array(testcase_scores)
        pick_random = True
        for i in testing_order:
            # Find ids of best individuals for the test case
            winner_ids = list(np.flatnonzero(
                survivor_scores[i] >= np.max(survivor_scores[i]) - epsilon))
            loser_ids = list(np.flatnonzero(
                survivor_scores[i] < np.max(survivor_scores[i]) - epsilon))

            # Assemble remaining individuals and scores based on winning ids
            survivors = [survivors[j] for j in winner_ids]
            survivor_scores = np.delete(survivor_scores, loser_ids, 1)
            if (len(survivors) == 1):
                parents.append(survivors[0])
                pick_random = False
                break

        if (pick_random):
            r = random.randint(len(survivors))
            parents.append(survivors[r])
    return parents

#Selection
#pop: population
#f_list: fitnesses
#n_con: number of contestants
def cgp_tournament_elitism(pop, f_list, max_p, n_con = 4):
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

def fight(contestants, c_fitnesses):
    winner = np.argmin(c_fitnesses)
    return contestants[winner], winner

def lgp_tournament_elitism(pop, fitnesses, max_p, n_tour = 4):
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
