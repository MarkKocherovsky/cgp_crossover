import numpy as np
from numpy import random

# pop: population
# f_list: fitnesses
# n_con: number of contestants
def tournament_elitism(pop, f_list, max_p, n_con=4):
    new_p = []
    idx = np.array(range(0, len(pop)))
    # keep best ind
    best_f_i = np.argmin(f_list)
    new_p.append(pop[best_f_i])
    while len(new_p) < max_p:
        # get contestants id
        c_id = random.choice(idx, (n_con,), replace=False)
        f_c = f_list[c_id]
        winner = np.argmin(f_c)
        w_id = c_id[winner]
        new_p.append(pop[w_id])
    return new_p

# Population is a list of solutions.
# Fitness_scores is a list of scores where fitness[i] corresponds to pop[i].
def elitism_selection(population, fitness_scores, num_parents):
    combined = list(zip(fitness_scores, population))
    combined.sort(reverse=True)
    _, parents = list(map(list, zip(*combined)))
    return parents[:num_parents]


# Population is a list of solutions.
# Testcase_scores is a 2-layered list of scores where test[i][j] corresponds
# to the score of pop[j] on test[i].
def lexicase_selection(population, testcase_scores):
    # Assemble test cases in random order
    testing_order = list(range(len(testcase_scores)))
    random.shuffle(testing_order)

    # Selection
    survivors = list(population)
    survivor_scores = np.array(testcase_scores)
    for i in testing_order:
        # Find ids of best individuals for the test case
        winner_ids = list(np.flatnonzero(
            survivor_scores[i] == np.max(survivor_scores[i])))
        loser_ids = list(range(len(survivors))) - winner_ids

        # Assemble remaining individuals and scores based on winning ids
        survivors = [survivors[j] for j in winner_ids]
        survivor_scores = np.delete(survivor_scores, loser_ids, 1)
        if (len(survivors) == 1):
            return survivors[0]
    return random.choice(survivors)


def double_tournament_selection(population, scores1, scores2, n1, n2, num_parents):
    parents = []
    pop_ids = np.array(range(len(population)))
    while len(parents) < num_parents:
        # Layer 1
        finalist_ids = []
        for i in range(n2):
            contestant_ids = random.choice(pop_ids, (n1,), replace=False)
            temp = np.argmin(scores1[contestant_ids])
            winner_id = contestant_ids[temp]
            finalist_ids.append(winner_id)

        # Layer 2
        temp = np.argmin(scores2[finalist_ids])
        winner_id = finalist_ids[temp]
        parents.append(population[winner_id])
    return parents


def roulette_wheel_selection(population, fitness_scores):
    parents = []
    total_fitness = sum(fitness_scores)
    for i in range(len(population)):
        spin = random.random_sample() * total_fitness
        points = 0
        for j in range(len(population)):
            points += fitness_scores[j]
            if points >= spin:
                parents.append(population[j])
                break
