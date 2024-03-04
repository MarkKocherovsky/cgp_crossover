import random
import numpy as np

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
    testing_order = list(range(len(testcase_scores)))
    random.shuffle(testing_order)

    survivor_ids = []
    survivors = list(population)
    survivor_scores = np.array(testcase_scores)
    for i in testing_order:
        survivor_ids = list(np.flatnonzero(survivor_scores[i] == np.max(survivor_scores[i])))
        loser_ids = list(np.flatnonzero(survivor_scores[i] != np.max(survivor_scores[i])))
        survivors = [survivors[j] for j in survivor_ids]
        survivor_scores = np.delete(survivor_scores, loser_ids, 1)
        if (len(survivors) == 1):
            return survivors[0]        
        
    random.shuffle(survivors)
    return survivors[0]


def double_tournament_selection(population, fitness_scores, n1, n2):
    pass