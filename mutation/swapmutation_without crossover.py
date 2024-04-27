import numpy as np
from numpy import random
from scipy.stats import pearsonr
import warnings
import matplotlib.pyplot as plt
from copy import deepcopy
from sys import path
from pathlib import Path

warnings.filterwarnings('ignore')
from function import *
from sys import argv

t = int(argv[1])  # trial
# Set seed for NumPy
np.random.seed(t)
# Set seed for Python's random module
random.seed(t)
func_bank = Collection()
func = func_bank.func_list[int(argv[2])]
func_name = func_bank.name_list[int(argv[2])]
x = func.x_dom
y_real = func.y_test

pop_size = 80
generations = 10000
inputs = 2
instructions = 64
nodes = 11
outputs = 1
mutation_prob = 0.025
crossover_prob = 0.5
input_nodes = np.arange(0, 10, 1).astype(np.int32)


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def multiply(x, y):
    return x * y


def division(x, y):
    return x / np.sqrt(1 + y ** 2)


def average(list):
    return sum(list) / len(list)


def get_functions():
    return add, subtract, multiply, division


operators = get_functions()
operators_string = ("+", "-", "*", "/")


def generate_population():
    pop = []
    for p in range(0, pop_size):
        individual = np.zeros((instructions, inputs + 1), dtype=np.int32)
        for i in range(0, instructions):
            for j in range(0, 2):
                if i < nodes:
                    individual[i, j] = random.randint(0, nodes)
                else:
                    individual[i, j] = random.randint(0, i + nodes)
            individual[i, -1] = random.randint(0, len(operators_string))
        output_individual = random.randint(0, instructions + nodes, (outputs,), np.int32)
        pop.append((individual.copy(), output_individual.copy()))
    return pop


def tournament_elitism_selection(pop, fitnesses, tournament_size=4):
    fit_parents2 = []
    id = np.array(range(0, len(pop)))
    best_individual = np.argmin(fitnesses)
    parents1 = pop[best_individual]
    idx = np.delete(id, best_individual, axis=0)
    parents2_id = random.choice(idx, (tournament_size,), replace=False)
    for i in parents2_id:
        fit_parents2.append(fitnesses[i])
    winner = np.argmin(fit_parents2)
    winner_id = parents2_id[winner]
    parents2 = pop[winner_id]
    return parents1, parents2, best_individual, winner_id


def mutation(individual):
    mutant = deepcopy(individual)
    individual_input = mutant[0].copy()
    individual_output = mutant[1].copy()
    rand = int((len(individual_input) + len(individual_output)) * random.random_sample())
    if rand == len(individual_input):
        individual_output[0] = random.randint(0, (len(individual_input) + nodes), (1,), np.int32)
    else:
        rand1 = int(individual_input.shape[1] * random.random_sample())
        if rand1 < inputs:
            individual_input[rand, rand1] = random.randint(0, (rand + nodes), (1,), np.int32)
        else:
            individual_input[rand, rand1] = random.randint(0, len(operators))
    mutant = (individual_input, individual_output)
    return mutant
def one_point_xover(p1, p2):
    if random.random() > crossover_prob:
        return [p1, p2]
    else:
        ind1 = p1[0]
        ind2 = p2[0]
        out1 = p1[1]
        out2 = p2[1]
        s = ind1.shape  # preserve real shape
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        i = random.randint(1, ind1.shape[0] - 1)
        front_1 = ind1[:i].copy()
        back_1 = ind1[i:].copy()
        front_2 = ind2[:i].copy()
        back_2 = ind2[i:].copy()
        ind1 = np.concatenate((front_1, back_2))
        ind2 = np.concatenate((front_2, back_1))

        if len(out1.shape) <= 1 and out1.shape[0] == 1:
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
def random_numbers(individual1):
    indices = random.choice(range(len(individual1)), 2, replace=False)
    indices = np.sort(indices, axis=0)
    if indices[0].copy() < nodes:
        if indices[1].copy() >= nodes:
            if individual1[indices[1], 0] >= nodes or individual1[indices[1], 1] >= nodes:
                indices = random_numbers(individual1)
        else:
            indices = indices
    else:
        if individual1[indices[0], 0] >= (indices[1] + nodes - 1) or individual1[indices[0], 1] >= (
                indices[1] + nodes - 1):
            indices = random_numbers(individual1)
        else:
            if individual1[indices[1], 0] >= (indices[0] + nodes - 1) or individual1[indices[1], 1] >= (
                    indices[0] + nodes - 1):
                indices = random_numbers(individual1)
            else:
                indices = indices
    return indices


def swap_mutation(individual):
    individual1 = individual[0].copy()
    out = individual[1].copy()
    indices = random_numbers(individual1)
    inst1 = individual1[indices[0]]
    inst2 = individual1[indices[1]]
    individual2 = individual1.copy()
    individual2[indices[0]] = inst2.copy()
    individual2[indices[1]] = inst1.copy()
    individual = (individual2, out)
    return individual


def correlation(y_pred, y_real):
    if any(np.isnan(y_pred)) or any(np.isinf(y_pred)):
        return np.PINF
    r = pearsonr(y_pred, y_real)[0]
    if np.isnan(r):
        r = 0
    return 1 - (r ** 2)


def run_ind(individual, inst, inp):
    operands = []
    ind_input = individual[0].copy()
    for j in range(inputs):
        if (inst[j]) < nodes:
            operands.append(inp[inst[j]])
        else:
            operands.append(run_ind(individual, ind_input[inst[j] - nodes], inp))
    operator = operators[inst[-1]]
    return operator(operands[0], operands[1])


def run_output(individual, inp):
    ind_input = individual[0].copy()
    output = individual[1].copy()
    inp = np.array(inp)
    out = np.zeros(len(output), )
    for i in range(len(output)):
        if output[i] < nodes:
            out[i] = inp[output[i]]
        else:
            out[i] = run_ind(individual, ind_input[output[i] - nodes], inp)
    return out


def fitness(individual, y_real, x):
    preds = np.zeros(len(x), )
    input_data = np.zeros((len(x), nodes))
    input_data[:, 0] = x
    input_data[:, 1:] = input_nodes
    for k in range(len(x)):
        in_val = input_data[k, :]
        with np.errstate(invalid='raise'):
            try:
                preds[k] = run_output(individual, in_val)
            except (OverflowError, FloatingPointError):
                preds[k] = np.nan
    return correlation(preds, y_real)


def replacement(parents, offspring1, offspring2):
    childrens = []
    eval = []
    childrens.append(offspring1)
    childrens.append(offspring2)
    population = parents + childrens
    for k in range(len(population)):
        eval.append(fitness(population[k], y_real, x))
    id_worst = np.argmax(eval)
    del population[id_worst]
    del eval[id_worst]
    id_worst2 = np.argmax(eval)
    del population[id_worst2]
    del eval[id_worst2]
    return population


Path(f"../output/swap/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle

parents = generate_population()
# evolution
generation = []
fit2 = []
for g in range(1, generations + 1):
    fitnesses = np.zeros(pop_size, )
    fitnesses = [fitness(individual, y_real, x) for individual in parents]
    best = np.argmin(fitnesses)
    individual1, individual2, best_fit_id, parent2_id = tournament_elitism_selection(parents, fitnesses,
                                                                                     tournament_size=4)
    offspring1 = swap_mutation(individual1)
    offspring2 = swap_mutation(individual2)
    parents = replacement(parents, offspring1, offspring2)
    generation.append(g)
    fit2.append(fitnesses[best])
best_individual = fitnesses[best]
with open(f"../output/swap/{func_name}/log/output_{t}.pkl", "wb") as f:
    pickle.dump(fit2, f)
    pickle.dump(best_individual, f)
