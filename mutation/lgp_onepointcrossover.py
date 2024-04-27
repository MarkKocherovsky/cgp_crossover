import numpy as np
from numpy import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
from numpy import sin, cos, tan, pi, sqrt
from pathlib import Path
from sys import path

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

rules = 10  # number of instructions
operand = 2  # number of sources per instructions
arity = 1  # number of inputs
destination = 4  # number of destinations
cross_prob = 0.5  # crossover probability
constants = np.arange(0, 10, 1)
numbers = constants.shape[0]  # number of constants
pop = 80  # population size
mut_prob = 1 / pop  # mutation probability
generations = 10000
# p_mut = [0.1, 0.01]
output_index = 0  # output register
input_indices = np.arange(1, arity + 1, 1)
tournament = 4  # size of tournament


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def multiply(x, y):
    return x * y


def divide(x, y):
    return x / np.sqrt(1 + y ** 2)


def get_functions():
    return add, subtract, multiply, divide


def average(list):
    return sum(list) / len(list)


operators = get_functions()
operators_string = ("+", "-", "*", "/")


def get_registers():
    possible_indices = np.arange(0, destination + numbers + arity + 1)
    possible_destinations = np.delete(possible_indices, np.arange(1, numbers + arity + 1))
    possible_sources = possible_indices
    return possible_destinations, possible_sources


def generate_instruction(possible_sources, possible_destinations):
    instruction = np.zeros((2 + operand,), dtype=np.int32)
    instruction[0] = random.choice(possible_destinations)
    instruction[1] = random.randint(0, len(operators))
    instruction[2:] = random.choice(possible_sources, (operand,))
    return instruction


def generate_ind():
    instruction_count = random.randint(2, rules)
    instructions = np.zeros((instruction_count, (2 + operand)))
    possible_destinations, possible_sources = get_registers()
    for i in range(instruction_count):
        instructions[i, :] = generate_instruction(possible_sources, possible_destinations)
    return instructions


def generate_single_ind():
    instruction_count = 1
    instructions = np.zeros((instruction_count, (2 + operand)))
    possible_destinations, possible_sources = get_registers()
    for i in range(instruction_count):
        instructions[i, :] = generate_instruction(possible_sources, possible_destinations)
    return instructions


def one_point_xover(parent1, parent2):
    if random.random() > cross_prob:
        children1 = parent1.copy()
        children2 = parent2.copy()
    else:
        pnt = [random.randint(1, len(parent1)), random.randint(1, len(parent2))]
        parent11 = parent1[:pnt[0]].copy()
        parent12 = parent1[pnt[0]:].copy()
        parent21 = parent2[:pnt[1]].copy()
        parent22 = parent2[pnt[1]:].copy()
        children1 = np.concatenate((parent11, parent22), axis=0)
        children2 = np.concatenate((parent21, parent12), axis=0)
        if len(children1) > rules:
            idxs = np.array(range(0, rules))
            to_del = random.choice(idxs, ((len(children1) - rules),), replace=False)
            children1 = np.delete(children1, to_del, axis=0)
        if len(children2) > rules:
            idxs = np.array(range(0, rules))
            to_del = random.choice(idxs, ((len(children2) - rules),), replace=False)
            children2 = np.delete(children2, to_del, axis=0)
    return children1, children2


def mutation(individual):
    children = np.array([])
    if random.random() <= mut_prob:
        if len(individual) <= 2:
            mutation = random.randint(1, 3)
        else:
            mutation = random.randint(0, 3)
        if mutation == 0:  # remove instruction
            inst = random.randint(0, len(individual))
            children = (np.delete(individual, inst, axis=0))
        elif mutation == 1:  # change instruction
            inst = random.randint(0, len(individual))
            part = random.randint(0, len(individual[inst]))
            possible_destinations, possible_sources = get_registers()
            if part == 0:  # destination
                individual[inst, part] = random.choice(possible_destinations)  # destination index
            elif part == 1:  # operator
                individual[inst, part] = random.randint(0, len(operators))
            else:  # source
                individual[inst, part] = random.choice(possible_sources)
            children = individual.copy()
        elif mutation == 2:  # insert a new instruction
            inst = random.randint(0, len(individual))
            new_inst = generate_single_ind()
            individual = np.insert(individual, inst, new_inst, axis=0)
            children = individual.copy()
    else:
        children = individual.copy()
    return children


parents = []
for i in range(0, pop):
    parents.append(generate_ind())
data_bias = np.zeros((x.shape[0], constants.shape[0] + 1))
data_bias[:, 0] = x
data_bias[:, 1:] = constants


def corr(pred, y_real):
    if any(np.isnan(pred)) or any(np.isinf(pred)):
        return np.PINF
    r = pearsonr(pred, y_real)[0]
    if np.isnan(r):
        r = 0
    return 1 - r ** 2


def fitness(individual, y_real, x):
    pred = np.zeros((len(x),))
    for i in range(len(x)):
        registers = np.zeros((1 + arity + numbers + destination,))
        registers[1:arity + numbers + 1] = data_bias[i, :]
        for j in range(len(individual)):
            operation = individual[j].astype(int)
            dest = operation[0]
            operator = operators[operation[1]]
            sources = operation[2:]
            registers[dest] = operator(registers[sources[0]], registers[sources[1]])
        pred[i] = registers[0]
    return corr(pred, y_real)


def lgp_tournament_elitism_selection(parents, fitnesses, tournament=4):
    best_fit_id = np.argmin(fitnesses)
    parents1 = (parents[best_fit_id])
    id = np.array(range(0, len(parents)))
    idx = np.delete(id, best_fit_id, axis=0)
    parents2_id = random.choice(idx, (tournament,), replace=False)
    fit_parents2 = []
    for i in parents2_id:
        fit_parents2.append(fitnesses[i])
    winner = np.argmin(fit_parents2)
    winner_id = parents2_id[winner]
    parents2 = parents[winner_id]
    return parents1, parents2, best_fit_id, winner_id


def lgp_replacement(parents, offspring1, offspring2):
    childrens = []
    eval = []
    childrens.append(offspring1)
    childrens.append(offspring2)
    pop = parents + childrens
    for k in range(len(pop)):
        eval.append(fitness(pop[k], y_real, x))
    id_worst = np.argmax(eval)
    del pop[id_worst]
    del eval[id_worst]
    id_worst2 = np.argmax(eval)
    del pop[id_worst2]
    del eval[id_worst2]
    return pop


Path(f"../output/lgp_one/{func_name}/log/").mkdir(parents=True, exist_ok=True)
import pickle

# evolution
generation = []
fit2 = []
for g in range(1, generations + 1):
    fitnesses = np.zeros(pop, )
    fitnesses = [fitness(individual, y_real, x) for individual in parents]
    best = np.argmin(fitnesses)
    individual1, individual2, best_fit_id, id2 = lgp_tournament_elitism_selection(parents, fitnesses, tournament=4)
    children1, children2 = one_point_xover(individual1, individual2)
    offspring1 = mutation(children1)
    offspring2 = mutation(children2)
    parents = lgp_replacement(parents, offspring1, offspring2)
    generation.append(g)
    fit2.append(fitnesses[best])
best_individual = fitnesses[best]
with open(f"../output/lgp_one/{func_name}/log/output_{t}.pkl", "wb") as f:
    pickle.dump(fit2, f)
    pickle.dump(best_individual, f)
