import copy
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if module_path not in sys.path:
    sys.path.append(module_path)
import random
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm
#---cgp stuff---
from effProg import cgp_active_nodes, percent_change
from similarity import find_similarity
from cgp_fitness import DriftImpact, corr
from cgp_parents import *
from sharpness import *
from cgp_operators import *
from helper import *
from cgp_mutation import basic_mutation
from ga_auxiliary import create_output_folder, save_list, save_dict, is_in_dict, get_fit, \
    save_fitness, n_point_crossover_pairs, n_point_mutate, tournament_selection
import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)

class SelectionGA:

    def __init__(self, n_generations: int, population_size: int, crossover_prob: float, mutation_prob: float,
                 ind_length: int, min_selection_val, max_selection_val, random_state: int = 42, tournament_size=5,
                 save_every_n_generations: int = 10, crossover_func: callable = n_point_crossover_pairs,
                 flip_mutation_prob=0.005, save_population_info=False,
                 save_fitness_info=False, elitism=False, n_parents=2, target_function=None):
        """
        :param n_generations: Number of generations to run
        :param population_size: Population Size
        :param crossover_prob: Crossover probability
        :param mutation_prob:  Mutation probability
        :param ind_length: Individual length
        :param random_state: Initial random seed
        """
        assert 0 <= crossover_prob <= 1, "ILLEGAL CROSSOVER PROBABILITY"
        assert 0 <= mutation_prob <= 1, "ILLEGAL MUTATION PROBABILITY"
        assert population_size > 1, "Population size must be at least 2"
        assert n_generations > 0, "Number of generations must be a positive integer"
        assert ind_length > 0, "Illegal individual length"

        # params
        self.n_generations = n_generations
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_state = random_state
        self.save_every_n_generations = save_every_n_generations
        self.flip_mutation_prob = flip_mutation_prob
        self.min_selection_val = min_selection_val
        self.max_selection_val = max_selection_val
        self.tournament_size = tournament_size
        self.crossover_func = crossover_func
        self.ind_length = ind_length
        self.save_population_info = save_population_info
        self.save_fitness_info = save_fitness_info
        self.elitism = elitism
        self.n_parents = n_parents
        self.target_function = target_function
        # params

        self.fitness_dict = {}
        self.pop_dict = {}

        self.generation_metrics = {
            'std': [],
            'mean': [],
            'median': [],
            'max': [],
            'min': [],
            'time': []
        }

    def reset_stats(self) -> None:
        """
        Resets saved-metrics.
        :return: None
        """
        self.fitness_dict = {}
        self.pop_dict = {}
        for key in self.generation_metrics.keys():
            self.generation_metrics[key] = []

    def update_progress_arrays(self, fits, g, gen_start, output_path):
        """
        :param fits: list of fitness scores
        :param g: current generation number
        :param gen_start: time that this generation started
        :param output_path: output to save the data
        :return: None
        """
        generation_time = time.time() - gen_start
        fits = np.array(fits)

        self.generation_metrics['mean'] += [fits.mean()]
        self.generation_metrics['std'] += [fits.std()]
        self.generation_metrics['median'] += [np.median(fits)]
        self.generation_metrics['max'] += [np.max(fits)]
        self.generation_metrics['min'] += [np.min(fits)]
        self.generation_metrics['time'] += [generation_time]

        self.save_all_data(g, output_path)

    def save_all_data(self, curr_generation: int, output_folder: str) -> None:
        """
        Saves metrics for the given generations in a given path.
        :param curr_generation: Current generation (int)
        :param output_folder: folder to save data in.
        :return: None
        """
        with open(output_folder + "am_alive", "w") as f:
            f.write("Still running at generation :" + str(curr_generation) + "\n")

        is_last_gen = (curr_generation == (self.n_generations - 1))
        if (curr_generation % self.save_every_n_generations != 0) and not is_last_gen:
            return

        print(f"saving data for {curr_generation=}")

        if is_last_gen:
            if self.save_fitness_info:
                written_dict = {str(k): v.tolist() for k, v in self.fitness_dict.items()}
                save_dict(written_dict, output_folder + "fitness_dict.json")

            if self.save_population_info:
                save_dict({gen_num: pop.tolist() for gen_num, pop in self.pop_dict.items()},
                          output_folder + "gens_dict.json")

        for metric_name, metric_list in self.generation_metrics.items():
            save_list(metric_list, output_folder + metric_name + ".txt")

    def init_population(self):
        """
        Creates a random individual with a given length
        :param population_size:
        :param length_to_gen: How many '1/0' to generate
        :return: A random binary array of the given size
        """
        return generate_parents(self.population_size, self.ind_length, bank = (add, sub, mul, div)) #things like first_body_node or arity aren't a concern right now

    def __init_seeds(self) -> None:
        """
        Resets seeds back to the initial seed
        :return: None
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(self.random_state)

    def fit(self, output_folder, x, y, fitness_func, crossover_func=None, stopping_func=None):
        self.reset_stats()
        self.__init_seeds()

        if crossover_func is not None:
            self.crossover_func = crossover_func

        create_output_folder(output_folder)

        return self.__run_gens(output_folder, x, y, fitness_func, stopping_func=stopping_func)

    def __prepare_individual(self, ind: list):
        return np.concatenate((ind[0].flatten(), np.array(ind[1])))

    def __reshape_individual(self, pop: list[np.ndarray], shape, out_num):
        pop = np.array(pop)
        pop = pop.reshape(-1, pop.shape[2])
        return [(np.array(p[:-out_num]).reshape(shape), np.atleast_1d(p[-out_num:])) for p in pop]

    def crossover(self, population, crossover_prob):
        """
        Apply crossover and mutation on the offsprings
        """
        pairs_to_cross = []
        next_gen = []
        retention = []
        parent_shape = population[0][0].shape
        distro = []
        out_num = len(population[0][1])
        crossover_masks = np.random.uniform(size=len(population) // 2)
        extended_crossover_parent_indexes = np.random.randint(low=0, high=len(population),
                                                         size=(len(population) // 2, self.n_parents - 2))
        crossover_index = 0
        for child1, child2, cross_mask in zip( population[::2], population[1::2], crossover_masks):
            child1 = self.__prepare_individual(list(child1).copy())
            child2 = self.__prepare_individual(list(child2).copy())
            if cross_mask < crossover_prob:
               current_tuple = [child1.copy(), child2.copy()]
               if self.n_parents > 2:
                   current_tuple += [self.__prepare_individual(list(population[extended_crossover_parent_indexes[crossover_index, i]]).copy()) for i in range(extended_crossover_parent_indexes.shape[1])]
               pairs_to_cross.append(current_tuple)
            crossover_index += 1
        if len(pairs_to_cross) < 1:
            next_gen = population
            return next_gen, [], distro
        crossed_pairs, new_distro = self.crossover_func(pairs_to_cross)
        crossed_parents_pairs = self.__reshape_individual(crossed_pairs, parent_shape, out_num)

        for i, child1, child2, cross_mask in zip(range(len(population)), population[::2], population[1::2], crossover_masks):
           if cross_mask < crossover_prob:
               next_gen.append(tuple(crossed_parents_pairs.pop(0)))
               next_gen.append(tuple(crossed_parents_pairs.pop(0)))
               retention.append(i)
               distro.append(np.array(new_distro))
           else:
               next_gen.append(tuple(list(child1).copy()))
               next_gen.append(tuple(list(child2).copy()))
               distro.append(np.zeros(new_distro.shape))
        distro = np.array(distro)
        next_gen = [(np.array(ind[0]), np.array(ind[1])) if not isinstance(ind, tuple) else ind for ind in next_gen]
        return next_gen, retention, distro


    def mutate(self, population, mutation_prob):
        """
        Apply mutation on the offsprings
        """

        return basic_mutation(population, self.mutation_prob)

    def mutate_individual(self, ind):
        """
        Mutate a single individual
        """
        return n_point_mutate(ind.copy(), self.flip_mutation_prob, self.min_selection_val,
                              self.max_selection_val)

    def evaluate_and_set_fitness(self, population, x, y, fitness_func):
        """
        Evaluate and set fitness for the given population
        """
        fitness_values = [fitness_func(x, y, ind) for ind in tqdm(population)]
        for ind, fit in zip(population, fitness_values):
            save_fitness(ind, fit, self.fitness_dict)

    def select(self, population):
        """
        Select the next generation
        """
        num_elitists = int(self.elitism)
        next_gen = tournament_selection(population, len(population) - num_elitists, self.tournament_size,
                                        self.fitness_dict)

        if num_elitists > 0:
            elitists = copy.deepcopy(np.array([ind for ind, _ in
                                               sorted(self.fitness_dict.items(), key=lambda x: x[1], reverse=True)[
                                               :num_elitists]]))
            next_gen = np.concatenate((next_gen, elitists), axis=0)

        return next_gen
    def getNoise(self, shape, pop_size, inputs, func, sharp_in_manager, opt = 0):
        x = []
        y = []
        if opt == 1:
            fixed_inputs = sharp_in_manager.perturb_data()[:, :inputs]
        for p in range(pop_size):
            noisy_x = np.zeros((shape))
            if opt == 1:
                noisy_x[:, :inputs] = deepcopy(fixed_inputs)
            else:
                noisy_x[:, :inputs] = sharp_in_manager.perturb_data()[:, :inputs]
            noisy_x[:, inputs:] = sharp_in_manager.perturb_constants()[:, inputs:]
            noisy_y = np.fromiter(map(func.func, list(noisy_x[:, :inputs].flatten())), dtype=np.float32)
            x.append(noisy_x)
            y.append(noisy_y)
        return np.array(x), np.array(y)
    def get_neighbor_map(self, preds, sharp_out_manager, train_y):
        neighborhood = sharp_out_manager.perturb(preds)
        return [corr(neighbor, train_y) for neighbor in neighborhood]
    def __run_gens(self, output_folder: str, x :np.ndarray, y:np.ndarray, fitness_func: callable, stopping_func: callable = None):
        """
        :param output_folder: Folder to save output data
        :param fitness_func: Function that receives an individual and returns its fitness
        :param stopping_func: Function that receives the current fitness and returns whether to stop
        :return: None
        """
        population = self.init_population()
        self.evaluate_and_set_fitness(population, x, y, fitness_func)
        start_time = time.time()
        
        density_distro = initDensityDistro(self.ind_length, 1, 2, outputs=1)
        fit_track = []
        p_size = []
        ret_avg_list = [] #best parent best child
        ret_std_list = []

        avg_change_list = [] #best parent best child
        avg_hist_list = []
        std_change_list = []

        sharp_in_manager = SAM_IN(x)
        sharp_out_manager = SAM_OUT()
        sharp_in_list = []
        sharp_out_list = []
        sharp_in_std = []
        sharp_out_std = []
        mut_impact = DriftImpact(neutral_limit = 1e-3)
        print('starting generations')
        for generation_index in tqdm(range(0, self.n_generations)):
            #print(population)
            print("-- Generation %i --" % generation_index)

            if self.save_population_info:
                self.pop_dict[generation_index] = deepcopy(population)

            offspring = self.select(population)
            invalid_individuals = [ind for ind in offspring if not is_in_dict(ind, self.fitness_dict)]
            self.evaluate_and_set_fitness(invalid_individuals, x, y, fitness_func)
            fitness_values = [get_fit(ind, self.fitness_dict) for ind in offspring]

      
            parent_fitness = deepcopy(fitness_values)
            if stopping_func is not None and stopping_func(fitness_values):
                print('Stopping after %i generations' % generation_index)
                break
            #record best fit 
            best_i = np.argmin(fitness_values)
            best_fit = np.min(fitness_values)
            print('generation %i, best fitness: %f' % (generation_index, best_fit))
            fit_track.append(best_fit)
            p_size.append(cgp_active_nodes(offspring[best_i][0], offspring[best_i][1], opt = 2))
            #self.update_progress_arrays(fitness_values, generation_index, start_time, output_folder)
            start_time = time.time()
            parents = deepcopy(offspring)
            offspring, retention, d_distro = self.crossover(offspring, self.crossover_prob)
            offspring, mutated_inds = self.mutate(offspring, self.mutation_prob)

            invalid_individuals = [ind for ind in offspring if not is_in_dict(ind, self.fitness_dict)]
            self.evaluate_and_set_fitness(invalid_individuals, x, y, fitness_func)
            child_fitness = [get_fit(ind, self.fitness_dict) for ind in offspring]
            change_list = []
            full_change_list = []
            ret_list = []
            
            pop = parents + offspring
            fitnesses = np.concatenate((parent_fitness, child_fitness))
            max_p = len(parents)
            max_c = len(offspring)
            inputs = 1
            g = generation_index
            train_x_bias = x
            train_y = y

            drift_per_parent_mut, drift_per_parent_xov = mut_impact(parent_fitness+child_fitness, len(parent_fitness), retention, mutated_inds, opt=1)
            avg_hist_list, avg_change_list, std_change_list, ret_avg_list, ret_std_list = processRetention(retention, pop,
                                                                                                   fitnesses, max_p,
                                                                                                   avg_hist_list,
                                                                                                   avg_change_list,
                                                                                                   std_change_list,
                                                                                                   ret_avg_list,
                                                                                                   ret_std_list, g)

            fitness_objects, _= initFitness(max_p, max_c)
            sharp_in_list, sharp_in_std, sharp_out_list, sharp_out_std = processSharpness(train_x_bias, max_p, max_c,
                                                                                  inputs, self.target_function, sharp_in_manager,
                                                                                  fitness_objects, pop, train_y,
                                                                                  sharp_in_list, sharp_in_std,
                                                                                  sharp_out_manager, sharp_out_list,
                                                                                  sharp_out_std)
            density_distro = associateDistro(drift_per_parent_xov, retention, d_distro, density_distro, mode='dnc')            
            invalid_individuals = [ind for ind in offspring if not is_in_dict(ind, self.fitness_dict)]
            self.evaluate_and_set_fitness(invalid_individuals, x, y, fitness_func)
            population = offspring
        best_i = np.argmin(fitness_values)
        best_ind = population[best_i]
        return best_ind, fit_track, avg_change_list, ret_avg_list, p_size, avg_hist_list, mut_impact, sharp_in_list, sharp_out_list, sharp_in_std, sharp_out_std, density_distro
