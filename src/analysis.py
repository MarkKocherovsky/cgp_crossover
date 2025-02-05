import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from analysis_helper import Method, AnalysisToolkit

base_path = "../output/"

crossover_methods = {
    'Canonical': Method('None', 'CGP(1+4)', 'CGP(1+4)', 'blue'),
    'N-Point': Method('n_point', 'CGP(24+24)-1x', 'CGP(24+24) - One Point', 'green'),
    'Uniform': Method('uniform', 'cgp(24+24)-Ux', 'CGP(24+24) - Uniform', 'deeppink'),
    'Subgraph': Method('subgraph', 'CGP(24+24)-SGx', 'CGP(24+24) - Subgraph', 'orange'),
    'Semantic N-Point': Method('semantic_n_point', 'CGP(24+24)-S1x', 'CGP(24+24) - Semantic One Point',
                               'turquoise'),
    'Semantic Uniform': Method('semantic_uniform', 'CGP(24+24)-SUx', 'CGP(24+24) - Semantic Uniform', 'mediumvioletred')

}

# canonical will always have 'elite'
selection_methods = {'elite_tournament': 'Elite Tournament'}

metrics = ['Min Fitness', 'Median Fitness', 'Best Model Size', 'Median Model Size', 'Semantic Diversity',
           'Min Similarity', 'Median Similarity']

# key->name
problems = {
    'Koza1_1d': 'Koza 1',
    'Koza2_1d': 'Koza 2',
    'Koza3_1d': 'Koza 3',
    #'Nguyen4_1d': 'Nguyen 4',
    #'Nguyen5_1d': 'Nguyen 5',
    'Nguyen6_1d': 'Nguyen 6',
    'Nguyen7_1d': 'Nguyen 7',
    #'Ackley_1d': 'Ackley',
    'Levy_1d': 'Levy',
    'Rastrigin_1d': 'Rastrigin',
    'Griewank_1d': 'Griewank'

}

# Path structure:
# Output
#   Problem
#       Crossover
#           Selection
#               Trial 0..n
#                   best_model.csv
#                   statistics.csv
#                   xover_density_beneficial.csv
#                   xover_density_deleterious.csv
#                   xover_density_neutral.csv

analyzer = AnalysisToolkit(crossover_methods, selection_methods, problems, metrics, 50, 1000)
analyzer.compile_averages()
