from analysis_helper import Method, Metric, AnalysisToolkit

base_path = "/mnt/gs21/scratch/kocherov/Documents/cgp/output/"
graph_path = "../output/"
crossover_methods = {
    'Canonical': Method('None', 'CGP(1+4)', 'CGP(1+4)', 'blue', 'solid'),
    # 'Canonical_NP': Method('None/full', 'CGP(1+4)-NP', 'CGP(1+4) - Node Preservation', 'dodgerblue', 'dashed'),
    'N-Point': Method('n_point', 'CGP(40+40)-1x', 'CGP(40+40) - One Point', 'green', 'solid'),
    # 'N-Point 1D': Method('n_point_1d', 'CGP(40+40)-1x1D', 'CGP(40+40) - One Point 1D', 'limegreen', 'dotted'),
    'Uniform': Method('uniform', 'CGP(40+40)-Ux', 'CGP(40+40) - Uniform', 'deeppink', 'solid'),
    # 'Uniform 1D': Method('uniform_1d', 'CGP(40+40)-Ux1D', 'CGP(40+40) - Uniform 1D', 'hotpink', 'dotted'),
    'Subgraph': Method('subgraph', 'CGP(40+40)-SGx', 'CGP(40+40) - Subgraph', 'orange', 'solid'),
    'Semantic N-Point': Method('semantic_n_point', 'CGP(40+40)-S1x', 'CGP(40+40) - Semantic One Point', 'red', 'solid'),
    #'Aligned Semantic N-Point': Method('aligned_semantic_n_point', 'CGP(40+40)-AS1x', 'CGP(40+40) - Aligned Semantic One Point', 'darkred', 'dashed'),
    'Aligned Inverted Semantic N-Point': Method('aligned_homologous_semantic_n_point', 'CGP(40+40)-AIS1x', 'CGP(40+40) - Aligned Inverted Semantic One Point', 'lightcoral', 'dashdot'),
    #'Semantic Uniform': Method('semantic_uniform', 'CGP(40+40)-SUx', 'CGP(40+40) - Semantic Uniform', 'brown', 'solid'),
    'Aligned Semantic Uniform': Method('aligned_semantic_uniform', 'CGP(40+40)-ASUx', 'CGP(40+40) - Aligned Semantic Uniform', 'grey', 'dashed'),
    'Aligned Inverted Semantic Uniform': Method('aligned_homologous_semantic_uniform', 'CGP(40+40)-AISUx', 'CGP(40+40) - Aligned Inverted Semantic Uniform', 'olivedrab', 'dashdot'),
    #'Inverted Semantic N-Point': Method('homologous_semantic_n_point', 'CGP(40+40)-IS1x', 'CGP(40+40) - Inverted Semantic One Point', 'salmon', 'solid'),
    'Inverted Uniform N-Point': Method('homologous_semantic_uniform', 'CGP(40+40)-ISUx', 'CGP(40+40) - Inverted Semantic Uniform', 'purple', 'dashed'),
    #'DNC Uniform': Method('dnc_semantic_uniform', 'CGP(40+40)-DNCUx', 'CGP(40+40) - DNC Semantic Uniform', 'indigo', 'dashed'),
    #'DNC N-Point': Method('dnc_semantic_n_point', 'CGP(40+40)-DNC1x', 'CGP(40+40) - DNC Semantic One-Point', 'mediumorchid', 'dashed'),

}

metrics = {
    'Minimum Fitness': Metric('min_fitness', 'Minimum Fitness', r"\mathrm{Median}(\min(\mathrm{f}))", True),
    'Median Fitness': Metric('med_fitness', 'Median Fitness', r"\mathrm{Median}(\mathrm{Median}(\mathrm{f}))", True),
    'Minimum Test Fitness': Metric('min_test_fitness', 'Minimum Test Fitness', r"\mathrm{Median}(\min(\mathrm{f}))", True),
    'Median Test Fitness': Metric('med_test_fitness', 'Median Test Fitness', r"\mathrm{Median}(\mathrm{Median}(\mathrm{f}))", True),
    'Best Model Size': Metric('best_model_size', 'Best Model Size', r"\mathrm{Median}(\mathrm{Best Model Size})", False),
    'Median Model Size': Metric('median_model_size', 'Median Model Size', r"\mathrm{Median}(\mathrm{Median Model Size})", False),
    'Semantic Diversity': Metric('semantic_diversity', 'Semantic Diversity',r"\mathrm{Median}(\mathrm{Semantic Diversity})", False),
}
best_fitnesses = Metric('min_fitnesses', 'Best Fitness', r"$\mathrm{Median}(\min(\mathrm{f}))$", True)
best_test_fitnesses = Metric('min_test_fitnesses', 'Best Test Fitness', r"$\mathrm{Median}(\min(\mathrm{f}))$", True)
best_sizes = Metric('best_sizes', 'Best Sizes', "Active Nodes", True)
# canonical will always have 'elite'
selection_methods = {
                      'elite_tournament': 'Elite Tournament', 
#                      'competent_tournament': 'Competent Tournament'
                    }

# key->name
problems = {
    'Koza1_1d': 'Koza 1',
    'Koza2_1d': 'Koza 2',
    'Koza3_1d': 'Koza 3',
    'Nguyen4_1d': 'Nguyen 4',
    'Nguyen5_1d': 'Nguyen 5',
    'Nguyen6_1d': 'Nguyen 6',
    'Nguyen7_1d': 'Nguyen 7',
    'Ackley_1d': 'Ackley',
    'Levy_1d': 'Levy',
    'Rastrigin_1d': 'Rastrigin',
    #'Griewank_1d': 'Griewank'
}

# Path structure:
# Output
#   Problem
#       Crossover
#           Selection
#               Trial 0…n
#                   best_model.csv
#                   statistics.csv
#                   xover_density_beneficial.csv
#                   xover_density_deleterious.csv
#                   xover_density_neutral.csv

analyzer = AnalysisToolkit(crossover_methods, selection_methods, base_path, problems, metrics, 50, 3001, output_format = '.pdf')
analyzer.compile_averages([0, 2, 5, 7, 10, 13, 16])
for selection in (selection_methods.keys()):
    analyzer.plot_box_plots(selection, best_test_fitnesses, f'minimum_test_fitness_{selection}_box_graph',
                             'Fitness of Best Models', 'Crossover Methods', 'Best Fitness', log=True, violin=False, jitter=True)
    analyzer.plot_box_plots(selection, best_fitnesses, f'minimum_fitness_{selection}_box_graph',
                             'Fitness of Best Models', 'Crossover Methods', 'Best Fitness', log=True, violin=False, jitter=True)
    analyzer.plot_box_plots(selection, best_sizes, f'best_sizes_{selection}_box_graph',
                             'Number of Active Nodes in Best Models', 'Crossover Methods', 'Active Nodes', log=False, violin=False, jitter=True)

    analyzer.get_median_end_values(metric='best_sizes', save_path=f'../output/graphs/median_end_sizes_{selection}.csv')
    analyzer.get_median_end_values(save_path=f'../output/graphs/median_end_fitnesses_{selection}.csv')
    analyzer.get_median_end_values(metric='min_test_fitnesses', save_path=f'../output/graphs/median_end_test_fitnesses_{selection}.csv')
    analyzer.get_significance_tables()
    for metric in list(metrics.values()):
        analyzer.plot_line_graph(selection, metric, f'{metric.full_name.lower().replace(" ", "_")}_{selection}_graph',
                             f'{metric.full_name} Over Generations', 'Generations', rf'${metric.short_name}$', log=metric.log)
old_problem_names = {
    #'Koza1_1d': 'Koza 1',
    #'Koza2_1d': 'Koza 2',
    'Koza3_1d': 'Koza 3',
    #'Nguyen4_1d': 'Nguyen 4',
    'Nguyen5_1d': 'Nguyen 5',
    #'Nguyen6_1d': 'Nguyen 6',
    'Nguyen7_1d': 'Nguyen 7',
    #'Ackley_1d': 'Ackley_1D',
    #'Levy_1d': 'Levy_1D',
    'Rastrigin_1d': 'Rastrigin_1D',
    #'Griewank_1d': 'Griewank_1D'
}

old_method_names = {
    'None': 'cgp_base',
    'None': Method('cgp_base', 'CGP(1+4) Old', 'CGP(1+4) Old', 'lightsteelblue', 'solid'),
    'n_point': 'cgp_1x',
    'uniform': 'cgp_unx',
    'dnc_semantic_n_point': 'cgp_dnc_1x',
    'dnc_semantic_uniform': 'cgp_dnc',
    'subgraph': 'cgp_sgx',
}

#analyzer.plot_box_plots_compare_old_new(old_method_names, old_problem_names,'elite_tournament', best_fitnesses, f'minimum_fitness_box_graph_compare',
#                             'Fitness of Best Models\nOld vs New', 'Crossover Methods', 'Best Fitness', log=True, jitter=True)

#analyzer.plot_box_plots_compare_old_new(old_method_names, old_problem_names,'elite_tournament', best_sizes, f'best_sizes_box_graph_compare',
#                             'Size of Best Models\nOld vs New', 'Crossover Methods', 'Size of Best Models', log=False, jitter=True)


