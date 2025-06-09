from analysis_helper import Method, Metric, AnalysisToolkit

base_path = "../output/"

crossover_methods = {
    'Canonical': Method('None', 'CGP(1+4)', 'CGP(1+4)', 'blue', 'solid'),
    'N-Point': Method('n_point', 'CGP(40+40)-1x', 'CGP(40+40) - One Point', 'green', 'solid'),
    'N-Point 1D': Method('n_point_1d', 'CGP(40+40)-1x1D', 'CGP(40+40) - One Point 1D', 'limegreen', 'dotted'),
    'Uniform': Method('uniform', 'cgp(40+40)-Ux', 'CGP(40+40) - Uniform', 'deeppink', 'solid'),
    'Uniform 1D': Method('uniform_1d', 'cgp(40+40)-Ux1D', 'CGP(40+40) - Uniform 1D', 'hotpink', 'dotted'),
    'Subgraph': Method('subgraph', 'CGP(40+40)-SGx', 'CGP(40+40) - Subgraph', 'orange', 'solid'),
    'Semantic N-Point': Method('semantic_n_point', 'CGP(40+40)-S1x', 'CGP(40+40) - Semantic One Point', 'red', 'solid'),
    'Aligned Semantic N-Point': Method('aligned_semantic_n_point', 'CGP(40+40)-AS1x', 'CGP(40+40) - Aligned Semantic One Point', 'darkred', 'dashed'),
    'Aligned Homologous Semantic N-Point': Method('aligned_homologous_semantic_n_point', 'CGP(40+40)-AHS1x', 'CGP(40+40) - Aligned Homologous Semantic One Point', 'lightcoral', 'dashdot'),
    'Semantic Uniform': Method('semantic_uniform', 'CGP(40+40)-SUx', 'CGP(40+40) - Semantic Uniform', 'brown', 'solid'),
    'Aligned Semantic Uniform': Method('aligned_semantic_uniform', 'CGP(40+40)-ASUx', 'CGP(40+40) - Aligned Semantic Uniform', 'indianred', 'dashed'),
    # 'Aligned Homologous Semantic Uniform': Method('aligned_homologous_semantic_uniform', 'CGP(40+40)-AHSUx', 'CGP(40+40) - Aligned Homologous Semantic Uniform', 'tomato', 'dashdot'),
    'Homologous Semantic N-Point': Method('homologous_semantic_n_point', 'CGP(40+40)-HS1x', 'CGP(40+40) - Homologous Semantic One Point', 'salmon', 'solid'),
    'Homologous Uniform N-Point': Method('homologous_semantic_uniform', 'CGP(40+40)-HSUx', 'CGP(40+40) - Homologous Semantic Uniform', 'purple', 'solid'),
    #'Semantic DNC Uniform': Method('dnc_semantic_uniform', 'CGP(40+40)-DNCUx', 'CGP(40+40) - DNC Semantic Uniform', 'indigo', 'dashed'),

}

metrics = {
    'Minimum Fitness': Metric('min_fitness', 'Minimum Fitness', r"\mathrm{Median}(\min(\mathrm{f}))", True),
    'Median Fitness': Metric('med_fitness', 'Median Fitness', r"\mathrm{Median}(\mathrm{Median}(\mathrm{f}))", True),
    'Best Model Size': Metric('best_model_size', 'Best Model Size', r"\mathrm{Median}(\mathrm{Best Model Size})", False),
    'Median Model Size': Metric('median_model_size', 'Median Model Size', r"\mathrm{Median}(\mathrm{Median Model Size})", False),
    'Semantic Diversity': Metric('semantic_diversity', 'Semantic Diversity',r"\mathrm{Median}(\mathrm{Semantic Diversity})", False),
}
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
    'Griewank_1d': 'Griewank'
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

analyzer = AnalysisToolkit(crossover_methods, selection_methods, problems, metrics, 50, 3001)
#analyzer.compile_averages([0, 2, 5, 8, 11])
for selection in (selection_methods.keys()):
    analyzer.plot_box_plots(selection, metrics['Minimum Fitness'], f'minimum_fitness_{selection}_box_graph',
                             'Fitness of Best Models', 'Crossover Methods', 'min_fitness', log=True, violin=False, jitter=True)
    for metric in list(metrics.values()):
        analyzer.plot_line_graph(selection, metric, f'{metric.full_name.lower().replace(" ", "_")}_{selection}_graph',
                             f'{metric.full_name} Over Generations', 'Generations', f'metric.short_name', log=metric.log)

