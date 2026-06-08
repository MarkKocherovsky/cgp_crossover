import argparse
from analysis_helper import Method, Metric, AnalysisToolkit

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#parser = argparse.ArgumentParser(prog='Analysis.py')
#parser.add_argument("--conf_comp", type=str2bool, default=False, help="Are we comparing configs of a single problem and runtype?")



base_path = "/mnt/gs21/scratch/kocherov/Documents/cgp/output/"
graph_path = "../../output/"
crossover_methods = {
    'Canonical': Method('None', 'Canonical', 'Canonical', 'blue', 'solid'),
     'N-Point': Method('n_point', 'One-Point', 'One-Point', 'green', 'solid'),
    'Subgraph': Method('subgraph', 'Subgraph', 'Subgraph', 'orange', 'solid'),
    'Semantic N-Point': Method('homologous_semantic_n_point', 'Semantic One-Point', 'Semantic One-Point', 'darkred', 'solid'),
    'Aligned Semantic N-Point': Method('aligned_homologous_semantic_n_point', 'Aligned Semantic One-Point', 'Aligned Semantic One-Point', 'brown', 'solid'),

}

metrics = {
    'Minimum Fitness': Metric('min_fitness', 'Minimum Fitness', r"\mathrm{Median}(\min(\mathrm{f}))", True),
    'Median Fitness': Metric('med_fitness', 'Median Fitness', r"\mathrm{Median}(\mathrm{Median}(\mathrm{f}))", True),
    'Minimum Test Fitness': Metric('min_test_fitness', 'Minimum Test Fitness', r"\mathrm{Median}(\min(\mathrm{f}))", True),
    'Median Test Fitness': Metric('med_test_fitness', 'Median Test Fitness', r"\mathrm{Median}(\mathrm{Median}(\mathrm{f}))", True),
    'Best Model Size': Metric('best_model_size', 'Best Model Size', r"\mathrm{Median}(\mathrm{Best Model Size})", False),
    'Median Model Size': Metric('median_model_size', 'Median Model Size', r"\mathrm{Median}(\mathrm{Median Model Size})", False),
    'Semantic Diversity': Metric('semantic_diversity', 'Semantic Diversity',r"\mathrm{Median}(\mathrm{Semantic Diversity})", True),
}
best_fitnesses = Metric('min_fitnesses', 'Best Fitness', r"$\mathrm{Median}(\min(\mathrm{f}))$", True)
best_test_fitnesses = Metric('min_test_fitnesses', 'Best Test Fitness', r"$\mathrm{Median}(\min(\mathrm{f}))$", True)
best_sizes = Metric('best_sizes', 'Best Sizes', "Active Nodes", True)
best_test_sizes = Metric('best_test_sizes', 'Best Test Sizes', "Active Nodes", True)
# canonical will always have 'elite'
selection_methods = {
                      'paretotournament': 'Pareto Tournament',
#                      'competent_tournament': 'Competent Tournament'
                    }
mutation_methods = {
                      'Point': '',
                      'Full': 'full'
}
# key->name
problems = {
    #'Koza3_1d': 'Koza 3',
    #'Nguyen5_1d': 'Nguyen 5',
    #'Nguyen7_1d': 'Nguyen 7',
    #'Ackley_1d': 'Ackley',
    #'Levy_1d': 'Levy',
    #'Rastrigin_1d': 'Rastrigin',
    #'Diabetes_1d': 'Diabetes',
    #'HeartDisease_1d': 'Heart Disease',
    #'ConcreteStrength_1d': 'Concrete Strength',
    #'ComputerHardware_1d': 'Computer Hardware',
    #'CreditApproval_1d': 'Credit Approval',
    'EnergyEfficiency_1d': 'Energy Efficiency'
}
"""
problems = {
    'Diabetes_1d': 'Diabetes',
    'Abalone_1d': 'Abalone',
    'Airfoil_1d': 'Airfoil',
    'California_1d': 'California'
}
"""

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

analyzer = AnalysisToolkit(crossover_methods, selection_methods, base_path, problems, metrics, 50, 6001, output_format = '.pdf')
analyzer.compile_averages([0, 2, 5, 7, 10, 13, 16], restart=False, mutation='full')
exit()
for selection in (selection_methods.keys()):
    analyzer.plot_box_plots(selection, best_test_fitnesses, f'minimum_test_fitness_{selection}_box_graph',
                             'Fitness of Best Models - Test Set', 'Crossover Methods', 'Best Fitness', log=True, violin=False, jitter=True)
    analyzer.plot_box_plots(selection, best_fitnesses, f'minimum_fitness_{selection}_box_graph',
                             'Fitness of Best Models', 'Crossover Methods', 'Best Fitness', log=True, violin=False, jitter=True)
    analyzer.plot_box_plots(selection, best_sizes, f'best_sizes_{selection}_box_graph',
                             'Number of Active Nodes in Best Models', 'Crossover Methods', 'Active Nodes', log=False, violin=False, jitter=True)
    #analyzer.plot_box_plots(selection, best_sizes, f'best_test_sizes_{selection}_box_graph',
    #                         'Number of Active Nodes in Best Models - Test Set', 'Crossover Methods', 'Active Nodes', log=False, violin=False, jitter=True)


    analyzer.get_median_end_values(metric='best_sizes', save_path=f'../output/graphs/median_end_sizes_{selection}.csv')
    #analyzer.get_median_end_values(metric='best_test_sizes', save_path=f'../output/graphs/median_end_test_sizes_{selection}.csv')
    analyzer.get_median_end_values(save_path=f'../output/graphs/median_end_fitnesses_{selection}.csv')
    analyzer.get_median_end_values(metric='min_test_fitnesses', save_path=f'../output/graphs/median_end_test_fitnesses_{selection}.csv')
    #analyzer.get_significance_tables()
    #analyzer.plackett_luce('../output/graphs/min_test_fitnesses_rankings.csv', '../output/graphs/placket_luce.csv')
    for metric in list(metrics.values()):
        analyzer.plot_line_graph(selection, metric, f'{metric.full_name.lower().replace(" ", "_")}_{selection}_graph',
                             f'{metric.full_name} Over Generations', 'Generations', rf'${metric.short_name}$', log=metric.log)

