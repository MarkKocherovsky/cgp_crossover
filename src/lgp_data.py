from lgp_mutation import macromicro_mutation
from lgp_select import lgp_tournament_elitism_selection


class LgpConfig:
    def __init__(self, key: str, run_name: str, label: str, xover, mut, sel, dp):
        self.key = key
        self.run_name = run_name
        self.label = label
        self.xover = xover
        self.mutation = mut
        self.selection = sel
        self.drift_parents = dp

    def __call__(self):
        return self.run_name, self.label, self.xover, self.mutation, self.selection, self.drift_parents


class LgpCollection:
    def __init__(self):
        # Define the configurations as a list of tuples

        configs = [
            ('lgp_uniform', 'lgp', 'LGP-Ux (40+40)', 'Uniform', macromicro_mutation, lgp_tournament_elitism_selection,
             'TwoParent'),
            ('lgp_1x', 'lgp_1x', 'LGP-1x (40+40)', 'OnePoint', macromicro_mutation, lgp_tournament_elitism_selection,
             'TwoParent'),
            ('lgp_2x', 'lgp_2x', 'LGP-2x (40+40)', 'TwoPoint', macromicro_mutation, lgp_tournament_elitism_selection,
             'TwoParent'),

        ]

        # Populate the dictionary using dictionary comprehension
        self.lgp_dict = {
            identifier: LgpConfig(identifier, path, description, crossover_type, mutation, selection, drift_parents)
            for identifier, path, description, crossover_type, mutation, selection, drift_parents in configs}

    def __call__(self):
        return self.lgp_dict
