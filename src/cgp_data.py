from cgp_mutation import mutate_1_plus_4, basic_mutation, macromicro_mutation
from cgp_selection import tournament_elitism, select_elite

class CgpConfig:
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



class CgpCollection:
    def __init__(self):
        # Define the configurations as a list of tuples

        configs = [
            ('cgp_basic', 'cgp', 'CGP (1+4)', None, mutate_1_plus_4, select_elite, 'OneParent'),
            ('cgp_40', 'cgp_40', 'CGP (16+64)', None, mutate_1_plus_4, tournament_elitism, 'OneParent'),
            ('cgp_1x', 'cgp_1x', 'CGP-1x', 'OnePoint', basic_mutation, tournament_elitism, 'TwoParent'),
            ('cgp_2x', 'cgp_2x', 'CGP-2x', 'TwoPoint', basic_mutation, tournament_elitism, 'TwoParent'),
            ('cgp_sgx', 'cgp_sgx', 'CGP-SGx', 'Subgraph', basic_mutation, tournament_elitism, 'TwoParent'),
            ('cgp_vlen', 'cgp_vlen', 'CGP-vlen', 'OnePointVlen', macromicro_mutation, tournament_elitism, 'TwoParent'),
            ('cgp_dnc', 'cgp_dnc', 'CGP-DNC', 'DNC', basic_mutation, tournament_elitism, 'TwoParent')
        ]

        # Populate the dictionary using dictionary comprehension
        self.cgp_dict = {identifier: CgpConfig(identifier, path, description, crossover_type, mutation, selection, drift_parents)
                         for identifier, path, description, crossover_type, mutation, selection, drift_parents in configs}

    def __call__(self):
        return self.cgp_dict