# Cartesian Genetic Programming General Use Module

This repository contains the work done primarily by Rachel Kocherovsky for her PhD studies at Michigan State University. If you intend to use this code, please cite whichever of the following papers is more relevant to your work:

* Mark Kocherovsky, Wolfgang Banzhaf; July 22–26, 2024. "Crossover Destructiveness in Cartesian versus Linear Genetic Programming." Proceedings of the ALIFE 2024: Proceedings of the 2024 Artificial Life Conference. ALIFE 2024: Proceedings of the 2024 Artificial Life Conference. Online. (pp. 20). ASME. https://doi.org/10.1162/isal_a_00735

* Kocherovsky, M., Kianinejad, M., Bakurov, I., Banzhaf, W. (2025). On the Effectiveness of Crossover Operators in Cartesian Genetic Programming. In: Xue, B., Manzoni, L., Bakurov, I. (eds) Genetic Programming. EuroGP 2025. Lecture Notes in Computer Science, vol 15609. Springer, Cham. https://doi.org/10.1007/978-3-031-89991-1_5

* Kocherovsky, Mark, Illya Bakurov, and Wolfgang Banzhaf. "Node Preservation and its Effect on Crossover in Cartesian Genetic Programming." arXiv preprint arXiv:2511.00634 (2025).

---

# Installation:

To install, please clone the repository into your working directory. You may need to alter import paths according to the directory structure of your system.

Once cloned, you can navigate to cgp/src/cgp and run ``pip install -e .`` to install. After this, you can import as needed.

---

# Basic Tutorial:

The top-level evolution module is ``CartesianGP`` in ``cgp_evolver.py``. It is used as such:

```
evolution_module = CartesianGP(self, parents=1, children=4, max_generations=100, mutation='Point', selection='Elite',
                 xover=None, fixed_length=True, fitness_function='correlation_complexity', model_parameters=None,
                 function_bank=None, solution_threshold=0.005, checkpoint_filename='checkpoint.pkl', seed=42,
                 dnc_hp: dict = None, **kwargs)
```

**Parameters:**

``parents``: ``int``

Number of parent individuals to generate at start and select during each generation.

``children``: ``int``

Number of child individuals to generate during the crossover and/or mutation phase.

``max_generations``: ``int``

Number of generations for evolution. Checkpointing is turned on by default.

``mutation``: ``str``

Type of mutation to use. The options are:

* ``point``: Traditional point mutation. A random value within an instruction is chosen and randomly changed to another legal value.
* ``full``: "Full" or "Node" mutation. A random instruction is replaced by a new legal randomly generated instruction.

``selection``: ``str``

Selection algorithm to use. By default, diversity is forced, meaning that once an individual is chosen to be a parent, it cannot be selected again in the same generation. This ensures that population diversity is maintained, to a point.

The possible selection algorithms are:

* ``elite``: Elite selection. The top ``x`` individuals are selected to be parents.
* ``tournament``: Tournament selection. A set of ``x`` individuals are compared; the best of that subset is chosen to be a new parent. This repeats until the specified number of parents are chosen.
* ``elite_tournament``: Elite Tournament selection. The top ``x`` individuals are selected to be parents. The rest undergo tournament selection.
* ``competent_tournament``: Comptent Tournament Selection. **UNTESTED**. A geometric semantic method, please see T. P. Pawlak and K. Krawiec, "Competent Geometric Semantic Genetic Programming for Symbolic Regression and Boolean Function Synthesis," in Evolutionary Computation, vol. 26, no. 2, pp. 177-212, June 2018, doi: 10.1162/evco_a_00205. 
