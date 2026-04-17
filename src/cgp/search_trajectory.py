from dataclasses import dataclass
from .cgp_model import CGP
import numpy as np

@dataclass
class StnInd:
    def __init__(self, semantics: list | np.ndarray, fitness: float, complexity: float):

        self.semantics = semantics
        self.fitness = fitness
        self.complexity = complexity
        self.id = self.__hash__()

    def to_dict(self):
        return {
            "id": self.id,
            "semantics": np.asarray(self.semantics).tolist(),
            "fitness": self.fitness,
            "complexity": self.complexity,
        }

    def __eq__(self, other):
        return np.array_equal(np.asarray(self.semantics), np.asarray(other.semantics))

    def __ne__(self, other):
        return not np.array_equal(np.asarray(self.semantics), np.asarray(other.semantics))

    def __hash__(self):
        return hash(tuple(np.asarray(self.semantics).ravel().tolist()))

class STN:
    def __init__(self):
        # dictionary is structured as hash (ind.id): individual (ind)
        self.stn = {}


    def _add_to_network(self, stn: StnInd):
        self.stn[stn.id] = stn

    def get_semantics(self, model: CGP, x: list | np.ndarray):
        semantics = model(x)
        fitness = model.fitness
        complexity = model.complexity

        new_candidate = StnInd(semantics, fitness, complexity)
        new_id = new_candidate.id
        if new_id not in self.stn:
            self._add_to_network(new_candidate)

    def print(self):
        print(self.stn)

    def to_dict(self):
        return {
            str(k): v.to_dict()
            for k, v in self.stn.items()
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def get_target(self, y: list | np.ndarray):
        y = np.asarray(y).reshape(-1, 1)
        return StnInd(y, 0.0, 0.0).to_dict()