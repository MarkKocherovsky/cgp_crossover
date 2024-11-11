import pandas as pd
import numpy as np
from numpy import random

class CGP:
    def __init__(self, model=None):
        self.model = model if model is not None else generate_model()