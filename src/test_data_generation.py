import numpy as np
import pickle
from sys import path
from pathlib import Path
from functions import *
from numpy import random

func_bank = Collection().func_list

n_test = 20 #0.33*20

dataset = []
#indices = random.choice(np.arange(0, 20.1, 1), n_test, replace=False).astype(np.int32)
for f in func_bank:
	dataset.append(np.array([f.x_dom, f.y_test]))
	print(f)
	print('x', list(np.sort(f.x_dom)))
	x_args = np.argsort(f.x_dom)
	print('y', list(f.y_test[x_args]))
	print('---')
dataset = np.array(dataset)
print(dataset)
with open(f"test_set.pkl", "wb") as f:
	pickle.dump(dataset, f)

