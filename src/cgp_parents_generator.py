import pickle
from cgp_parents import *
from helper import loadBank

max_p = 40
max_n = 64
bank, bank_string = loadBank()
parents = generate_parents(max_p, max_n, bank, first_body_node=11, outputs=1, arity=2)
with open('data/cgp_parents.pkl', 'wb') as f:
    pickle.dump(parents, f)
