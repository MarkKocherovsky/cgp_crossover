import pickle
import numpy as np
import matplotlib.pyplot as plt
path = '../output/cgp_dnc_OnePoint/Koza 1/log/output_0.pkl' 
log = 1

with open(path, 'rb') as f:
    data = [pickle.load(f) for _ in range(20)]

mut_list = data[-5]
print(mut_list)
#np.savetxt('mut_list_d.csv', mut_list['d'][:, 63:66])

fig, ax = plt.subplots(figsize=(18, 8))

cax = ax.imshow(mut_list['n'].T, aspect='auto', origin='lower', cmap='hot')
fig.colorbar(cax, ax=ax, orientation='vertical', label='Count')
#                                extent=[0, 10000, 0, 193], cmap='viridis')
ax.set_xticks(range(0, mut_list['n'].shape[0]))
fig.savefig(f"mut_list_n.png", format='png')
