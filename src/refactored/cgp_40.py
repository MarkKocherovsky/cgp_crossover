# cgp_40.py
from cgp_utils import *

print("started")
t = int(argv[1]) #trial
print(f'trial {t}')
max_g = int(argv[2]) #max generations
print(f'generations {max_g}')
max_n = int(argv[3]) #max body nodes
print(f'max body nodes {max_n}')
max_p = int(argv[4]) #max parents
print(f'Parents {max_p}')
max_c = int(argv[5]) #max children
print(f'children {max_c}')
outputs = 1
inputs = 1
biases = np.arange(0, 10, 1).astype(np.int32)
bias = biases.shape[0] #number of biases
print(f'biases {biases}')
arity = 2
p_mut = float(argv[8])
p_xov = float(argv[9])
random.seed(t+200)
bank = (add, sub, mul, div)
bank_string = ("+", "-", "*", "/")

run_name = 'cgp_40'
func_bank = Collection()
func = func_bank.func_list[int(argv[6])]
func_name = func_bank.name_list[int(argv[6])]
train_x = func.x_dom
train_y = func.y_test
print(train_x)

f = int(argv[7])
fits = FitCollection()
fit = fits.fit_list[f]
print(f)
print(fits.fit_list)
fit_name  = fits.name_list[f]
print('Fitness Function')
print(fit)
print(fit_name)

alignment = np.zeros((max_p+max_c, 2))
alignment[:, 0] = 1.0

train_x_bias = np.zeros((train_x.shape[0], biases.shape[0]+1))
train_x_bias[:, 0] = train_x
train_x_bias[:, 1:] = biases
print(train_x_bias)

mutate = basic_mutation
select = tournament_elitism
parents = initialize_population(max_p, max_n, bank, first_body_node=11)

fitness_objects = [Fitness() for i in range(0, max_p+max_c)]
fitnesses, alignment_a, alignment_b = calculate_fitness(fitness_objects, train_x_bias, train_y, parents, max_p)
print(np.round(fitnesses, 4))

sharp_in_manager = SAM_IN(train_x_bias)
sharp_out_manager = SAM_OUT()

noisy_x, noisy_y = get_noise(train_x_bias.shape, max_p+max_c, inputs, func, sharp_in_manager)
sharpness = calculate_sharpness(fitness_objects, noisy_x, noisy_y, parents, max_p)
sharp_in_list = [np.mean(sharpness)]
sharp_in_std = [np.std(sharpness)]

preds = [fitness_objects[i](train_x_bias, train_y, parent, opt=1)[0] for i, parent in zip(range(0, max_p), parents)]
neighbor_map = np.array([get_neighbor_map(pred, sharp_out_manager, fitness_objects[i]) for i, pred in zip(range(0, max_p), preds)])
print(neighbor_map.shape)
sharp_out_list = [np.mean(np.std(neighbor_map, axis=1)**2)] #variance
sharp_out_std = [np.std(np.std(neighbor_map, axis=1))]

print(np.round(sharpness, 4))
print(np.round(np.std(neighbor_map, axis=1)**2, 4))

fit_track = []
ret_avg_list = [] #best parent best child
ret_std_list = []

avg_change_list = [] #best parent best child
avg_hist_list = []
std_change_list = []

best_i = np.argmin(fitnesses[:max_p])
p_size = [cgp_active_nodes(parents[best_i][0], parents[best_i][1], opt=2)]

mut_impact = MutationImpact(neutral_limit=0.1)
num_elites = 7 #for elite graph plotting

for g in range(1, max_g+1):
	children, retention, density_distro = xover(deepcopy(parents), density_distro, method='TwoPoint') 
	children = mutate(deepcopy(children))
	pop = parents + children
	fit_temp = np.array([fitness_objects[i](train_x_bias, train_y, ind) for i, ind in zip(range(0, max_p+max_c), pop)])
	fitnesses = fit_temp[:, 0].copy().flatten()
	alignment_a = fit_temp[:, 1].copy()
	alignment_b = fit_temp[:, 2].copy()
	parents = select(pop, fitnesses, max_p)

