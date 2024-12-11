import matplotlib.pyplot as plt
from sharpness import *
from cgp_fitness import *
import numpy as np
from numpy import random
from functions import *
from copy import copy
from cgp_fitness import corr
func = Levy(1)
np.random.seed(420)
points = 10
#sharpness of the original problem
x = np.linspace(-5, 5, points)
y = np.array([func(x1) for x1 in x])
#print(x)
#print(y)
std_dev_y = np.nanstd(y)
print(f'std_dev_y: {std_dev_y}')
epsilon = 0.2
fitness = 0
x_s = copy(x)
for i in range(len(x_s)):
    x_s[i] += random.uniform(-epsilon*std_dev_y, epsilon*std_dev_y)
x_s = np.sort(x_s)
#print(x_s)
y_s = np.array([func(x_) for x_ in x_s])
#print(y_s)
new_fit = corr(y_s, y)
print(f'fitness of function: {fitness}')
print(f'fitness after perturbation: {new_fit}')
sharpness_in = np.abs(fitness-new_fit)
print(f'sharpness-in: {np.abs(fitness-new_fit)}')

#line
flat_y = np.zeros(points)
flat_fit = corr(flat_y, y)

print(f'fitness of line: {flat_fit}')
fig, ax = plt.subplots(figsize = (10, 10))
ax.plot(x, y, color='black', label='ground truth')
ax.plot(x_s, y_s, color='blue', label='perturbed function')
ax.plot(x, flat_y, color='red', label='flat line')
ax.legend()
ax.set_title(f'function sharpness: {sharpness_in:.3e}')
plt.savefig('sharp_test.png')
