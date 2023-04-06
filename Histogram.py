# -*- coding: utf-8 -*-


import numpy as np

def H(x): # defining the Hamiltonian
    H = x**2
    return H

configurations = 100000
x = 1
d = 1
b = 50

def met_alg(x, d, beta): 
    # d = amount that x is changed by 
    x_new = x + np.random.uniform(-d,d) # interval of fixed width, 2d 
    H_change = H(x_new) - H(x)
    exp = np.exp(-beta*H_change)
    if exp > 1:
        return x_new
    elif exp < 1:
        if np.random.uniform(0,1) <= exp:
            return x_new
        else:
            return x
    
x_array = np.zeros(int(configurations)) # storage to be filled with values of x 

for n in range(configurations):
    x = met_alg(x, d, b)
    x_array[n] = x
    
print(x_array)
xax = np.zeros(int(configurations))
for n in range(configurations):
    xax[n] = n+1
import matplotlib.pyplot as plt
plt.hist(x_array, bins = 20, density = True)
plt.xlabel('H(x)')
plt.ylabel('Probability density')
plt.title("Gaussian distribution")
plt.savefig('Gaussian_Distribution.pdf')
