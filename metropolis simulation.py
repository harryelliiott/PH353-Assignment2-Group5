# -*- coding: utf-8 -*-


import numpy as np
import random

def H(x): # defining the Hamiltonian
    H = x**2
    return H

configurations = 100000
x = 0
d = 0.1

def met_alg(x_initial, d): 
    # d = amount that x is changed by 
    x_new = x + np.random.uniform(-d,d) # interval of fixed width, 2d 
    H_change = H(x_new) - H(x)
    if H_change < 0:
        return x_new
    elif H_change > 0:
        return x 
    
x_array = np.zeros(int(configurations)) # storage to be filled with values of x 

for n in range(configurations):
    x = met_alg(x, d)
    x_array[n] = x
    
print(x_array)