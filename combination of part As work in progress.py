# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:15:45 2023

@author: harry
"""

import numpy as np

def H(x): # defining the Hamiltonian
    H = x**2
    return H

configurations = 100000
x = 0
d = 0.1
b = 1

def met_alg(x_initial, d, b): 
    # d = amount that x is changed by 
    x_new = x + np.random.uniform(-d,d) # interval of fixed width, 2d 
    H_change = H(x_new) - H(x)
    if H_change < 0:
        return x_new
    elif np.exp(-(b)*H_change) > np.random.uniform(0,1):
        return x_new
    elif H_change > 0:
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
plt.hist(x_array, bins = 50, density = True)

def U(beta):
    t = 100000 # changed, placeholder, is probably more complex than this: think (monte carlot time) should be one unit for every configuration?
    # if so, then t = configurations
    z_array = np.zeros(round(int(configurations)/t))
    # unsure of what z_array is for, unused in the rest of the function
    U_array = np.zeros(round(int(configurations)/t)) # however this would then just be 1
    # only applies if t is supposed to be monte carlo time 
    placeholder1 = np.zeros(int(t)) 
    placeholder2 = np.zeros(int(t))
    #placeholders for what 
    for n in range(round(int(configurations/t))):
        for i in range(t):
            placeholder1[i] = np.exp(-beta*H(x_array[4*n+1]))
            placeholder2[i] = H(x_array[4*n+1]) * np.exp(-beta*H(x_array[4*n+1]))
        exp_array = np.zeros(10000)
        exp_array[n] = sum(placeholder1)/len(placeholder1)
        U_array[n] = sum(placeholder2)/len(placeholder2)
    Z = sum(exp_array)/len(exp_array)
    U = sum(U_array)/(len(U_array)*Z)
    delta_U = np.sqrt(sum((U_array-U)**2)/(len(U_array)*(len(U_array)-1)))
    result = np.zeros((1,2), dtype = float) # changed 
#    result[0,0],[0,1] = U,delta_U
    U, delta_U = result[0,0], result[0,1] # changed 
    return U, delta_U

# what is the above trying to achieve: finding internal energy as function of beta
# find out why is producing zeros 
# find out appropriate value for monte carlo time 