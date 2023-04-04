# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
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
    # for reweighting maybe put in U function
    
xax = np.zeros(int(configurations))
for n in range(configurations):
    xax[n] = n+1
import matplotlib.pyplot as plt
plt.hist(x_array, bins = 50, density = True)

    
def U(beta, tau):
    
    z_array = np.zeros(round(int(configurations)/tau)) 
    U_array = np.zeros(round(int(configurations)/tau))
    # once all is generally working, find out why this doesnt work for some tau values 
    du_array = np.zeros(round(int(configurations)/tau))
    placeholder1 = np.zeros(int(tau))
    placeholder2 = np.zeros(int(tau))
    for n in range(round(int(configurations/tau))):
        for i in range(tau):
            placeholder1[i] = np.exp(-beta*H(x_array[tau*n+1]))
            placeholder2[i] = H(x_array[tau*n+1]) * np.exp(-beta*H(x_array[n*tau+1]))
        z_array[n] = sum(placeholder1)/len(placeholder1)
        U_array[n] = sum(placeholder2)/len(placeholder2)
    Z = sum(z_array)/len(z_array)
    plt.plot(range(len((U_array))),U_array/Z)
    U = sum(U_array)/(len(U_array)*Z)
    delta_U = np.sqrt(sum((U_array-U)**2)/(len(U_array)*(len(U_array)-1)))
    result = np.zeros((1,2))
    result[0,0] = U
    result[0,1] = delta_U
    return result


tau = np.array(i for t in range(1,10)[0,1]
#plt.plot(configurations/tau,U(beta, tau)
matplotlib.pyplot.plot(configurations/tau, U(beta,tau))







