
import numpy as np
import matplotlib.pyplot as plt
import random

def H(x): # defining the Hamiltonian
    H = x**2
    return H

configurations = 100000
x = 1
d = 0.1
t = np.array
beta = 5
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
    x = met_alg(x, d, beta)
    x_array[n] = x


def U(beta, t):
    z_array = np.zeros(round(int(configurations)/t))
    U_array = np.zeros(round(int(configurations)/t))
    du_array = np.zeros(round(int(configurations)/t))
    placeholder1 = np.zeros(int(t))
    placeholder2 = np.zeros(int(t))
    for n in range(round(int(configurations/t))):
        for i in range(t):
            placeholder1[i] = np.exp(-beta*H(x_array[t*n+1]))
            placeholder2[i] = H(x_array[4*n+1]) * np.exp(-beta*H(x_array[4*t+1]))
        z_array[n] = sum(placeholder1)/len(placeholder1)
        U_array[n] = sum(placeholder2)/len(placeholder2)
    plt.plot(range(len((U_array))),U_array)
    Z = sum(z_array)/len(z_array)
    plt.plot(range(len((U_array))),U_array/Z)
    U = sum(U_array)/(len(U_array)*Z)
    delta_U = np.sqrt(sum((U_array-U)**2)/(len(U_array)*(len(U_array)-1)))
    result = np.zeros((1,2))
    result[0,0] = U
    result[0,1] = delta_U
    return result

plt.plot(configurations/t,U(beta, int(i for t in range(1,10)))[0,1])
        
print(x_array)