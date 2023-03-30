
import numpy as np
import matplotlib.pyplot as plt
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
def H(x): # defining the Hamiltonian
    H = x**2
    return H
def U(x,d,beta, t, configurations):
    while configurations % t != 0:
        configurations = configurations + 1
    x_array = np.zeros(int(configurations))  # storage to be filled with values of x 
    for n in range(configurations):
        
        x_array[n] = met_alg(x, d, beta)
    z_array = np.zeros(int(configurations/t))
    U_array = np.zeros(int(configurations/t))
    placeholder1 = np.zeros(int(t))
    placeholder2 = np.zeros(int(t))
    for n in range(int(configurations/t)):
        for i in range(t):
            placeholder1[i] = np.exp(-beta*H(x_array[t*n+i]))
            placeholder2[i] = H(x_array[t*n+i]) * np.exp(-beta*H(x_array[t*n+i]))
        z_array[n] = sum(placeholder1)/len(placeholder1)
        U_array[n] = sum(placeholder2)/len(placeholder2)
    Z = sum(z_array)/len(z_array)
    U = sum(U_array)/(len(U_array)*Z)
    delta_U = np.sqrt(sum((U_array-U)**2)/(len(U_array)*(len(U_array)-1)))
    result = np.zeros((1,2))
    result[0,0] = U
    result[0,1] = delta_U
    return result

configurations = 100000
x = 1
d = 0.1
max_tau = 20
beta = 5

        
    
U_data = np.zeros((max_tau-1,2))
for t in range (1,max_tau):
    ext_U_array = U(x,d,beta,t,configurations)
    U_data[t-1,0] = ext_U_array[0,0]
    U_data[t-1,1] = ext_U_array[0,1]


    
plt.errorbar(range(1,max_tau),U_data[:,0], U_data[:,1])
        
