# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
def met_alg(x, d, beta): 
    
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
        
def H(x): 
    H = x**2
    return H

def U(x,d,beta, t, configurations):
    while (configurations-50) % t != 0:
        configurations = configurations + 1
        
    x_array = np.zeros(int(configurations))  # storage to be filled with values of x 
    
    for n in range(configurations):
        x_array[n] = met_alg(x, d, beta)
        
    new_x_array = np.zeros(configurations-20)
    
    for n in range(20,configurations):
        new_x_array[n-50] = x_array[n]
        
    U_array = np.zeros(int((len(new_x_array)/t)))
    placeholder = np.zeros(int(t))
    Z_array = np.zeros(len(new_x_array))
    
    for n in range(len(new_x_array)):
        Z_array[n] = np.exp(-beta*H(new_x_array[n]))
        
    Z = sum(Z_array)/len(Z_array)
    
    for n in range(int((configurations-50)/t)):
        for i in range(t):
            placeholder[i] = H(new_x_array[n+i]) * np.exp(-beta*H(new_x_array[n+i]))
        U_array[n] = sum(placeholder)/(len(placeholder))
        
    U = sum(U_array)/(Z*len(U_array))
    delta_U = np.sqrt(sum((U_array - U)**2)/(len(U_array)*(len(U_array)-1)))
    
    result1 = np.zeros((1,2))
    result1[0,0] = U
    result1[0,1] = delta_U
    
    beta_list = np.zeros(50) # should be changed, just to get code to work
    for i in range(len(beta_list)):
        beta_list[i] = 50 + i
        
    for b in range(len(beta_list)):
        beta_prime = beta_list[b]
        numerator_array = np.zeros(int(len(new_x_array)/t))
        denominator_array = np.zeros(int(len(new_x_array)/t)) # both are the same size 
        placeholder1 = np.zeros(int(t)) # empty array of size t
        placeholder2 = np.zeros(int(t)) # the same
        
        for n in range(int((configurations-50)/t)):
            
            for i in range(t):
                placeholder1[i] = H(new_x_array[n+i]) * np.exp(-(beta_prime-beta)*H(new_x_array[n+i])) * np.exp(-beta*H(new_x_array[n+i]))
                placeholder2[i] = np.exp(-(beta_prime-beta)*H(new_x_array[n+i])) *np.exp(-beta*H(new_x_array[n+i]))
                
            numerator_array[n] = sum(placeholder1)/len(placeholder1)
            denominator_array[n] = sum(placeholder2)/len(placeholder2)
            
        numerator = sum(numerator_array)/len(numerator_array)
        denominator = sum(denominator_array)/len(denominator_array)
        
        result2 = np.zeros((len(beta_list),2), dtype = float)
        for b in range(len(beta_list)):
            #result2 = np.zeros((b,2))
            result2[b,0] = beta_prime
            result2[b,1] = numerator/denominator

    return result1, result2

configurations = 100000
x = 1
d = 1
max_tau = 20
beta = 50

        
    
U_data = np.zeros((max_tau-1,2))
for t in range (1,max_tau):
    ext_U_array = U(x,d,beta,t,configurations)
    U_data[t-1,0] = ext_U_array[0,0]
    U_data[t-1,1] = ext_U_array[0,1]


#plt.plot(range(1,max_tau),U_data[:,1])
plt.errorbar(range(1,max_tau),U_data[:,0], U_data[:,1])
