
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
def U(x,d,beta, beta_list, t, configurations):
    while (configurations-20) % t != 0:
        configurations = configurations + 1
    x_array = np.zeros(int(configurations))  # storage to be filled with values of x 
    for n in range(configurations):       
        x_array[n] = met_alg(x, d, beta)
    new_x_array = np.zeros(configurations-20)
    for n in range(20,configurations):
        new_x_array[n-20] = x_array[n]
    U_array = np.zeros(int((len(new_x_array)/t)))
    placeholder = np.zeros(int(t))
    Z_array = np.zeros(len(new_x_array))
    for n in range(len(new_x_array)):
        Z_array[n] = np.exp(-beta*H(new_x_array[n]))
    Z = sum(Z_array)/len(Z_array)
    for n in range(int((len(new_x_array)/t))):
        for i in range(t):
            placeholder[i] = H(new_x_array[t*n+i]) * np.exp(-beta*H(new_x_array[t*n+i]))
        U_array[n] = sum(placeholder)/(len(placeholder))
    U = sum(U_array)/(Z*len(U_array))
    delta_U = np.sqrt(sum((U_array - U)**2)/(len(U_array)*(len(U_array)-1)))
    result = np.zeros((len(beta_list)+1,3))
    result[0,0] = beta
    result[0,1] = U
    result[0,2] = delta_U
    for b in range(1,len(beta_list)+1):
        beta_new = beta_list[b-1]
        num_array = np.zeros(int((len(new_x_array)/t)))
        den_array = np.zeros(int((len(new_x_array)/t)))
        placeholder1 = np.zeros(int(t))
        placeholder2 = np.zeros(int(t))
        for n in range(int((len(new_x_array)/t))):
            for i in range(t):
                placeholder1[i] = H(new_x_array[t*n+i]) * np.exp(-(beta_new-beta)*H(new_x_array[t*n+i])) *np.exp(-beta*H(new_x_array[t*n+i]))
                placeholder2[i] = np.exp(-(beta_new-beta)*H(new_x_array[t*n+i])) *np.exp(-beta*H(new_x_array[t*n+i]))
            num_array[n] = sum(placeholder1)/(len(placeholder1))
            den_array[n] = sum(placeholder2)/(len(placeholder2))
        numerator = sum(num_array)/len(num_array)
        denominator = sum(den_array)/len(den_array)
        result[b,0] = beta_new
        result[b,1] = numerator/denominator
        result[b,2] = np.sqrt((sum(((np.divide(num_array,den_array)-result[b,1]))**2)/(len(num_array)*(len(num_array)-1))))
    return result

configurations = 100000
x = 1
d = 1
max_tau = 20
beta = 50
beta_list = [40,42,44,46,48,50,52,54,56,58,60]
MC_data = np.zeros(len(beta_list),3)
for n in range(len(beta_list)+1):
    MC_data[n,0] = U(x,d,beta_list[n], [], 20, configurations)[0,0]
    MC_data[n,1] = U(x,d,beta_list[n], [], 20, configurations)[0,1]
    MC_data[n,2] = U(x,d,beta_list[n], [], 20, configurations)[0,2]
re_beta_list = [40,42,44,46,48,52,54,56,58,60]
reweight_data = U(x,d,beta, re_beta_list, 20, configurations)
plt.figure()
plt.subplot(1,2,1)
plt.errorbar(MC_data[:,0],MC_data[:,1],MC_data[:,2],'b*')
plt.xlabel("Inverse temperature beta")
plt.ylabel("Average energy <U>")
plt.title("Pure Monte Carlo data")
plt.subplot(1,2,2)
plt.errorbar(reweight_data[:,0],reweight_data[:,1],reweight_data[:,2],'g*')  
plt.title("Reweighted data")
plt.xlabel("Inverse temperature beta")
plt.ylabel("Average energy <U>") 
"""U_data = np.zeros((max_tau-1,2))
for t in range (1,max_tau):
    ext_U_array = U(x,d,beta,beta_list,t,configurations)
    U_data[t-1,0] = ext_U_array[0,1]
    U_data[t-1,1] = ext_U_array[0,2]


#plt.plot(range(1,max_tau),U_data[:,1])
#plt.errorbar(range(1,max_tau),U_data[:,0], U_data[:,1])"""
        
