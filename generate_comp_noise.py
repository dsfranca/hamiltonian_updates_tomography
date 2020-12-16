#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:17:18 2020

@author: danielstilckfranca
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:26:30 2020

@author: danielstilckfranca
"""

import numpy as np 
import random as random
import pandas as pd
#import matplotlib.pyplot as plt
import math


from scipy.linalg import expm, sinm, cosm, sqrtm
from scipy import sparse
import time
from randomstate import random_state



from gibbsstate import gibbs_state
from general_func import *
from update_functions import *




#set the total number of measurements
measurements=30
#matrix for the results
results=np.zeros([80,33])
run=0

#maximial number of qubits
m=8
#one loop for each noise model
#0 stands for AD
for n in range(m,m+1):
    #locality of the measurements is n/locality. E.g. 1 is global
    for locality in [1]: 
        #number of shots
        for trial in range(0,5):
           #error tolerance
            for epsilon in [0.03]:
                
                print('N is:',n,'epsilon is',epsilon,'trial is', trial)
                
                #switch for random state or epr
                a=random_state(n)
                #a.density=generate_epr(n)
                
                sigma=gibbs_state(n)
                sigma.initialize_hamiltonian(np.zeros([2**n,2**n]))
    #           
                sigma.generate_density()
                distances=[]
                distances.append(trial)
                distances.append(0)
    #            sigma.generate_density()
    #            
                distances.append(np.linalg.norm(sigma.density-a.density,ord='fro')/2)
                start=time.time()
                for k in range(0,measurements):
                    U=rand_unitary(n,locality)
                    
                    print("noiseless",k)
                    
                    sigma=first_update_guess_amp_damping(sigma,a,epsilon,U,0.01/(4*8))
                    #found=1 implements  full recycling
                    found=0
                    
                    while found==1:
                    
                        found=sigma.check_old(epsilon)
                    addition=np.linalg.norm(sigma.density-a.density,ord='fro')/2
                    distances.append(addition)
                    #ranks.append(np.trace(sqrtm(sigma.density))**2)
                    print("current distance is:",addition)
                finish=time.time()
                normal_way=finish-start
                quality=np.linalg.norm(sigma.density-a.density,ord='fro')/2
                print("final",quality,"time",normal_way)
                
                results[run,:]=distances
                run+=1
                results_df=pd.DataFrame(results)
  


                results_df.to_csv("comparisonrand"+str(measurements)+"repeat8.csv")
#1 stands for shot noise
for n in range(m,m+1):
    for locality in [1]: 
        for trial in range(0,5):
           
            for epsilon in [0.03]:
                
                print('N is:',n,'epsilon is',epsilon,'trial is', trial)
                a=random_state(n)
                #a.density=generate_epr(n)
                
                sigma=gibbs_state(n)
                sigma.initialize_hamiltonian(np.zeros([2**n,2**n]))
    #           
                sigma.generate_density()
                distances=[]
                distances.append(trial)
                distances.append(1)
    #            sigma.generate_density()
    #            
                distances.append(np.linalg.norm(sigma.density-a.density,ord='fro')/2)
                start=time.time()
                for k in range(0,measurements):
                    U=rand_unitary(n,locality)
                    
                    print("noiseless",k)
                    
                    sigma=first_update_guess_noise(sigma,a,epsilon,U,0.01/4)
                    
                    found=0
                    while found==1:
                    
                        found=sigma.check_old(epsilon)
                    addition=np.linalg.norm(sigma.density-a.density,ord='fro')/2
                    distances.append(addition)
                    #ranks.append(np.trace(sqrtm(sigma.density))**2)
                    print("current distance is:",addition)
                finish=time.time()
                normal_way=finish-start
                quality=np.linalg.norm(sigma.density-a.density,ord='fro')/2
                print("final",quality,"time",normal_way)
                
                results[run,:]=distances
                run+=1
                results_df=pd.DataFrame(results)
                results_df.to_csv("comparisonrand"+str(measurements)+"repeat8.csv")                
                
                
                
#2 for noiseless            
for n in range(m,m+1):
    for locality in [1]: 
        for trial in range(0,5):
           
            for epsilon in [0.03]:
                
                print('N is:',n,'epsilon is',epsilon,'trial is', trial)
                a=random_state(n)
                #a.density=generate_epr(n)
                
                sigma=gibbs_state(n)
                sigma.initialize_hamiltonian(np.zeros([2**n,2**n]))
    #           
                sigma.generate_density()
                distances=[]
                distances.append(trial)
                distances.append(2)
    #            sigma.generate_density()
    #            
                distances.append(np.linalg.norm(sigma.density-a.density,ord='fro')/2)
                start=time.time()
                for k in range(0,measurements):
                    U=rand_unitary(n,locality)
                    
                    print("noiseless",k)
                    
                    sigma=first_update_guess(sigma,a,epsilon,U)
                    
                    found=0
                    while found==1:
                    
                        found=sigma.check_old(epsilon)
                    addition=np.linalg.norm(sigma.density-a.density,ord='fro')/2
                    distances.append(addition)
                    #ranks.append(np.trace(sqrtm(sigma.density))**2)
                    print("current distance is:",addition)
                finish=time.time()
                normal_way=finish-start
                quality=np.linalg.norm(sigma.density-a.density,ord='fro')/2
                print("final",quality,"time",normal_way)
                
                results[run,:]=distances
                run+=1
                results_df=pd.DataFrame(results)
                results_df.to_csv("comparisonnoise"+str(measurements)+".csv")      
                

