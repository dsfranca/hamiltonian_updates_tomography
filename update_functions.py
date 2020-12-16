#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:55:30 2020

@author: danielstilckfranca
"""

#this file contains the different update functions. That is, they take in a gibbs  state, unitary and target state
# and generate
# a new gibbs state under several different noise models. The method used to compute the Gibbs state is
#pedestrian, just exponentiating the  matrix.



import numpy as np 
import random as random
import pandas as pd
#import matplotlib.pyplot as plt
import math
from general_func import *
        
        

 #updates the Hamiltonian without noise given current Gibbs state current, target state target
#error epsilon and unitary U. The distinguishability parameter is epsilon        
            
        
def first_update_guess(current,target,epsilon,U):
    n=current.n

    q=target.get_statistics(U)
    current.add_statistics(q,U)
    tv_dist=100
    repeat=0
    while (tv_dist>epsilon):
        #print("I am at update",repeat)
        repeat+=1
        p=current.generate_statistics(U)

        dist=p-q
        tv_dist=np.sum(np.abs(dist))
        projector=np.zeros([2**n])
        if tv_dist>epsilon:
            entries=dist>0
            
            for k in range(0,len(entries)):
                if entries[k]:
                    projector[k]=1
        projector=np.diag(projector)
        new_dir=hermitian(U).dot(projector)
        new_dir=new_dir.dot(U)

        current.update_hamiltonian(-new_dir,tv_dist/2)
   
    return current    

 #updates the Hamiltonian with amplitude damping noise given current Gibbs state current, target state target
#error epsilon and unitary U. The distinguishability parameter is epsilon  . and the LOCAL amplitude damping
#rate is p_damp      
            
def first_update_guess_amp_damping(current,target,epsilon,U,p_damp):
    n=current.n
    
    
    q=target.get_statistics_amp_damping(U,p_damp)
    current.add_statistics(q,U)
    tv_dist=100
    repeat=0
    while (tv_dist>epsilon):
        #print("I am at update",repeat)
        repeat+=1
        p=current.generate_statistics(U)


        
        dist=p-q
        tv_dist=np.sum(np.abs(dist))
        projector=np.zeros([2**n])
        if tv_dist>epsilon:
            entries=dist>0
            
            for k in range(0,len(entries)):
                if entries[k]:
                    projector[k]=1
        projector=np.diag(projector)
        new_dir=hermitian(U).dot(projector)
        new_dir=new_dir.dot(U)

        current.update_hamiltonian(-new_dir,tv_dist/2)
   
    return current     


 #updates the Hamiltonian with shot noise given current Gibbs state current, target state target
#error epsilon and unitary U. The distinguishability parameter is epsilon. The standard deviation is p_noise
        
        
def first_update_guess_noise(current,target,epsilon,U,p_noise):
    current.generate_density()
    n=current.n

    q=target.get_statistics_one_shot_noise(U,p_noise)
    
    tv_dist=100
    repeat=0
    while (tv_dist>epsilon):
        
        repeat+=1
        p=current.generate_statistics(U)
        
    
        dist=p-q
        tv_dist=np.sum(np.abs(dist))
        
        if tv_dist>epsilon:
            entries=dist>0
            projector=np.zeros([2**n])
            for k in range(0,len(entries)):
                if entries[k]:
                    projector[k]=1
            projector=np.diag(projector)
            new_dir=hermitian(U).dot(projector)
            new_dir=new_dir.dot(U)

            current.update_hamiltonian(-new_dir,tv_dist/2)
    return current  


#updates with a fixed, heavy tail noise. particularly adversarial
def first_update_guess_heavy(current,target,epsilon,U):
    n=current.n

    q=target.get_statistics(U)
    current.add_statistics(q,U)
    tv_dist=100
    repeat=0
    while (tv_dist>epsilon):
        #print("I am at update",repeat)
        repeat+=1
        p=current.generate_statistics(U)
        noise=np.zeros([2**(current.n)])
        for l in range(0,2**(current.n)):
            noise[l]=(1/(l+1)**2)
#                
        noise=noise/np.sum(noise)
        p=(0.995)*p+0.005*noise
        
        dist=p-q
        tv_dist=np.sum(np.abs(dist))
        projector=np.zeros([2**n])
        if tv_dist>epsilon:
            entries=dist>0
            
            for k in range(0,len(entries)):
                if entries[k]:
                    projector[k]=1
        projector=np.diag(projector)
        new_dir=hermitian(U).dot(projector)
        new_dir=new_dir.dot(U)

        current.update_hamiltonian(-new_dir,tv_dist/2)
   
    return current    