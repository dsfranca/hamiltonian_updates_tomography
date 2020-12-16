#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:41:34 2020

@author: danielstilckfranca
"""


import numpy as np 
import random as random
import pandas as pd
#import matplotlib.pyplot as plt
import math
from general_func import *

from scipy.linalg import expm, sinm, cosm, sqrtm
from scipy import sparse
import time
from randomstate import random_state


#this class stores the statistics in a basis and a unitary descrbing it
class statistics:
    def __init__(self,p,U):
        self.p=p
        self.U=U

#this is one of the main working horses. it is a giobbs state plus functions to calculate estatistics 
class gibbs_state:
    #n is the number of qubits
    def __init__(self,n):
        self.n=n
        #this stores the target statistics
        self.list_statistics=[]
        self.beta=0
        self.measurements=0
    
    #will mostly be used to set the Hamiltonian  ot 0
    def initialize_hamiltonian(self,H):
        self.Hamiltonian=H
    #generates a density matirx from Hamiltonian. Stores it in self.density. Note the minus sign
    #for the Gibbs state.
    def generate_density(self):
        self.density=expm(-self.Hamiltonian)
        self.density=self.density/np.trace(self.density)
    
    #generates a density matrix through a taylor expansion
    def generate_density_taylor(self):
        degree=int(9*(self.beta)+1)
        current=np.eye(2**self.n)
        power=current
        for k in range(1,degree):
            power=power.dot(-self.Hamiltonian)/k
            current=current+(power)
            
        self.density=current/np.trace(current)
    
    #gets the statistics of Gibbs state in basis defined via U
    def generate_statistics(self,U):
        self.generate_density()
        new=U.dot(self.density)
        new=new.dot(hermitian(U))

        new=np.diag(new)

        return new
    
    #generates the statistics through a tayor series approximation
    def generate_statistics_taylor(self,U):
        self.generate_density_taylor()
        new=U.dot(self.density)
        new=new.dot(hermitian(U))

        new=np.diag(new)

        return new
    
    
    #updates the Hamiltonian in the new_dir by a step-size of delta
    def update_hamiltonian(self,new_dir,delta):
        
        
        
        self.Hamiltonian=self.Hamiltonian-(delta/2)*new_dir
        self.beta+=(delta/2)
        
    #adds new statitistics of the targeet state to our list
    def add_statistics(self,p,U):
        self.list_statistics.append(statistics(p,U))
        self.new=1
        
    #implements the total recycling step, i.e. checks every basis measured so far.
    def check_old(self,epsilon):
        found=0
        for basis in self.list_statistics:
            
            q=basis.p
        
            tv_dist=100
            repeat=0
            while (tv_dist>epsilon):
                
                repeat+=1
                p=self.generate_statistics(basis.U)
            
        
                dist=p-q
                tv_dist=np.sum(np.abs(dist))
            
                if tv_dist>epsilon:
                    
                    found=1
                    entries=dist>0
                    projector=np.zeros([2**n])
                    for k in range(0,len(entries)):
                        if entries[k]:
                            projector[k]=1
                    projector=np.diag(projector)
                    new_dir=hermitian(basis.U).dot(projector)
                    new_dir=new_dir.dot(basis.U)
                
                    self.update_hamiltonian(-new_dir,tv_dist/2)
            
        return found



