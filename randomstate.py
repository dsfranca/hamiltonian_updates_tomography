#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:01:04 2020

@author: danielstilckfranca
"""
import numpy as np
      
        
from general_func import *
#generates and comoptes statistics of random states, also under noise.

class random_state:
    def __init__(self,n):
        self.n=n
        #generates random normal vector
        self.ket=np.random.normal(size=[2**n,1])+1j*np.random.normal(size=[2**n,1])
        self.density=self.ket.dot(hermitian(self.ket))
        #generates the corresponding density
        self.density=self.density/np.trace(self.density)
        
    #generates noiseless statistics in U basis.
    def get_statistics(self,U):
        prob=U.dot(self.density)
        prob=prob.dot(hermitian(U))
        prob=np.diag(prob)
        
        return np.real(prob)
    
    
    
    #applies amplitude damping to rho with local prob. p_noise
    def apply_amplitude_damping(self,p_noise,rho):
        #define Kraus oprators
        A0=np.array([[1,0],[0,np.sqrt(1-p_noise)]])
        A1=np.array([[0,np.sqrt(p_noise)],[0,0]])
        
        #first qubit
        A00=np.kron(A0,np.eye(2**(self.n-1)))
        A10=np.kron(A1,np.eye(2**(self.n-1)))
        
        self.density_damp=A00.dot(rho.dot(hermitian(A00)))+A10.dot(rho.dot(hermitian(A10)))
        
        
        
        #last qubit
        A00=np.kron(np.eye(2**(self.n-1)),A0)
        A10=np.kron(np.eye(2**(self.n-1)),A1)    
        
        self.density_damp=A00.dot(self.density_damp.dot(hermitian(A00)))+A10.dot(self.density_damp.dot(hermitian(A10)))
        #qubits in between
        for k in range(1,self.n-1):
            A00=np.kron(np.eye(2**(k)),A0)
            A00=np.kron(A00,np.eye(2**(self.n-k-1)))
            A10=np.kron(np.eye(2**(k)),A1)
            A10=np.kron(A10,np.eye(2**(self.n-k-1)))
            self.density_damp=A00.dot(self.density_damp.dot(hermitian(A00)))+A10.dot(self.density_damp.dot(hermitian(A10)))
        
        return self.density_damp
        
    #generates statistics with LOCAL amplitude damping with rate p_noise
    def get_statistics_amp_damping(self,U,p_noise):
        prob=U.dot(self.density)
        prob=prob.dot(hermitian(U))
        prob=self.apply_amplitude_damping(p_noise,prob)
        prob=np.real(np.diag(prob))
        
        
        

        return np.real(prob)        
    #get statistics with shot noise
    def get_statistics_noise(self,U,p_noise):
        prob=U.dot(self.density)
        prob=prob.dot(hermitian(U))
        prob=np.diag(prob)
        noise=np.random.normal(size=[2**self.n])
        noise=np.abs(noise)
        noise=noise/np.sum(noise)
        
        

        return np.real((1-p_noise)*prob+p_noise*noise)
    
    def get_statistics_one_shot_noise(self,U,p_noise):
        prob=U.dot(self.density)
        prob=prob.dot(hermitian(U))
        prob=np.diag(prob)
        noise=np.random.normal(size=[2**self.n])
        prob=np.real(prob)
        prob=(1+p_noise*noise)*prob
        prob=prob/np.sum(prob)
        
        
        

        return prob    


        
        

        return prob    
    #gets the empirical distribution of q up to error epsilon in TV. Assumes full support
    #of the probability distribution.
    def get_empirical(self,q,epsilon):
        
        samples=np.random.choice(2**self.n, 50*int(epsilon**(-2))*(2**self.n), p=q)
        unique_elements, counts_elements = np.unique(samples, return_counts=True)
        print(counts_elements)
        return counts_elements/len(samples)

