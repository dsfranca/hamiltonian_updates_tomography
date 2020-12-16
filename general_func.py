#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:47:50 2020

@author: danielstilckfranca
"""



#this file contains general use functions for the code


import numpy as np
from general_func import *


#generates the hermitian conjugate of a matrix
def hermitian(V):
    return np.conj(np.transpose(V))


#genreates a n/k local random unitary on n qubits.
def rand_unitary(n,k):
    #divide n qubits into k groups
    #locality is then l
    l=int(n/k)
    
    A=np.random.normal(size=[2**l,2**l])+1j*np.random.normal(size=[2**l,2**l])
    [q,r]=np.linalg.qr(A)
    for m in range(1,k):
        A=np.random.normal(size=[2**l,2**l])+1j*np.random.normal(size=[2**l,2**l])
        [q1,r]=np.linalg.qr(A)
        q=np.kron(q,q1)
    return q



    
#generates the density matrix of a erp pair of n qubits
    
    
def generate_epr(n):
    m=int(n/2)
    omega=np.zeros([2**n,2**n])
    for k in range(0,2**(m)):
        for j in range(0,2**(m)):
            blob=np.zeros([2**(m),2**(m)])
            blob[k,j]=1
            blob=np.kron(blob,blob)
            omega=omega+blob
    return omega/(2**(m))





