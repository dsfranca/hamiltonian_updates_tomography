# hamiltonian_updates_tomography
Code used to generate data for paper "Fast and robust quantum state tomography from few basis measurements".

It implements the "Hamiltonian updates" algorithm in Python to learn random pure states and EPR pairs, see https://arxiv.org/abs/2009.08216 for more details. 
It supports basis measurements with arbitrary locality structures. It also supports different noise models (amplitude damping, shot noise and others) for the samples.
We implement both the "last step recycling" procedure and the "total recycling" procedure, although the implementation is not optimized and works by computing the Gibbs state explicitely.
The geneterate_comp_noise.py file exemplifies a script to generate data. It generates data to compare the performance under several different noise models .
