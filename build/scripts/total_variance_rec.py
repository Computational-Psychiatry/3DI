#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:31:05 2023

@author: v
"""
import numpy as np
import cvxpy as cp
from time import time
from sklearn.impute import KNNImputer

import sys

e0path = sys.argv[1]
epath = sys.argv[2]
morphable_model = 'BFMmm-19830'

if len(sys.argv) > 3:
    morphable_model = sys.argv[3]
    
e1 = np.loadtxt(e0path)

imputer = KNNImputer(n_neighbors=2, weights="uniform")

for i in range(e1.shape[1]):
    e1[:,i:i+1] = imputer.fit_transform(e1[:,i:i+1])


T = e1.shape[0]
K = e1.shape[1]

sdir = './models/MMs/%s' % morphable_model 
li = np.loadtxt(f'{sdir}/li.dat').astype(int)


EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li,:]
EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li,:]
EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li,:]
E = np.concatenate((EX, EY, EZ), axis=0)

def create_expression_sequence(epsilons, E):
    ps = []
    for t in range(epsilons.shape[0]):
        epsilon = epsilons[t,:]
        p = ((E @ epsilon)).reshape(-1,1)
        ps.append(p)
    return np.array(ps)[:,:,0]


p = create_expression_sequence(e1, E)
p0 = p

W = 120
num_wins = int(T/W)+1
Es = E

es = []
xprev = None
lastpart = False
for ti in range(num_wins):
    time0 = time()
    t0 = ti*W
    tf = (ti+1)*W+1
    
    if t0 >= T:
        break
    
    if tf >= T:
        tf = T
        W = tf-t0
        lastpart = True
    
    #print('%d vs %d' % (t0, tf))
    
    pc = p[t0:tf,:].T    
    x = cp.Variable((K,W+1-int(lastpart)))
    
    objective = cp.Minimize(cp.sum(cp.norm(x[:,:W-int(lastpart)]-x[:,1:W+1-int(lastpart)],2,axis=1)))
    constraints = [cp.norm(pc-(Es@x[:,:W+1-int(lastpart)]),2,axis=0) <= 2.75*np.ones((W+1-int(lastpart),))]
    
    if ti > 0:
        constraints.append(x[:,0] == xprev[:,-1])
    
    prob = cp.Problem(objective, constraints)
    x.value = e1[t0:tf,:].T
    
    result = prob.solve(solver='SCS')
    #result = prob.solve()
    xprev = x.value
    
    if lastpart:
        es.append(x.value.T)
    else:
        es.append(x.value[:,:W].T)

ecomb = np.concatenate(es,)
Tnew = ecomb.shape[0]

np.savetxt(epath, ecomb)

