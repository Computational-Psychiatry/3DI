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

# bpath = sys.argv[1]
# ppath = sys.argv[2]
ppath = sys.argv[1]
pnewpath = sys.argv[2]

pose = np.loadtxt(ppath)
morphable_model='BFMmm-19830'

if len(sys.argv) > 3:
    morphable_model = sys.argv[3] # BFMmm-19830
    
# e1 = np.loadtxt(bpath+'.expressions')
# p1 = np.loadtxt(ppath)[:,6:7]
trans1 = np.loadtxt(ppath)[:,0:3]
p1 = np.loadtxt(ppath)[:,3:6]
a1 = np.loadtxt(ppath)[:,6:]
# p1 = np.loadtxt(ppath)[:,2:3]
# i1 = np.loadtxt(bpath+'.illums')

imputer = KNNImputer(n_neighbors=2, weights="uniform")

for i in range(p1.shape[1]):
    p1[:,i:i+1] = imputer.fit_transform(p1[:,i:i+1])
    
for i in range(trans1.shape[1]):
    trans1[:,i:i+1] = imputer.fit_transform(trans1[:,i:i+1])

T = p1.shape[0]
K = p1.shape[1]

W = 120
num_wins = int(T/W)+1

ps = []
ts = []
pprev = None
tprev = None
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
    
    pc = p1[t0:tf,:].T    
    x = cp.Variable((K,W+1-int(lastpart)))
    
    objective = cp.Minimize(cp.sum(cp.norm(x[:,:W-int(lastpart)]-x[:,1:W+1-int(lastpart)],1,axis=1)))
    constraints = [cp.norm((pc-x[:,:W+1-int(lastpart)]),2,axis=0) <= 0.006*np.ones((W+1-int(lastpart),))]

    if ti > 0:
        constraints.append(x[:,0] == pprev[:,-1])
    
    prob = cp.Problem(objective, constraints)
    x.value = p1[t0:tf,:].T
    
    result = prob.solve(solver='SCS')
    pprev = x.value
    if lastpart:
        ps.append(x.value.T)
    else:
        ps.append(x.value[:,:W].T)
    
    tc = trans1[t0:tf,:].T    
    x = cp.Variable((K,W+1-int(lastpart)))
    
    objective = cp.Minimize(cp.sum(cp.norm(x[:,:W-int(lastpart)]-x[:,1:W+1-int(lastpart)],2,axis=1)))
    constraints = [cp.norm((tc-x[:,:W+1-int(lastpart)]),2,axis=0) <= 3.5*np.ones((W+1-int(lastpart),))]
    
    if ti > 0:
        constraints.append(x[:,0] == tprev[:,-1])
    
    prob = cp.Problem(objective, constraints)
    x.value = trans1[t0:tf,:].T
    
    result = prob.solve(solver='SCS')
    tprev = x.value
    
    if lastpart:
        ts.append(x.value.T)
    else:
        ts.append(x.value[:,:W].T)

pcomb = np.concatenate(ps,)
tcomb = np.concatenate(ts,)
Tnew = pcomb.shape[0]

posenew = np.concatenate((tcomb, pcomb, a1),axis=1)
np.savetxt(pnewpath, posenew)

