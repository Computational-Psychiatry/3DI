#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:57:52 2023

@author: v
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from common import rel_ids, create_expression_sequence

from sklearn.exceptions import ConvergenceWarning

#warnings.filterwarnings("ignore", category=ConvergenceWarning)

exp_coeffs_file = sys.argv[1] 
local_exp_coeffs_file = sys.argv[2]
morphable_model = sys.argv[3] #  'BFMmm-19830'
basis_version = sys.argv[4] # '0.0.1.4'

sdir = f'models/MMs/{morphable_model}/'
localized_basis_file = f'models/MMs/{morphable_model}/E/localized_basis/v.{basis_version}.npy'
basis_set = np.load(localized_basis_file, allow_pickle=True).item()

# @TODO the code does not work for differential expression computation
# but only for absolute expressions. It needs to be adapted to the case where
# basis_set['use_abs'] is set to False!
assert basis_set['use_abs']

li = np.loadtxt(f'{sdir}/li.dat').astype(int)

facial_feats = list(rel_ids.keys())

epsilons = np.loadtxt(exp_coeffs_file)

T = epsilons.shape[0]

Es = {}

for feat in rel_ids:
    rel_id = rel_ids[feat]
    EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li[rel_id],:]
    EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li[rel_id],:]
    EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li[rel_id],:]
    Es[feat] = np.concatenate((EX, EY, EZ), axis=0)
    

ConvergenceWarning('ignore')

C = []
for feat in facial_feats:
    rel_id = rel_ids[feat]
    dp = create_expression_sequence(epsilons, Es[feat])
    coeffs = basis_set[feat].transform(dp).T

    C.append(coeffs)
    
C = np.concatenate(C)

np.savetxt(local_exp_coeffs_file, C)
