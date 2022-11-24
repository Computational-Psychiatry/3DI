#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:31:13 2022

@author: v
"""

import numpy as np
import matplotlib.pyplot as plt

L = np.loadtxt('./exampledata/vid01.landmarks_dexp')

X0 = np.loadtxt('Xl_mean.txt')
Y0 = np.loadtxt('Yl_mean.txt')
Z0 = np.loadtxt('Zl_mean.txt')

T = L.shape[0]
for t in range(T):
    plt.plot(X0+L[t,::3], Y0+L[t,1::3])
    plt.xlim((-1.6, 1.6))
    plt.ylim((-1.6, 1.6))
