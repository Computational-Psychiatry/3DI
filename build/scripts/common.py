#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:20:37 2024

@author: sariyanide
"""

import numpy as np

def create_expression_sequence(epsilons, E):
    ps = []
    for t in range(epsilons.shape[0]):
        epsilon = epsilons[t,:]
        p = ((E @ epsilon)).reshape(-1,1)
        ps.append(p)
    return np.array(ps)[:,:,0]


rel_ids = {'lb': np.array(list(range(0, 5))),
             'rb': np.array(list(range(5, 10))),
             'no': np.array(list(range(10, 19))),
             'le': np.array(list(range(19, 25))),
             're': np.array(list(range(25, 31))),
             'ul': np.array(list(range(31, 37))+list(range(43, 47))),
             'll': np.array(list(range(37, 43))+list(range(47, 51)))}
