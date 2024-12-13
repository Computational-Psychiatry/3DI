#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  13 15:20:05 2024

@author: b
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

