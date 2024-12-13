#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  13 15:20:05 2024

@author: b
"""
import numpy as np
import sys

alpha = sys.argv[1]
beta = sys.argv[2]
shp_path = sys.argv[3]
tex_path = sys.argv[4]
morphable_model = sys.argv[5]

sdir = './models/MMs/%s' % morphable_model 

IX  = np.loadtxt('%s/IX.dat' % sdir)
IY  = np.loadtxt('%s/IY.dat' % sdir)
IZ  = np.loadtxt('%s/IZ.dat' % sdir)

TEX  = np.loadtxt('%s/TEX.dat' % sdir)

tex_mu = np.loadtxt('%s/tex_mu.dat' % sdir)

x0 = np.loadtxt('%s/X0_mean.dat' % sdir)
y0 = np.loadtxt('%s/Y0_mean.dat' % sdir)
z0 = np.loadtxt('%s/Z0_mean.dat' % sdir)

x = (x0+(IX @ alpha)).reshape(-1,1)
y = (y0+(IY @ alpha)).reshape(-1,1)
z = (z0+(IZ @ alpha)).reshape(-1,1)

tex = (tex_mu+(TEX @ beta)).reshape(-1, 1)

np.savetxt(shp_path, np.concatenate((x,y,z), axis=1))
np.savetxt(tex_path, tex)

