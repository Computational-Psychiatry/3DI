#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:57:52 2023

@author: v
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from common import create_expression_sequence

assert len(sys.argv) >= 4, """At least three arguments are needed to produce canonicalized landmarks: 
    #expression_path #landmark_path #morphable_model_name"""

epath = sys.argv[1] # '/home/v/code/3DI/build/output_cur/BFMmm-19830.cfg11.global4.curt/briannajoy_cut2.expressions_smooth'
lpath = sys.argv[2] # '/home/v/code/3DI/build/output_cur/BFMmm-19830.cfg11.global4.curt/briannajoy_cut2.canonicalized_lmks'

assert epath != lpath, "epath should be different from lpath, otherwise expressions will be overwritten."

morphable_model = sys.argv[3] # BFMmm-19830

sdir = f'models/MMs/{morphable_model}/'

save_anim = False
if len(sys.argv) >= 5:
    save_anim = bool(int(sys.argv[4]))

assert os.path.exists(epath), 'Expression file does not exist!'

li = np.loadtxt(f'{sdir}/li.dat').astype(int)

X0 = np.loadtxt(f'{sdir}/X0_mean.dat').reshape(-1,1)[li]
Y0 = np.loadtxt(f'{sdir}/Y0_mean.dat').reshape(-1,1)[li]
Z0 = np.loadtxt(f'{sdir}/Z0_mean.dat').reshape(-1,1)[li]
shp0 = np.concatenate((X0, Y0, Z0), axis=0)

EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li,:]
EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li,:]
EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li,:]
E = np.concatenate((EX, EY, EZ), axis=0)

e = np.loadtxt(epath)
p = create_expression_sequence(e, E)
EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li,:]
EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li,:]
EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li,:]

Efull = np.concatenate((EX, EY, EZ), axis=0)

T = e.shape[0]

if save_anim:
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    container = []

L = []

for t in range(T):
    et = e[t,:].reshape(-1,1)
    dshp = E @ et
    shp = shp0+dshp
    x0 = shp[:51,0]
    y0 = -shp[51:2*51,0]
    z0 = shp[2*51:3*51,0]
    l = np.concatenate((x0.reshape(-1,1), y0.reshape(-1,1), z0.reshape(-1,1)), axis=1)
    l = l.reshape(-1,1).T
    L.append(l)
    
    if save_anim:
        line, = ax.plot(x0, y0, 'b.')
        
        title = ax.text(0.5,1.05,"Frame {}".format(t), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, )
    
        container.append([line, title])


np.savetxt(lpath, np.concatenate(L, axis=0), fmt='%.4f')

if save_anim:
    ani = animation.ArtistAnimation(fig, container, interval=200, blit=False)
    mp4_path = lpath+'.mp4'
    ani.save(mp4_path, writer=writer)

