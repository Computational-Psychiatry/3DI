#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:35:20 2024

@author: v
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
from sys import exit
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--ims_path', type=str, default='./ims/', 
                    help="""Directory that contains images (PNG or JPG) to be used for reconstrution. 
                    All images must be of the same size and (ideally) be recorded with the same camera.""")
parser.add_argument('--subj_key', type=str, default=None)
parser.add_argument('--out_rootpath', type=str, default='./output')
parser.add_argument('--cfgid', type=int, default=1)
parser.add_argument('--fov', type=int, default=20)
parser.add_argument('--which_bfm', type=str, default='BFMmm-23660')
parser.add_argument('--delete_tmp_files', type=int, default=1,
                    help="""Delete temporary files that are used to predict the face shape.
                        (One may want to keep them for inspection.)""")

args = parser.parse_args()
args.delete_tmp_files = bool(args.delete_tmp_files)

ims = glob(f'{args.ims_path}/*png')+glob(f'{args.ims_path}/*jpg')

if len(ims) == 0:
    print(f"The directory {args.ims_path} contains no images", file=sys.stderr)
    exit(1)

if args.subj_key is None:
    args.subj_key = os.path.basename(os.path.dirname(args.ims_path))

Ntot_frames = len(ims)

if Ntot_frames < 2:
    print(f"{args.ims_path} needs to contain at least 2 images.", file=sys.stderr)
    exit(1)

if not os.path.exists(args.out_rootpath):
    os.mkdir(args.out_rootpath)
    
tmp_out_dir = f'{args.out_rootpath}/3DI-cfg{args.cfgid}-cam{args.fov}-{args.which_bfm}-tmp-{args.subj_key}'
if not os.path.exists(tmp_out_dir):
    os.mkdir(tmp_out_dir)
    
out_dir = f'{args.out_rootpath}/3DI-cfg{args.cfgid}-cam{args.fov}-{args.which_bfm}'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_fpath = f'{out_dir}/{args.subj_key}.txt'

if os.path.exists(out_fpath):
    print('Reconstruction for {args.subj_key} is done; see {out_fpath}')
    exit(0)

if 2 == Ntot_frames:
    Nframes_rec = 2
    Ntot_recs = 1
if 3 <= Ntot_frames <= 4:
    Nframes_rec = 2
    Ntot_recs = 2
elif 5 <= Ntot_frames <= 7:
    Nframes_rec = 3
    Ntot_recs = 7
elif 8 <= Ntot_frames <= 15:
    Nframes_rec = 5
    Ntot_recs = 12
elif 16 <= Ntot_frames:
    Nframes_rec = 7
    Ntot_recs = 15

frame_combs_str = set()
frame_combs = list()

for i in range(1000):
    comb = random.sample(range(Ntot_frames), Nframes_rec)
    comb.sort()
    comb_str = '+'.join([str(x) for x in comb])

    if comb_str in frame_combs_str:
        continue

    frame_combs_str.add(comb_str)    
    frame_combs.append(comb)

    if len(frame_combs_str) == Ntot_recs:
        break

im_combs = []
for comb in frame_combs:
    im_combs.append(','.join([ims[x] for x in comb]))

content = ';'.join(im_combs)
imlist_file = f'{tmp_out_dir}/{args.subj_key}.imlist'

with open(imlist_file, 'w') as f:
    f.write(content)

cfg_filepath = f'./configs/{args.which_bfm}.cfg{args.cfgid}.global4.txt'

cmd = f'./fit_to_multiframe {imlist_file} {cfg_filepath} {args.fov} {tmp_out_dir}'
print(cmd)
os.system(cmd)

out_files = glob(f'{tmp_out_dir}/*id.txt') 

outs = []
for of in out_files:
    outs.append(np.loadtxt(of))

P = np.mean(outs, axis=0)
np.savetxt(out_fpath, P)

if args.delete_tmp_files:
    shutil.rmtree(tmp_out_dir)


# The full BFM model has 53490 points
est_full = np.zeros((53490, 3))

ix_3di = np.loadtxt('misc/ix_3di_23660.txt').astype(int)
ix_common = np.loadtxt('misc/ix_common.txt').astype(int)

np.savetxt('misc/ix_3di_23660.txt', ix_3di)
est_full[ix_3di,:]  = P
est = est_full[ix_common, :]

"""
import matplotlib.pyplot as plt
plt.figure(figsize=(40, 20))
plt.subplot(121)
plt.plot(est[:,0], -est[:,1], '.')
plt.subplot(122)
plt.plot(est[:,2], -est[:,1], '.')
"""


