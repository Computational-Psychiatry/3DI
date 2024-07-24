#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:57:52 2023

@author: v
"""

import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from common import rel_ids, create_expression_sequence

import cv2

from sklearn.decomposition import DictionaryLearning

expressions_dir = sys.argv[1] # '/media/v/SSD1TB/dataset/videos/treecam/ML/output/BFMmm-19830.cfg7.global4.curt'
morphable_model = sys.argv[2] # 'BFMmm-19830


def save_blank_video(T, videpath):
    # Define video properties
    width = 1080
    height = 1920
    frame_rate = 30  # Frames per second
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(videpath, fourcc, frame_rate, (width, height))
    
    # Create blank frames and write them to the video
    for _ in range(T):  # Create 10 seconds of blank video (adjust as needed)
        blank_frame = 255*np.ones((height, width, 3), dtype=np.uint8)
        out.write(blank_frame)
    
    # Release the VideoWriter
    out.release()
    
    # Display a message when the video is created
    print(f"Blank video saved as {videpath}")


sdir = f'models/MMs/{morphable_model}/'

cfgid = 1
landmark_model = 'global4'
cfg_fpath = './configs/%s.cfg%d.%s.txt' % (morphable_model, cfgid, landmark_model)
camera_param = '30'

T = 60
tmp_local = 'tmp_local'

if not os.path.exists(tmp_local):
    os.mkdir(tmp_local)


save_blank_video(T,  f'{tmp_local}/blank.avi')

illum = np.tile([48.06574, 9.913327, 798.2065, 0.005], (T, 1))
# pose = np.tile([6.9, 43., 799.,-0.16, -0.39, 0.125, -0.38, -0.18, 0.16], (T, 1))
pose = np.tile([3, 43., 799.,0, 0,0.01, -0.38, -0.18, 0.16], (T, 1))

X0 = np.loadtxt(f'{sdir}/X0_mean.dat').reshape(-1,1)
Y0 = np.loadtxt(f'{sdir}/Y0_mean.dat').reshape(-1,1)
Z0 = np.loadtxt(f'{sdir}/Z0_mean.dat').reshape(-1,1)

tex0 = np.loadtxt(f'{sdir}/tex_mu.dat')
li = np.loadtxt(f'{sdir}/li.dat').astype(int)

shp = np.concatenate((X0, Y0, Z0), axis=1)

illum_fpath = f'{tmp_local}/blank.illums'
poses_fpath = f'{tmp_local}/blank.poses'
shp_fpath = f'{tmp_local}/blank.shp'
tex_fpath = f'{tmp_local}/blank.tex'

np.savetxt(illum_fpath, illum)
np.savetxt(poses_fpath, pose)
np.savetxt(shp_fpath, shp)
np.savetxt(tex_fpath, tex0)


files_all = glob(f'{expressions_dir}/*.expressions_smooth')
files = []

for f in files_all:
    if f.find('undistorted') >=0 :
        continue
    files.append(f)

files = files[::2]

algorithm = 'cd'

num_comps = {'lb': 3+2,
             'rb': 3+2,
             'no': 4+1,
             'le': 3+2,
             're': 3+2,
             'ul': 5+1,
             'll': 5+2+1}
"""

num_comps = {'lb': 3+1,
             'rb': 3+1,
             'no': 4,
             'le': 3+1,
             're': 3+1,
             'ul': 5,
             'll': 5+2}
"""


VERSION = '0.0.1.F%d-%s-K%d' % (len(files), algorithm, sum([num_comps[key] for key in num_comps])) # version of the localized_basis

"""
"""
facial_feats = list(num_comps.keys())

use_abs = True
basis_set = {'use_abs': use_abs,
             'num_comps': num_comps}

for feat in facial_feats:
    rel_id = rel_ids[feat]
    K = num_comps[feat]
    
    EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li[rel_id],:]
    EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li[rel_id],:]
    EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li[rel_id],:]
    E = np.concatenate((EX, EY, EZ), axis=0)
    
    xs = []
    dxs = []
    for file in files:
        if file.find('undistorted') >= 0:
            continue
        e = np.loadtxt(file)
        p = create_expression_sequence(e, E)
        xs.append(p)
        dxs.append(np.diff(p, axis=0))
    
    X = np.concatenate(xs, )
    dX = np.concatenate(dxs, )
    
    if not basis_set['use_abs']:
        X = dX
        
    es = []
    es_rec = []
    
    if algorithm == 'cd':
        basis = DictionaryLearning(n_components=K, fit_algorithm='cd',
                                          transform_algorithm='lasso_cd', 
                                          alpha=0.4,transform_alpha=0.4,
                                          random_state=1907,
                                          max_iter=10000,
                                          verbose=False, n_jobs=12)
    elif algorithm == 'default':
        basis = DictionaryLearning(n_components=K, alpha=0.4,transform_alpha=0.4,
                                          #max_iter=5000,
                                          random_state=1907,
                                          verbose=False, n_jobs=1)
        


    print(X.shape)
    X = X[::100,:]
    basis.fit(X)
    W = basis.components_ # This is the basis
    X_transformed = basis.transform(X) #
    
    X_rec = X_transformed @ W
    
    """    
    plt.figure(figsize=(2*10,2*2))
    
    # plt.plot()
    plt.subplot(211)
    plt.plot(X[:500,2])
    plt.plot(X_rec[:500,2],':')
    plt.axis('off')
    
    plt.subplot(212)
    plt.plot(np.diff(X[:500,2]))
    plt.plot(np.diff(X_rec[:500,2]),':')
    plt.axis('off')
    plt.show()
    """
    
    es.append(np.mean(basis.error_))
    
    basis_set[feat] = basis
    basis_set[feat].stats = {'min_0.5pctl':  np.percentile(X_transformed, 0.5, axis=0),
                             'max_99.5pctl':  np.percentile(X_transformed, 99.5, axis=0),
                             'min_2.5pctl':  np.percentile(X_transformed, 2.5, axis=0),
                             'max_97.5pctl':  np.percentile(X_transformed, 97.5, axis=0),
                             'Q1':  np.percentile(X_transformed, 25.0, axis=0),
                             'Q3':  np.percentile(X_transformed, 75.0, axis=0),
                             'median': np.median(X_transformed, axis=0),
                             'mean': np.mean(X_transformed, axis=0),
                             'std': np.std(X_transformed, axis=0)}


bdir = '%s/E/localized_basis' % sdir
if not os.path.exists(bdir):
    os.mkdir(bdir)

isd = ''

if basis_set['use_abs']:
    isd = 'd'

target_fpath = '%s/v.%s%s.npy' % (bdir, VERSION, isd)
np.save(target_fpath, basis_set)

#%%

EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li,:]
EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li,:]
EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li,:]

Efull = np.concatenate((EX, EY, EZ), axis=0)

bix = 0
p0 = np.loadtxt(f'./models/MMs/{morphable_model}/p0L_mat.dat')
for feat in facial_feats:
    rel_id = rel_ids[feat]
    
    for k in range(0, basis_set['num_comps'][feat]):
        zs = np.zeros((T, Efull.shape[1]))
        xmin = basis_set[feat].stats['min_0.5pctl'][k]
        xmax = basis_set[feat].stats['max_99.5pctl'][k]
        
        dx = (xmax-xmin)/T
        
        xmin -= 0.25*dx
        xmax += 0.25*dx
        dx = (xmax-xmin)/T
        
        for t in range(T):
            lvar = np.zeros((1, 51*3))
            x = (xmin+dx*t)*basis_set[feat].components_[k:(k+1),:]
            
            lvar[:,rel_id] = x[:,0:int(x.shape[1]/3)]
            lvar[:,51+rel_id] = x[:,int(x.shape[1]/3):int(x.shape[1]/3)*2]
            lvar[:,51*2+rel_id] = x[:,int(x.shape[1]/3)*2:]
            
            z = np.linalg.lstsq(Efull, lvar.T)[0]
            # z = np.linalg.lstsq(E, x.T)[0]
            zs[t,:] = z.reshape(-1,)
            
        exp_fpath = f'{tmp_local}/blank{bix}.expressions'
        np.savetxt(exp_fpath, zs)
        
        meshvid_fpath = f'{tmp_local}/basis{bix}_mesh.avi'
        texvid_fpath = f'{tmp_local}/basis{bix}.avi'
        texvid_fpath_mp4_tmp = f'{tmp_local}/basis{bix}_tmp.mp4'
        texvid_fpath_mp4 = f'{tmp_local}/basis{bix}.mp4'
        bix += 1
        
        cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % ( f'{tmp_local}/blank.avi', cfg_fpath, camera_param, 
                                                                          shp_fpath, tex_fpath,
                                                                                    exp_fpath, poses_fpath, illum_fpath, 
                                                                                    meshvid_fpath, texvid_fpath)
        
        print(cmd_vis)
        os.system(cmd_vis)
        os.system(f'rm {meshvid_fpath}')
        
        
        btext = '%s-%d' % (feat.upper(), k+1) 
        ffmpeg_cmd_tmp = f'ffmpeg -i {texvid_fpath} -filter:v "crop=436:436:337:850" {texvid_fpath_mp4_tmp}'
        os.system(ffmpeg_cmd_tmp)
        ffmpeg_cmd = f"ffmpeg -y -i {texvid_fpath_mp4_tmp} -vf \"drawtext=text=\'{btext}\':fontcolor=black:fontsize=48:x=10:y=h-th-10,drawtext=text=\'{bix}\':fontcolor=black:fontsize=48:x=10:y=h-th-60\" {texvid_fpath_mp4}"
        os.system(ffmpeg_cmd)
        os.system(f'rm {texvid_fpath}')
        os.system(f'rm {texvid_fpath_mp4_tmp}')
        

ncols = int(np.ceil(np.sqrt((1920./1080.))*np.sqrt(bix)))
nrows = int(np.ceil(float(bix)/ncols))

vix = 0
s1 = ''
s2 = ''
s3 = ''
b = ''
vlist = ''
# width = int(1920.0/ncols)
width = int(3840.0/ncols)

if width % 2 == 1:
    width += 1

for i in range(int(nrows)):
    this_row = 0
    for j in range(int(ncols)):
        this_row += 1
        s1 += f'[{vix}:v]scale={width}:{width}[{vix}v];'
        if vix < bix:
            vlist += f'-i ./tmp_local/basis{vix}.mp4 '
        elif vix >= bix:
            vlist += '-i ./tmp_local/blank.avi '
        vix += 1

vix = 0
for i in range(int(nrows)):
    this_row = 0
    for j in range(int(ncols)):
        this_row += 1
        s2 += f'[{vix}v]'
        vix += 1
    
    s2 += f'hstack={ncols},scale=3840:{width}[v0{i+1}];'
    b += f'[v0{i}]'
    s3 += f'[v0{i+1}]'

s3 += f'vstack={nrows}'

s = s1+s2+s3

ffmpeg_cmd = f'ffmpeg -y {vlist} -filter_complex "{s}" -c:v libx264 -crf 18 output_collage_x.mp4'

print(ffmpeg_cmd)
os.system(ffmpeg_cmd)

ffmpeg_cmd = 'ffmpeg -y -i output_collage_x.mp4 -vf "scale=3840:2160" -c:a copy output_collage.mp4'
print(ffmpeg_cmd)
os.system(ffmpeg_cmd)

b += f'vstack=inputs={nrows}[v]'
    
ffmpeg_cmd = f'ffmpeg -y {vlist} -filter_complex "{s}{b}" -map "[v]" grid.mp4'
print(ffmpeg_cmd)
os.system(ffmpeg_cmd)

# os.system(cmd_vis)

#%%
# illum: 48.06574 9.913327 798.2065 0.005 
# pose 6.985187 43.19978 799.1279 -0.1551871 -0.3975639 0.1251821 -0.3844308 -0.1894762 0.1640978 

# cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % (vid_fpath, cfg_fpath, camera_param, 
#                                                                   shpsm_fpath, tex_fpath,
#                                                                             exps_path, poses_path, illum_path, 
#                                                                             render3ds_path, texturefs_path)

