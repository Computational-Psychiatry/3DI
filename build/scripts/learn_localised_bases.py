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

os.chdir('..')

import cv2

from sklearn.decomposition import DictionaryLearning
from scipy.signal import medfilt
sys.path.append(os.getcwd()+'/SyncRef/modules')
from scipy.ndimage import gaussian_filter1d

def create_expression_sequence(epsilons, E):
    ps = []
    for t in range(epsilons.shape[0]):
        epsilon = epsilons[t,:]
        p = ((E @ epsilon)).reshape(-1,1)
        ps.append(p)
    return np.array(ps)[:,:,0]

#%%
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

morphable_model='BFMmm-19830'

sdir = f'models/MMs/{morphable_model}/'


cfgid = 1
landmark_model = 'global4'
cfg_fpath = './configs/%s.cfg%d.%s.txt' % (morphable_model, cfgid, landmark_model)
camera_param = '30'

tmp_local = 'tmp_local'

if not os.path.exists(tmp_local):
    os.mkdir(tmp_local)

T = 60
save_blank_video(T,  f'{tmp_local}/blank.avi')

illum = np.tile([48.06574, 9.913327, 798.2065, 0.005], (T, 1))
# pose = np.tile([6.9, 43., 799.,-0.16, -0.39, 0.125, -0.38, -0.18, 0.16], (T, 1))
pose = np.tile([3, 43., 799.,0, 0,0.01, -0.38, -0.18, 0.16], (T, 1))

X0 = np.loadtxt(f'{sdir}/X0_mean.dat').reshape(-1,1)
Y0 = np.loadtxt(f'{sdir}/Y0_mean.dat').reshape(-1,1)
Z0 = np.loadtxt(f'{sdir}/Z0_mean.dat').reshape(-1,1)

tex0 = np.loadtxt(f'{sdir}/tex_mu.dat')

shp = np.concatenate((X0, Y0, Z0), axis=1)

illum_fpath = f'{tmp_local}/blank.illums'
poses_fpath = f'{tmp_local}/blank.poses'
shp_fpath = f'{tmp_local}/blank.shp'
tex_fpath = f'{tmp_local}/blank.tex'

np.savetxt(illum_fpath, illum)
np.savetxt(poses_fpath, pose)
np.savetxt(shp_fpath, shp)
np.savetxt(tex_fpath, tex0)


#%%

def znorm(x):
    x = np.array(x)
    return (x-x.mean())/x.std()

files = glob('/media/v/SSD1TB/dataset/videos/treecam/ML/output/BFMmm-19830.cfg7.global4.curt/*ns_smooth')

for file in files:
    print(file)


li = [17286,17577,17765,17885,18012,18542,18668,18788,18987,19236,7882,7896,7905,7911,6479,7323,
      7922,8523,9362,1586,3480,4770,5807,4266,3236, 10176,11203,12364,14269,12636,11602,5243,5875,
      7096,7936,9016,10244,10644,9638,8796,7956,7116,6269,5629,6985,7945,8905,10386,8669,7949,7229]

li = np.array(li)

rel_ids   = {'lb': np.array(list(range(0, 5))),
             'rb': np.array(list(range(5, 10))),
             'no': np.array(list(range(10, 19))),
             'le': np.array(list(range(19, 25))),
             're': np.array(list(range(25, 31))),
             'ul': np.array(list(range(31, 37))+list(range(43, 47))),
             'll': np.array(list(range(37, 43))+list(range(47, 51)))}

# num_comps = {'lb': 3,
#              'rb': 3,
#              'no': 4,
#              'le': 3,
#              're': 3,
#              'ul': 5,
#              'll': 5}

num_comps = {'lb': 3+2,
             'rb': 3+2,
             'no': 4+1,
             'le': 3+2,
             're': 3+2,
             'ul': 5+2,
             'll': 5+2}

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
        e = np.loadtxt(file)
        p = create_expression_sequence(e, E)
        xs.append(p)
        dxs.append(np.diff(p, axis=0))
    
    X = np.concatenate(xs, )
    dX = np.concatenate(dxs, )
    
    # X = X-np.mean(X,axis=0)
    # dX = dX-np.mean(dX,axis=0)
    
    if not basis_set['use_abs']:
        X = dX
        
    es = []
    es_rec = []
    
    basis = DictionaryLearning(n_components=K, fit_algorithm='cd',
                                      transform_algorithm='lasso_cd', 
                                      alpha=0.4,transform_alpha=0.4,
                                      random_state=1907,
                                      verbose=False, n_jobs=6)
    
    basis.fit(X[::10,:])
    W = basis.components_ # This is the basis
    X_transformed = basis.transform(X) #
    
    X_rec = X_transformed @ W
        
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
    
    es.append(np.mean(basis.error_))
    # es_rec.append(np.linalg.norm(dX_rec-dX, 'fro'))
    
    basis_set[feat] = basis
    basis_set['%s_min' % feat] = np.percentile(X_transformed, 0.1, axis=0)
    basis_set['%s_max' % feat] = np.percentile(X_transformed, 99.9, axis=0)

#%%

EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li,:]
EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li,:]
EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li,:]

Efull = np.concatenate((EX, EY, EZ), axis=0)
"""
e = np.loadtxt(files[0])
T = e.shape[0]

if not basis_set['use_abs']:
    T -= 1
    

coeffs = []
for feat in facial_feats:
    lvar = np.zeros((T, 51*3))
    
    rel_id = rel_ids[feat]
    K = num_comps[feat]
    
    EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li[rel_id],:]
    EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li[rel_id],:]
    EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li[rel_id],:]
    E = np.concatenate((EX, EY, EZ), axis=0)
    
    # landmarks corresponding to feature
    x = create_expression_sequence(e, E)
    # x = x-np.mean(x, axis=0)
    
    if not basis_set['use_abs']:
        x = np.diff(x, axis=0)
    
    print(x.shape)
    
    lvar[:,rel_id] = x[:,0:int(x.shape[1]/3)]
    lvar[:,51+rel_id] = x[:,int(x.shape[1]/3):int(x.shape[1]/3)*2]
    lvar[:,51*2+rel_id] = x[:,int(x.shape[1]/3)*2:]
    
    coeffs.append(basis_set[feat].transform(x))
    
    z = np.linalg.lstsq(Efull, lvar.T)
    
#%%
"""

bix = 0
p0 = np.loadtxt('/home/v/code/3DI/build/models/MMs/BFMmm-19830/p0L_mat.dat')
for feat in facial_feats:
    rel_id = rel_ids[feat]
    
    # EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li[rel_id],:]
    # EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li[rel_id],:]
    # EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li[rel_id],:]
    # E = np.concatenate((EX, EY, EZ), axis=0)
    
    for k in range(0, basis_set['num_comps'][feat]):
        zs = np.zeros((T, Efull.shape[1]))
        xmin = basis_set['%s_min' % feat][k]
        xmax = basis_set['%s_max' % feat][k]
        
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
        
            # dp = Efull @ z
            # # dp = dp[:,t]
            # dp = dp.reshape(3,-1).T
            # plt.plot(p0[:,0], -p0[:,1], 'o')
            
            # p = p0+dp
            # plt.plot(p[:,0], -p[:,1], 'o')
            # plt.show()
            # plt.xlim((-60,60))
            # plt.ylim((-60,60))
        
        exp_fpath = f'{tmp_local}/blank{bix}.expressions'
        np.savetxt(exp_fpath, zs)
        
        meshvid_fpath = f'{tmp_local}/basis{bix}_mesh.avi'
        texvid_fpath = f'{tmp_local}/basis{bix}.avi'
        texvid_fpath_mp4 = f'{tmp_local}/basis{bix}.mp4'
        bix += 1
        
        cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % ( f'{tmp_local}/blank.avi', cfg_fpath, camera_param, 
                                                                          shp_fpath, tex_fpath,
                                                                                    exp_fpath, poses_fpath, illum_fpath, 
                                                                                    meshvid_fpath, texvid_fpath)
        
        print(cmd_vis)
        os.system(cmd_vis)
        os.system(f'rm {meshvid_fpath}')
        
        ffmpeg_cmd = f'ffmpeg -i {texvid_fpath} -filter:v "crop=436:436:337:850" {texvid_fpath_mp4}'
        os.system(ffmpeg_cmd)
        os.system(f'rm {texvid_fpath}')
        # break
    # break
#%%

ncols = int(np.ceil(np.sqrt((1920./1080.))*np.sqrt(bix)))
nrows = int(np.ceil(float(bix)/ncols))

vix = 0
s1 = ''
s2 = ''
s3 = ''
b = ''
vlist = ''
width = int(1920.0/ncols)

if width % 2 == 1:
    width += 1

for i in range(int(nrows)):
    this_row = 0
    for j in range(int(ncols)):
        this_row += 1
        s1 += f'[{vix}:v]scale={width}:{width}[{vix}v];'
        vix += 1
        if vix < bix:
            vlist += f'-i ./tmp_local/basis{vix}.mp4 '
        elif vix >= bix:
            vlist += '-i ./tmp_local/blank.avi '

vix = 0
for i in range(int(nrows)):
    this_row = 0
    for j in range(int(ncols)):
        this_row += 1
        s2 += f'[{vix}v]'
        vix += 1
    
    s2 += f'hstack={ncols},scale=1920:{width}[v0{i+1}];'
    b += f'[v0{i}]'
    s3 += f'[v0{i+1}]'

s3 += f'vstack={nrows}'
    
s = s1+s2+s3

ffmpeg_cmd = f'ffmpeg -y {vlist} -filter_complex "{s}" -c:v libx264 -crf 18 output_collage_x.mp4'

print(ffmpeg_cmd)
os.system(ffmpeg_cmd)

ffmpeg_cmd = 'ffmpeg -y -i output_collage_x.mp4 -vf "scale=1920:1080" -c:a copy output_collage.mp4'
print(ffmpeg_cmd)
os.system(ffmpeg_cmd)

#%%
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

