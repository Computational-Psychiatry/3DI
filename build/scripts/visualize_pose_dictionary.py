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

import cv2

from sklearn.decomposition import DictionaryLearning


dict_path = sys.argv[1]

morphable_model = 'BFMmm-19830'
# expressions_dir =  '/media/v/SSD1TB/dataset/videos/treecam/ML/output/BFMmm-19830.cfg7.global4.curt'
# morphable_model = 'BFMmm-19830'

def create_expression_sequence(epsilons, E):
    ps = []
    for t in range(epsilons.shape[0]):
        epsilon = epsilons[t,:]
        p = ((E @ epsilon)).reshape(-1,1)
        ps.append(p)
    return np.array(ps)[:,:,0]

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
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
cfgid = 1
landmark_model = 'global4'
cfg_fpath = './configs/%s.cfg%d.%s.txt' % (morphable_model, cfgid, landmark_model)
camera_param = '30'

#tmp_local = 'tmp_local'
tmp_dir = 'tmp_pose'

if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

T = 120
save_blank_video(T,  f'{tmp_dir}/blank.avi')
illum = np.tile([48.06574, 9.913327, 798.2065, 0.005], (T, 1))
# pose = np.tile([6.9, 43., 799.,-0.16, -0.39, 0.125, -0.38, -0.18, 0.16], (T, 1))

exp = np.tile([-1.124897254600429797e+01,2.174177344873850615e+01,1.593945107606392853e+01,-1.319303954257389222e+00,1.822380493293616510e+01,4.345994412298833431e+00,9.398729539720104498e+00,-6.052618028738690370e+00,2.636741298405689271e-01,1.383993480233857376e+00,-1.058190916967946649e+00,3.986020933802426391e+00,-6.162488163161774501e+00,1.053224638353888576e+00,1.002550413563580456e+01,3.756578936489190390e+00,4.368704460012535762e+00,4.874665801066332627e+00,5.571453172495387740e+00,1.650939120231957880e+00,5.250078899037627922e-01,-1.894019702661432270e+00,2.827740291884905055e+00,4.894532295172115965e-03,8.233804427615325494e+00,2.514335919021796428e+00,-2.191154236495034269e-01,-2.730388468235593891e-01,-3.278278894185109316e+00,1.245909113202484209e+00,-5.771549093584050993e+00,-6.113828353097718882e+00,3.607199371758971207e+00,7.237250245724826669e-01,-4.376338471150852172e+00,-3.864534454523624873e+00,-7.683411415739715089e+00,8.250401553682372935e+00,-2.504809372951443791e+00,-2.412716889694685563e+00,5.302659260183126833e-01,1.138044998734565105e+00,-5.361496170814329609e+00,-5.165778722075312857e+00,-1.398770028305514224e+00,-7.775475260792426013e+00,-8.946153812416490325e+00,-8.368003519387768208e+00,4.709194140654929583e+00,1.139876590709995696e+00,1.970084616687126200e+00,5.141611965567787657e+00,-1.861674840073144210e-01,-5.248394813371268341e+00,4.202905808125509068e+00,4.814372995849146797e+00,-4.394234056007189082e-01,2.104571802250880808e-01,3.693677482735192807e+00,1.463689787599377290e+00,1.365529611255569931e+00,-5.799341445587493649e-01,2.411855249244532651e+00,3.441332831721835461e+00,1.519854847326495673e+00,4.808407436546727531e+00,6.905782429460756155e-01,-4.297449288556633995e-01,4.551818928861485425e+00,-8.580911390803658279e-01,-6.316818153323950913e-01,-2.434255208459015218e-01,1.664908814640547519e+00,2.492066070704616276e+00,3.578949339560217879e-01,2.917854628088227376e+00,2.924044671329598444e+00,1.332054973893395111e-01,2.583388267529891813e+00],
              (T, 1))

X0 = np.loadtxt(f'{sdir}/X0_mean.dat').reshape(-1,1)
Y0 = np.loadtxt(f'{sdir}/Y0_mean.dat').reshape(-1,1)
Z0 = np.loadtxt(f'{sdir}/Z0_mean.dat').reshape(-1,1)

tex0 = np.loadtxt(f'{sdir}/tex_mu.dat')

shp = np.concatenate((X0, Y0, Z0), axis=1)

illum_fpath = f'{tmp_dir}/blank.illums'
poses_fpath = f'{tmp_dir}/blank.poses'
shp_fpath = f'{tmp_dir}/blank.shp'
tex_fpath = f'{tmp_dir}/blank.tex'

np.savetxt(illum_fpath, illum)
np.savetxt(shp_fpath, shp)
np.savetxt(tex_fpath, tex0)
#Z = np.loadtxt(f'{os.path.expanduser("~")}/code/3DI/build/{tmp_dir}/Z.txt')
Z = np.loadtxt(dict_path)#(f'{os.path.expanduser("~")}/code/3DI/build/{tmp_dir}/Z.txt')

for bix in range(Z.shape[0]):
    
    pose = np.tile([3, 43., 799.,0, 0,0.01, -0.38, -0.18, 0.16], (T, 1))
    pose_bix = Z[bix,:].reshape(-1,3)
    pose[:,3:6] = pose_bix
    # pose[:,6:] = np.random.randn(T, 3)
    
    np.savetxt(poses_fpath, pose)
    
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    
    exp_fpath = f'{tmp_dir}/blank{bix}.expressions'
    np.savetxt(exp_fpath, exp)
    
    meshvid_fpath = f'{tmp_dir}/basis{bix}_mesh.avi'
    texvid_fpath = f'{tmp_dir}/basis{bix}.avi'
    texvid_fpath_mp4_tmp = f'{tmp_dir}/basis{bix}_tmp.mp4'
    texvid_fpath_mp4 = f'{tmp_dir}/basis{bix}.mp4'
    
    cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % ( f'{tmp_dir}/blank.avi', cfg_fpath, camera_param, 
                                                                      shp_fpath, tex_fpath,
                                                                                exp_fpath, poses_fpath, illum_fpath, 
                                                                                meshvid_fpath, texvid_fpath)
    
    print(cmd_vis)
    os.system(cmd_vis)
    #os.system(f'rm {meshvid_fpath}')
#%%%

# btext = '%s-%d' % (feat.upper(), k+1) 
# ffmpeg_cmd_tmp = f'ffmpeg -i {texvid_fpath} -filter:v "crop=436:436:337:850" {texvid_fpath_mp4_tmp}'
# os.system(ffmpeg_cmd_tmp)
# ffmpeg_cmd = f"ffmpeg -y -i {texvid_fpath_mp4_tmp} -vf \"drawtext=text=\'{btext}\':fontcolor=black:fontsize=48:x=10:y=h-th-10,drawtext=text=\'{bix}\':fontcolor=black:fontsize=48:x=10:y=h-th-60\" {texvid_fpath_mp4}"
# os.system(ffmpeg_cmd)
# os.system(f'rm {texvid_fpath}')
# os.system(f'rm {texvid_fpath_mp4_tmp}')


# ncols = int(np.ceil(np.sqrt((1920./1080.))*np.sqrt(bix)))
# nrows = int(np.ceil(float(bix)/ncols))

# vix = 0
# s1 = ''
# s2 = ''
# s3 = ''
# b = ''
# vlist = ''
# # width = int(1920.0/ncols)
# width = int(3840.0/ncols)

# if width % 2 == 1:
#     width += 1

# for i in range(int(nrows)):
#     this_row = 0
#     for j in range(int(ncols)):
#         this_row += 1
#         s1 += f'[{vix}:v]scale={width}:{width}[{vix}v];'
#         if vix < bix:
#             vlist += f'-i ./tmp_local/basis{vix}.mp4 '
#         elif vix >= bix:
#             vlist += '-i ./tmp_local/blank.avi '
#         vix += 1

# vix = 0
# for i in range(int(nrows)):
#     this_row = 0
#     for j in range(int(ncols)):
#         this_row += 1
#         s2 += f'[{vix}v]'
#         vix += 1
    
#     s2 += f'hstack={ncols},scale=3840:{width}[v0{i+1}];'
#     b += f'[v0{i}]'
#     s3 += f'[v0{i+1}]'

# s3 += f'vstack={nrows}'

# s = s1+s2+s3

# ffmpeg_cmd = f'ffmpeg -y {vlist} -filter_complex "{s}" -c:v libx264 -crf 18 output_collage_x.mp4'

# print(ffmpeg_cmd)
# os.system(ffmpeg_cmd)

# ffmpeg_cmd = 'ffmpeg -y -i output_collage_x.mp4 -vf "scale=3840:2160" -c:a copy output_collage.mp4'
# print(ffmpeg_cmd)
# os.system(ffmpeg_cmd)

# b += f'vstack=inputs={nrows}[v]'
    
# ffmpeg_cmd = f'ffmpeg -y {vlist} -filter_complex "{s}{b}" -map "[v]" grid.mp4'
# print(ffmpeg_cmd)
# os.system(ffmpeg_cmd)

# os.system(cmd_vis)

#%%
# illum: 48.06574 9.913327 798.2065 0.005 
# pose 6.985187 43.19978 799.1279 -0.1551871 -0.3975639 0.1251821 -0.3844308 -0.1894762 0.1640978 

# cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % (vid_fpath, cfg_fpath, camera_param, 
#                                                                   shpsm_fpath, tex_fpath,
#                                                                             exps_path, poses_path, illum_path, 
#                                                                             render3ds_path, texturefs_path)

