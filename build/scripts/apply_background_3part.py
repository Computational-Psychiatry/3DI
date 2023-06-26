#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:00:22 2023

@author: v
"""
import itertools
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os

vp1 = sys.argv[1] # '/media/v/SSD1TB/dataset/demo/output/20/BFMmm-19830.cfg9.global4.curt/aretha_cut_3D_sm.avi'
op  = sys.argv[2] # '/media/v/SSD1TB/dataset/demo/output/20/BFMmm-19830.cfg9.global4.curt/aretha_cut_3D_sm.avi'
temp_op = op.replace('.mp4', 'tmp.mp4')

def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[1], top_left[1])
    p3 = (bottom_right[1], bottom_right[0])
    p4 = (top_left[0], bottom_right[0])

    height = abs(bottom_right[0] - top_left[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height/2))

    if thickness < 0:

        #big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
        [top_left_main_rect, bottom_right_main_rect], 
        [top_left_rect_left, bottom_right_rect_left], 
        [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

    return src


top_left = (0, 0)
bottom_right = (1080, 1080)
color = (255, 255, 255)
image_size = (1080, 1080, 3)
img = np.zeros(image_size)
img[:,:,1] = 255
img = rounded_rectangle(img, top_left, bottom_right, color=color, radius=0.1, thickness=-1)
"""
cv2.imshow('rounded_rect', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
px = []
py = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j,0] == 0:
            py.append(i)
            px.append(j)
            


# def convert_bg_to_white(I):
#     for (i,j) in itertools.product(range(I.shape[0]), range(I.shape[1])):
#         val = I[i, j, :]
#         if val[1] >= 190 and val[0] <= 100 and val[2] <=100:
#             I[i, j, :] = [255, 255, 255]
#     return I


# bgim = cv2.imread('/home/v/Downloads/hand-painted-watercolor-pastel-sky-background/5183000.jpg')
bgim = cv2.imread('/home/v/code/3DI/build/scripts/bg_combined3.jpg')
bgim = cv2.resize(bgim, (1920, 1080))

# bgim[:,:,:] = 255   

def convert_bg_to_white(I):
    
    u_green = np.array([111, 255, 111])
    l_green = np.array([0, 160, 0])
      
    # u_green = np.array([104, 153, 70])
    # l_green = np.array([30, 30, 0])
    mask = cv2.inRange(I, l_green, u_green)
    res = cv2.bitwise_and(I, I, mask = mask)

    f = I - res
    f[:,:,0] = np.where(mask == 255, bgim[:,:,0], f[:,:,0])
    f[:,:,1] = np.where(mask == 255, bgim[:,:,1], f[:,:,1])
    f[:,:,2] = np.where(mask == 255, bgim[:,:,2], f[:,:,2])
    return f
    
    # for (i,j) in itertools.product(range(I.shape[0]), range(I.shape[1])):
    #     val = I[i, j, :]
    #     if val[1] >= 190 and val[0] <= 100 and val[2] <=100:
    #         I[i, j, :] = [255, 255, 255]
    # return I

    
    
def paint_edges_green(I, px, py):
    for i in range(len(px)):
        I[py[i], px[i], :] = (0,255,0)
    return I


cap1 = cv2.VideoCapture(vp1)

fps = cap1.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', etc.
out = cv2.VideoWriter(temp_op, fourcc, fps, (1920, 1080))

gif_fp = op.replace('.mp4', '.gif')
gif_frames = []
# out2 = cv2.VideoWriter(op.replace('.mp4', '_480.mp4'), fourcc, fps, (480, 270))

idx = 0
while cap1.isOpened():# and cap2.isOpened():
    ret1, frame1 = cap1.read()
    # ret2, frame2 = cap2.read()
    idx += 1
    
    if not ret1:
        break
    
    face = frame1[:1080, :1080, :]
    rec1 = frame1[:1080, 1080+140:1920+140, :]
    rec1[:80,:,:] = (0,255,0)
    
    rec2= frame1[:1080, 2*1080+140:, :]
    rec2[:80,:,:] = (0,255,0)
    rec1 = cv2.resize(rec1, None, fx=0.55, fy=0.55)
    rec2 = cv2.resize(rec2, None, fx=0.55, fy=0.55)
    # rec1 = convert_bg_to_white(rec1)
    # rec1 = frame1[:1080, 2260:-100, :]
    # rec2 = frame1[:1080, 2260:-100:, :]
    
    face = paint_edges_green(face, px, py)
    comb = np.zeros((1080, 1920, 3), dtype=face.dtype) # np.concatenate((face,rec1), axis=1)
    comb[:,:,:] = (0,255,0)
    
    size = 530
    offs = int((1080-size)/2)
    offx = 144
    face_resized = cv2.resize(face, (size, size))
    comb[offs:size+offs,offx:size+offx,:] = face_resized
    
    offrx1 = 800
    offry1 = 250
    
    offrx2 = 1350
    offry2 = 250
    
    comb[offry1:offry1+rec1.shape[0],offrx1:offrx1+rec1.shape[1],:] = rec1
    comb[offry2:offry2+rec2.shape[0],offrx2:offrx2+rec2.shape[1],:] = rec2
    comb = convert_bg_to_white(comb)
        
    # comb = cv2.resize(comb, None, fx=0.676056, fy=0.676056)
    # canv = np.zeros((1080,1920,3),dtype=np.uint8)
    # oy = int((canv.shape[0]-comb.shape[0])/2)    
    # canv[oy:(oy+comb.shape[0]),:,:] = comb
    out.write(comb)

    # comb = cv2.resize(comb, (480, 270))
    # comb = convert_bg_to_white(comb2)
    # comb = convert_bg_to_white(comb2)
    # gif_frames.append(comb)
    
    # if idx == 80:
    #     break
    # out2.write(comb)
    
cap1.release()
out.release()
# out2.release()
# print(len(gif_frames))
# imageio.mimsave(gif_fp, gif_frames, 'GIF', duration=30)


# Close all windows
        
os.system('ffmpeg -i %s %s 2> /dev/null' % (temp_op, op))
os.system('rm %s' % temp_op)

