#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:00:22 2023

@author: v
"""
import numpy as np
import copy
import cv2
import sys

vps = []
t0s = []
tfs = []

for i in range(1,len(sys.argv)-2, 3):
    vps.append(sys.argv[i])
    
for i in range(2,len(sys.argv)-1, 3):
    t0s.append(float(sys.argv[i]))
    
for i in range(3,len(sys.argv)-1, 3):
    tfs.append(float(sys.argv[i]))

Nvids = len(vps)

speed = int(float(sys.argv[-2]))
op = sys.argv[-1]

print(vps)
print(t0s)
print(tfs)
print(Nvids)
print(speed)


def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):
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


if Nvids == 1:
    ncols = 1
    nrows = 1
    fx = fy = 1.4
elif Nvids == 4:
    ncols = 2
    nrows = 2
    fx = fy = 0.9
elif Nvids == 2:
    ncols = 2
    nrows = 1
    fx = fy = 0.9
elif Nvids == 6:
    ncols = 3
    nrows = 2
    fx = fy = 0.6
elif Nvids == 9:
    ncols = 3
    nrows = 3
    fx = fy = 0.6


color = (255, 255, 255)
image_size = (int(round(1080*fy)), int(round(1920*fx)), 3)
print(image_size)
top_left = (0, 0)
bottom_right = (image_size[0], image_size[1])
img = np.zeros(image_size)
img[:,:,1] = 255
img = rounded_rectangle(img, top_left, bottom_right, color=color, radius=0.1, thickness=-1)
mask0 = cv2.inRange(img, np.array([0,254,0]), np.array([0, 255, 0]))
res0 = cv2.bitwise_and(img, img, mask = mask0)

img = img.astype(np.uint8)
res0 = res0.astype(np.uint8)

f0 = img-res0

px = []
py = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j,0] == 0:
            py.append(i)
            px.append(j)

bgim = cv2.imread('/home/v/code/3DI/build/scripts/bg2_combined2.jpg')
bgim = cv2.imread('/home/v/code/3DI/build/scripts/tech6.jpg')
bgim = cv2.resize(bgim, (3840, 2160))

def convert_bg_to_white(I, pI):
    u_green = np.array([0, 255, 0])
    l_green = np.array([0, 254, 0])
    mask = cv2.inRange(pI, l_green, u_green)
    res = cv2.bitwise_and(I, I, mask = mask)

    f = I - res
    f[:,:,0] = np.where(mask == 255, bgim[:,:,0], f[:,:,0])
    f[:,:,1] = np.where(mask == 255, bgim[:,:,1], f[:,:,1])
    f[:,:,2] = np.where(mask == 255, bgim[:,:,2], f[:,:,2])
    return f

def paint_edges_green(I):
    f = I-res0
    f[:,:,0] = np.where(mask0 == 255, img[:,:,0], I[:,:,0])
    f[:,:,1] = np.where(mask0 == 255, img[:,:,1], I[:,:,1])
    f[:,:,2] = np.where(mask0 == 255, img[:,:,2], I[:,:,2])
    return f


caps = [cv2.VideoCapture(vp) for vp in vps]
final_frames = []
Nframes_per_vid = []



# Here we skip the first n frames
for ci in range(len(caps)):
    fps = caps[ci].get(cv2.CAP_PROP_FPS)
    final_frames.append(fps*tfs[ci])
    Nframes_per_vid.append(int(fps*(tfs[ci]-t0s[ci])))
    
    num_offset_frames = t0s[ci]*fps
    
    for ti in range(int(num_offset_frames)):
        ret, frame = caps[ci].read()


Nframes = min([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])
Nframes_to_process = min(Nframes_per_vid)

fps = caps[0].get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V') # or 'XVID', 'MJPG', etc.
out = cv2.VideoWriter(op, fourcc, fps, (1920, 1080))

gapx = 36
gapy = 36

pw = round(fx*1920)
ph = round(fy*1080)

totw = ncols*pw+(ncols-1)*gapx
toth = nrows*ph+(nrows-1)*gapy

ox = int((3840-totw)/2)
oy = int((2160-toth)/2)
terminate = False
prevoframe = None
prevoframe2 = None
for idx in range(Nframes_to_process):
    if idx % 100 == 0:
        print(idx)   
    
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            terminate = True
            break
        frames.append(paint_edges_green(cv2.resize(frame, None, fx=fx, fy=fy)))
    
    if terminate:
        break
    oframe = copy.deepcopy(bgim)
    
    ox1 = int((1-fx)*(bgim.shape[1])/2)-int(gapx/2)
    oy1 = int((1-fy)*(bgim.shape[0])/2)-int(gapy/2)
    ox2 = gapx + ox1 + pw
    oy2 = gapy + oy1 + ph
    
    idx = 0
    for i in range(nrows):
        y0 = oy + (gapy+ph)*i
        yf = y0+ph
        
        for j in range(ncols):
            x0 = ox + (gapx+pw)*j
            xf = x0+pw
            
            oframe[y0:yf,x0:xf] = frames[idx]
            idx += 1

    oframe = convert_bg_to_white(oframe,oframe)
    
    oframe = cv2.resize(oframe, (1920, 1080))
    if speed == 100:
        out.write(oframe)
    elif speed == 50:
        oframed = oframe.astype(np.float32)
        if prevoframe is None and prevoframe2 is None: 
            out.write(oframe)
            out.write(oframe)
        elif prevoframe2 is None:
            out.write(oframe)
            out.write(oframe)
            # prevoframed = prevoframe.astype(np.float32)
            # oframed = cv2.cvtColor(oframe,cv2.CV_32FC3)
            # prevoframed = cv2.cvtColor(prevoframe,cv2.CV_32FC3)
            # out.write(((oframed+prevoframed)/2.).astype(np.uint8))
        else:
            out.write(oframe)
            out.write(oframe)
            # prevoframed = prevoframe.astype(np.float32)
            # prevoframe2d = prevoframe2.astype(np.float32)
            # out.write(((prevoframe2d+prevoframed)/2.).astype(np.uint8))
            # out.write(((oframed+prevoframed)/2.).astype(np.uint8))
            
            
    elif speed == 33:
        out.write(oframe)
        out.write(oframe)
        out.write(oframe)
    elif speed == 25:
        out.write(oframe)
        out.write(oframe)
        out.write(oframe)
        out.write(oframe)
        
    prevoframe2 = prevoframe
    prevoframe = oframe
    



for cap in caps:
    cap.release()

out.release()
