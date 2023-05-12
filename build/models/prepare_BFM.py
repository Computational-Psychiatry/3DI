#!/usr/bin/env python3

import os
import copy
import scipy.io
import numpy as np
from array import array

def prepare_expression_basis():
    a = './raw/Exp_Pca.bin'
    Expbin = open(a,'rb')
    n_vertex = 53215

    target_dir = 'tmp'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    exp_dim = array('i')
    exp_dim.fromfile(Expbin,1)
    expMU0 = array('f')
    expPC0 = array('f')
    expMU0.fromfile(Expbin,3*n_vertex)
    expPC0.fromfile(Expbin,3*exp_dim[0]*n_vertex)

    expPC0 = np.array(expPC0)
    expPC0 = np.reshape(expPC0,[exp_dim[0],-1])
    expPC0 = np.transpose(expPC0)

    expEV = np.loadtxt('./raw/std_exp.txt')

    w = np.loadtxt('./raw/w.dat')
    w = (w-1)
    w = w.astype(int)

    expMU = np.array(expMU0)[w]
    expPC = expPC0[w,:]

    EX = expPC[0::3,:]
    EY = expPC[1::3,:]
    EZ = expPC[2::3,:]

    np.savetxt('%s/EX_79.dat' % target_dir, EX)
    np.savetxt('%s/EY_79.dat' % target_dir, EY)
    np.savetxt('%s/EZ_79.dat' % target_dir, EZ)

    sigma_epsilons = np.sqrt(np.loadtxt('raw/std_exp.txt'))/35

    np.savetxt('%s/sigma_epsilons_79_upperv2.dat' % target_dir, expEV/26000.0)
    np.savetxt('%s/sigma_epsilons_79_lowerv2.dat' % target_dir, -expEV/26000.0)

    oi = [19106,19413,19656,19814,19981,20671,20837,20995,21256,21516,8161,8175,8184,8190,6758,7602,8201,8802,9641,1831,3759,5049,6086,4545,3515,10455,11482,12643,14583,12915,11881,5522,6154,7375,8215,9295,10523,10923,9917,9075,8235,7395,6548,5908,7264,8224,9184,10665,8948,8228,7508]

    EL = None
    for i in oi:
        ex = EX[i,:].reshape(1,-1)
        ey = EY[i,:].reshape(1,-1)
        ez = EZ[i,:].reshape(1,-1)
        if EL is None:
            EL = np.concatenate((ex, ey, ez), axis=0)
        else:
            tmp =  np.concatenate((ex, ey, ez), axis=0)
            EL = np.concatenate((EL, tmp),axis=0)

    #np.savetxt('%s/EL_79.dat' % target_dir, EL)





def prepare_morphable_model():
    print('Adapting BFM models to 3DI...')
    a = scipy.io.loadmat('raw/01_MorphableModel.mat')
    ix_53215  = np.loadtxt('raw/ids_53215.dat').astype(int)
    ix_53215X = [3*int(x) for x in np.loadtxt('raw/ids_53215.dat')]
    ix_53215Y = [3*int(x)+1 for x in np.loadtxt('raw/ids_53215.dat')]
    ix_53215Z = [3*int(x)+2 for x in np.loadtxt('raw/ids_53215.dat')]
    ix_23660_3 = [int(x) for x in np.loadtxt('raw/w.dat')]
    ix_23660 = [int((ix_23660_3[x]-1)/3) for x in range(0,len(ix_23660_3), 3)]

    IX = a['shapePC'][ix_53215X,:][ix_23660,:].astype(np.float64)
    IY = a['shapePC'][ix_53215Y,:][ix_23660,:].astype(np.float64)
    IZ = a['shapePC'][ix_53215Z,:][ix_23660,:].astype(np.float64)

    EX = np.loadtxt('tmp/EX_79.dat')
    EY = np.loadtxt('tmp/EY_79.dat')
    EZ = np.loadtxt('tmp/EZ_79.dat')

    TR = a['texPC'][ix_53215X,:][ix_23660,:].astype(np.float64)
    TG = a['texPC'][ix_53215Y,:][ix_23660,:].astype(np.float64)
    TB = a['texPC'][ix_53215Z,:][ix_23660,:].astype(np.float64)

    mu  = a['shapeMU'].astype(np.float64)
    mu3 = mu.reshape(-1,3)[ix_53215,:][ix_23660,:]
    mu3 -= mu3.mean(axis=0)
    mu3std = mu3.std()
    mu3 /= mu3std
    mu3[:,1] *= -1
    mu3[:,2] *= -1

# landmark indices
    oi = [19106,19413,19656,19814,19981,20671,20837,20995,21256,21516,8161,8175,8184,8190,6758,7602,8201,8802,
          9641,1831,3759,5049,6086,4545,3515,10455,11482,12643,14583,12915,11881,5522,6154,7375,8215,9295,
          10523,10923,9917,9075,8235,7395,6548,5908,7264,8224,9184,10665,8948,8228,7508]

    EL = np.zeros((3*len(oi), EX.shape[1]))
    EL[::3,:] = EX[oi,:]
    EL[1::3,:] = EY[oi,:]
    EL[2::3,:] = EZ[oi,:]


    p0L_mat = mu3[oi,:]

#%%
    pset_orig = set((np.array(ix_53215).astype(int)+1).tolist())
    old2new0 = {ix_53215[x]: x for x in range(len(ix_53215))}
    old2new_tl0 = {ix_53215[x]+1: x+1 for x in range(len(ix_53215))}

    tl_orig = a['tl'].astype(int)
    delix = []
    for i in range(tl_orig.shape[0]):
       keep = True 
       for j in tl_orig[i,:]:
           if j not in pset_orig:
               keep = False
       if not keep:
            delix.append(i)

    delix = set(delix)
    tl_pre = np.delete(copy.deepcopy(tl_orig), list(delix),axis=0)


    for i in range(tl_pre.shape[0]):
        for j in range(tl_pre.shape[1]):
            tl_pre[i,j] = old2new_tl0[tl_pre[i,j]]
            
#%%

    pset_pre = set((np.array(ix_23660).astype(int)+1).tolist())
    old2new0 = {ix_23660[x]: x for x in range(len(ix_23660))}
    old2new_tl0 = {ix_23660[x]+1: x+1 for x in range(len(ix_23660))}

    delix = []
    for i in range(tl_pre.shape[0]):
       keep = True 
       for j in tl_pre[i,:]:
           if j not in pset_pre:
               keep = False
       if not keep:
            delix.append(i)

    delix = set(delix)
    tl = np.delete(copy.deepcopy(tl_pre), list(delix),axis=0)


    for i in range(tl.shape[0]):
        for j in range(tl.shape[1]):
            tl[i,j] = old2new_tl0[tl[i,j]]
            

#%%
    X0 = mu3[:,0]
    Y0 = mu3[:,1]
    Z0 = mu3[:,2]

    TEX = 0.2989 * TR + 0.5870 * TG + 0.1140 *TB

    sigma_alphas = 2*a['shapeEV']/mu3std
    sigma_betas  = 3*0.7*a['texEV']/255

    tex_mu3  = a['texMU'].reshape(-1,3)[ix_53215,:][ix_23660,:]
    tex_mu = 0.2989 *tex_mu3[:,0] + 0.5870 *tex_mu3[:,1] +  0.1140 *tex_mu3[:,2]
    tex_mu /= 255

    ALX = IX[oi,:]
    ALY = IY[oi,:]
    ALZ = IZ[oi,:]

    AL = np.zeros((153, 199))
    AL[::3,:] = ALX[:,:]
    AL[1::3,:] = ALY[:,:]
    AL[2::3,:] = ALZ[:,:]

    AL_60 = np.zeros((153, 60))
    AL_60[::3,:] = ALX[:,:60]
    AL_60[1::3,:] = ALY[:,:60]
    AL_60[2::3,:] = ALZ[:,:60]

    lis = oi

    iod_ptg = 0.35
    G0 = np.concatenate((X0.reshape(-1, 1), Y0.reshape(-1, 1), Z0.reshape(-1, 1)), axis=1)
    if iod_ptg is not None:
        # ref_lis = all_lis[5]
        curli = lis
        ref_lis = [0, 2, 4, 5, 7, 9, 20, 21, 23, 24, 26, 27, 29, 30, 19, 22, 25, 28, 13, 14, 18, 31, 33, 34, 35, 37, 44, 45, 46, 39, 40, 41, 49, 48, 50]

        eoc1 = curli[28]
        eoc2 = curli[19]       
        iod = np.linalg.norm(G0[eoc1,:]-G0[eoc2,:])
        
        adists = []
        for i in range(len(ref_lis)):
            dists = np.sqrt(np.sum(((G0[curli[ref_lis[i]],:]-G0)**2), axis=1))
            adists.append(dists)
        
        mean_dists = np.mean(np.array(adists),axis=0)#/np.median(np.array(adists),axis=0)
        mean_dists -= mean_dists.min()

        adists = []
        for i in range(len(curli)):
            dists = np.sqrt(np.sum(((G0[curli[i],:]-G0)**2), axis=1))
            adists.append(dists)

        min_dists = np.min(np.array(adists),axis=0)#/np.median(np.array(adists),axis=0)
        
        cpts = []
        
        cpts += np.where(mean_dists < iod*(1.25*iod_ptg))[0].tolist() # madists[np.where(madists < 0.2)]
        cpts += np.where(min_dists < iod*0.50*iod_ptg)[0].tolist() # madists[np.where(madists < 0.2)]
        cpts = np.unique(cpts)

    Nnew = len(cpts)

    #tl = np.loadtxt('data/tl.dat').astype(int)

    pset = set((cpts+1).tolist())

    old2new = {cpts[x]: x for x in range(len(cpts))}
    old2new_tl = {cpts[x]+1: x+1 for x in range(len(cpts))}

    delix = []
    for i in range(tl.shape[0]):
       keep = True 
       for j in tl[i,:]:
           if j not in pset:
               keep = False
       if not keep:
            delix.append(i)

    delix = set(delix)
    tl_new = np.delete(copy.deepcopy(tl), list(delix),axis=0)

    for i in range(tl_new.shape[0]):
        for j in range(tl_new.shape[1]):
            tl_new[i,j] = old2new_tl[tl_new[i,j]]
            
    lis_new = []
    for li in lis:
        lis_new.append(old2new[li])


#%%
    ext = 'mm'
    tdir = './MMs/BFM%s-%05d' % (ext, Nnew)
    if not os.path.exists(tdir):
        os.mkdir(tdir)
        
    tdirE = './MMs/BFM%s-%05d/E' % (ext, Nnew)
    if not os.path.exists(tdirE):
        os.mkdir(tdirE)
        
    iod = np.abs(X0[lis[28]]-X0[lis[19]])
    coef = 89/iod
    seu2 = coef*np.loadtxt('tmp/sigma_epsilons_79_upperv2.dat' )
    sel2 = coef*np.loadtxt('tmp/sigma_epsilons_79_lowerv2.dat' )

    X0new = coef*X0[cpts]
    Y0new = coef*Y0[cpts]
    Z0new = coef*Z0[cpts]

    IXnew = IX[cpts,:]
    IYnew = IY[cpts,:]
    IZnew = IZ[cpts,:]

    TEXnew = TEX[cpts,:]
    tex_mu_new = tex_mu[cpts]

    EXnew = EX[cpts,:]
    EYnew = EY[cpts,:]
    EZnew = EZ[cpts,:]

    np.savetxt('%s/IX.dat' % tdir, IXnew)
    np.savetxt('%s/IY.dat' % tdir, IYnew)
    np.savetxt('%s/IZ.dat' % tdir, IZnew)

    np.savetxt('%s/EX_79.dat' % tdirE, EXnew)
    np.savetxt('%s/EY_79.dat' % tdirE, EYnew)
    np.savetxt('%s/EZ_79.dat' % tdirE, EZnew)

    np.savetxt('%s/TEX.dat' % tdir, TEXnew)
    np.savetxt('%s/tex_mu.dat' % tdir, tex_mu_new)
    np.savetxt('%s/tl.dat' % tdir, tl_new, fmt='%d')

    np.savetxt('%s/X0_mean.dat' % tdir, X0new)
    np.savetxt('%s/Y0_mean.dat' % tdir, Y0new)
    np.savetxt('%s/Z0_mean.dat' % tdir, Z0new)

    np.savetxt('%s/AL_full.dat' % tdir, AL)
    np.savetxt('%s/AL_60.dat' % tdir, AL_60)

    p0L = coef*p0L_mat
    sigma_alphas = coef*sigma_alphas

    np.savetxt('%s/p0L_mat.dat' % tdir, p0L)
    np.savetxt('%s/sigma_alphas.dat' % tdir, sigma_alphas)
    np.savetxt('%s/sigma_betas.dat' % tdir, sigma_betas)
    np.savetxt('%s/E/sigma_epsilons_79_upperv2.dat' % tdir, seu2)
    np.savetxt('%s/E/sigma_epsilons_79_lowerv2.dat' % tdir, sel2)
    np.savetxt('%s/E/EL_79.dat' % tdir, EL)

    os.system('rm tmp/EX_79.dat')
    os.system('rm tmp/EY_79.dat')
    os.system('rm tmp/EZ_79.dat')
    os.system('rm tmp/sigma_epsilons_79_upperv2.dat' )
    os.system('rm tmp/sigma_epsilons_79_lowerv2.dat' )
    os.system('rmdir tmp' )



if __name__ == "__main__":
    prepare_expression_basis()
    prepare_morphable_model()


