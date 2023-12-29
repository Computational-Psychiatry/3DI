/*
 * newfuncs.cu
 *
 *  Created on: Aug 8, 2020
 *      Author: root
 */

#include "newfuncs.h"

//extern const int KERNEL_LENGTH;
//extern const int KERNEL_RADIUS;

//extern const int KERNEL_RADIUS; // (2 * KERNEL_RADIUS + 1)
//extern const int KERNEL_LENGTH; // (2 * KERNEL_RADIUS + 1)



__global__ void fill_diffuse_component_and_populate_texture_and_shape(const float *X0, const float *Y0, const float* Z0,
                                                                      const float *X, const float *Y, const float* Z,
                                                                      const uint* indices, const ushort *tl,  const ushort* triangle_idx,
                                                                      const float *alphas, const float *betas, const float *gammas, const float *Ztmp, float *Id_, const float *tex_mu,
                                                                      float *tex_torender, float *dI_dlambda, float *dI_dLintensity,
                                                                      float *vx, float *vy, float *vz, float *px, float *py, float *pz,
                                                                      float *inv_vz, float *inv_vz2, const float *R__, const float* tau__, const float *lambda__,  const float *Lintensity__, const uint N_unique_pixels,
                                                                      const uint Nrender_estimate, bool just_render,
                                                                      int N_TRIANGLES)
{
    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    __shared__ float R[9];
    __shared__ float lambda[3];
    __shared__ float tau[3];
    __shared__ float Lintensity[1];

    if (colix < 9) {
        R[colix] = R__[colix];

        if (colix < 3) {
            lambda[colix] = lambda__[colix];
            tau[colix] = tau__[colix];
        }

        if (colix < 1) {
            Lintensity[0] = Lintensity__[0];
        }
    }

    __syncthreads();

    const float R00 = R[0];
    const float R10 = R[1];
    const float R20 = R[2];
    const float R01 = R[3];
    const float R11 = R[4];
    const float R21 = R[5];
    const float R02 = R[6];
    const float R12 = R[7];
    const float R22 = R[8];


    const uint n = colix + rowix*blockDim.x;

    if (n >= N_unique_pixels)
        return;


    const int rel_index = indices[n];

    const int tl_i1 = triangle_idx[rel_index];
    const int tl_i2 = tl_i1 + N_TRIANGLES;
    const int tl_i3 = tl_i2 + N_TRIANGLES;

    float b0x = X0[tl[tl_i1]];
    float b0y = Y0[tl[tl_i1]];
    float b0z = Z0[tl[tl_i1]];

    float c0x = X0[tl[tl_i2]];
    float c0y = Y0[tl[tl_i2]];
    float c0z = Z0[tl[tl_i2]];

    float a0x = X0[tl[tl_i3]];
    float a0y = Y0[tl[tl_i3]];
    float a0z = Z0[tl[tl_i3]];

    float bx = X[tl[tl_i1]];
    float by = Y[tl[tl_i1]];
    float bz = Z[tl[tl_i1]];

    float cx = X[tl[tl_i2]];
    float cy = Y[tl[tl_i2]];
    float cz = Z[tl[tl_i2]];

    float ax = X[tl[tl_i3]];
    float ay = Y[tl[tl_i3]];
    float az = Z[tl[tl_i3]];


    float n_vx = b0x-a0x;
    float n_vy = b0y-a0y;
    float n_vz = b0z-a0z;

    float n_wx = c0x-a0x;
    float n_wy = c0y-a0y;
    float n_wz = c0z-a0z;


    // below we are computing normal direction of a triangle (nrefx, nrefy, nrefz)
    // this is taken from from https://math.stackexchange.com/questions/305642/how-to-find-surface-normal-of-a-triangle
    // Compute cross product of v and w
    float nrefx = n_vy*n_wz-n_vz*n_wy;
    float nrefy = n_vz*n_wx-n_vx*n_wz;
    float nrefz = n_vx*n_wy-n_vy*n_wx;

    float normn = sqrtf(nrefx*nrefx+nrefy*nrefy+nrefz*nrefz);

    float nref_norm[3] = {nrefx/normn, nrefy/normn, nrefz/normn};

    float shp_x0[3] = {b0x*alphas[rel_index] + c0x*betas[rel_index] + a0x*gammas[rel_index],
                       b0y*alphas[rel_index] + c0y*betas[rel_index] + a0y*gammas[rel_index],
                       b0z*alphas[rel_index] + c0z*betas[rel_index] + a0z*gammas[rel_index]};
    float dlt[3] = {lambda[0]-tau[0], lambda[1]-tau[1], lambda[2]-tau[2]};

    //WARNING!!! This will need to be verified
    //WARNING!!! This will need to be verified
    //WARNING!!! This will need to be verified
    Id_[n] = 0.0f;
    Id_[n] =  nref_norm[0]*(R00*dlt[0] + R10*dlt[1] + R20*dlt[2])
            + nref_norm[1]*(R01*dlt[0] + R11*dlt[1] + R21*dlt[2])
            + nref_norm[2]*(R02*dlt[0] + R12*dlt[1] + R22*dlt[2]);

    ////// - curN_ref*(shpX0(i,:)'
    Id_[n] -= (nref_norm[0]*shp_x0[0] + nref_norm[1]*shp_x0[1] + nref_norm[2]*shp_x0[2]);

    float cur_tex_mu = tex_mu[tl[tl_i1]]*alphas[rel_index] + tex_mu[tl[tl_i2]]*betas[rel_index] + tex_mu[tl[tl_i3]]*gammas[rel_index];

    tex_torender[n] = cur_tex_mu + Lintensity[0]*cur_tex_mu*Id_[n];

    // The steps below are necessary only if we'll complete an optimization step
    if (!just_render)
    {
        float shp_x[3] = {bx*alphas[rel_index] + cx*betas[rel_index] + ax*gammas[rel_index],
                          by*alphas[rel_index] + cy*betas[rel_index] + ay*gammas[rel_index],
                          bz*alphas[rel_index] + cz*betas[rel_index] + az*gammas[rel_index]};

        // Be careful here! This is NOT THE FINAL dI_dlambda component, there is a multiplicatin
        // with the rotation matrix, but that takes place in derivatuve_computer.cu (see
        // 	the line   ====  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Nrender_estimated, 	3, 3, &plus_one_,
        //		o.d_dI_dlambda, Nrender_estimated, rc.R, 3, &zero_, o.dId_dlambda, Nrender_estimated); ===
        dI_dlambda[n] 						= Lintensity[0]*cur_tex_mu*nref_norm[0];
        dI_dlambda[n + Nrender_estimate*1] 	= Lintensity[0]*cur_tex_mu*nref_norm[1];
        dI_dlambda[n + Nrender_estimate*2] 	= Lintensity[0]*cur_tex_mu*nref_norm[2];

        dI_dLintensity[n] =  cur_tex_mu*Id_[n];

        px[n] = shp_x0[0];
        py[n] = shp_x0[1];
        pz[n] = shp_x0[2];

        vx[n] = shp_x[0];
        vy[n] = shp_x[1];
        vz[n] = shp_x[2];

        inv_vz[n] = 1.0f/(vz[n]);
        inv_vz2[n] = inv_vz[n]*inv_vz[n];
    }
}





__global__ void fill_optimization_auxiliary_variables_phase1(const float phix, const float phiy,
                                                             const float *vx__, const float *vy__, const float *vz__, const float *inv_vz, const float *inv_vz2,
                                                             const float *px__, const float *py__, const float *pz__,
                                                             const float *R__,  const float *dR_du1__, const float *dR_du2__, const float *dR_du3__,
                                                             const float *gx, const float *gy,
                                                             float *dI_dtaux, float *dI_dtauy, float *dI_dtauz,
                                                             float *dI_du1, float *dI_du2, float *dI_du3,
                                                             const uint N_unique_pixels)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N_unique_pixels)
        return;

    // This is a very critical step -- those commonly accessed
    // variables need to be made shared (and copied as below)
    // otherwise code runs super slowly
    __shared__ float dR_du1[9];
    __shared__ float dR_du2[9];
    __shared__ float dR_du3[9];

    if (colix < 9)
    {
        dR_du1[colix] = dR_du1__[colix];
        dR_du2[colix] = dR_du2__[colix];
        dR_du3[colix] = dR_du3__[colix];
    }

    __syncthreads();

    // The 6 vars below are accessed multiple times, it may be better to
    // declare them explicitly (so they are allocated at local register and accessed faster)
    float px = px__[n];
    float py = py__[n];
    float pz = pz__[n];

    float vx = vx__[n];
    float vy = vy__[n];
    float vz = vz__[n];

    float dWx_dtaux = phix * inv_vz[n];
    float dWx_dtauy = 0;
    float dWx_dtauz = -(phix) * vx * inv_vz2[n];

    float dWy_dtaux = 0;
    float dWy_dtauy = (phiy) * inv_vz[n];
    float dWy_dtauz = -(phiy) * vy* inv_vz2[n];

    float dvx_du1 = dR_du1[0]*px + dR_du1[3]*py + dR_du1[6]*pz;
    float dvy_du1 = dR_du1[1]*px + dR_du1[4]*py + dR_du1[7]*pz;
    float dvz_du1 = dR_du1[2]*px + dR_du1[5]*py + dR_du1[8]*pz;

    float dvx_du2 = dR_du2[0]*px + dR_du2[3]*py + dR_du2[6]*pz;
    float dvy_du2 = dR_du2[1]*px + dR_du2[4]*py + dR_du2[7]*pz;
    float dvz_du2 = dR_du2[2]*px + dR_du2[5]*py + dR_du2[8]*pz;

    float dvx_du3 = dR_du3[0]*px + dR_du3[3]*py + dR_du3[6]*pz;
    float dvy_du3 = dR_du3[1]*px + dR_du3[4]*py + dR_du3[7]*pz;
    float dvz_du3 = dR_du3[2]*px + dR_du3[5]*py + dR_du3[8]*pz;


    float dWx_du1 = (phix)*inv_vz2[n]*(vz*dvx_du1-vx*dvz_du1); // (*phix)*inv_vz2[rowix]*(vz[rowix]*_dvx_du1-vx[rowix]*_dvz_du1);
    float dWy_du1 = (phiy)*inv_vz2[n]*(vz*dvy_du1-vy*dvz_du1); //(*phiy)*inv_vz2[rowix]*(vz[rowix]*_dvy_du1-vy[rowix]*_dvz_du1);

    float dWx_du2 = (phix)*inv_vz2[n]*(vz*dvx_du2-vx*dvz_du2); //(*phix)*inv_vz2[rowix]*(vz[rowix]*_dvx_du2-vx[rowix]*_dvz_du2);
    float dWy_du2 = (phiy)*inv_vz2[n]*(vz*dvy_du2-vy*dvz_du2); //(*phiy)*inv_vz2[rowix]*(vz[rowix]*_dvy_du2-vy[rowix]*_dvz_du2);

    float dWx_du3 = (phix)*inv_vz2[n]*(vz*dvx_du3-vx*dvz_du3); //(*phix)*inv_vz2[rowix]*(vz[rowix]*_dvx_du3-vx[rowix]*_dvz_du3);
    float dWy_du3 = (phiy)*inv_vz2[n]*(vz*dvy_du3-vy*dvz_du3); //(*phiy)*inv_vz2[rowix]*(vz[rowix]*_dvy_du3-vy[rowix]*_dvz_du3);

    // -(gx.*dWx_dc + gy.*dWy_dc);

    dI_dtaux[n] = -(gx[n]*dWx_dtaux + gy[n]*dWy_dtaux);
    dI_dtauy[n] = -(gx[n]*dWx_dtauy + gy[n]*dWy_dtauy);
    dI_dtauz[n] = -(gx[n]*dWx_dtauz + gy[n]*dWy_dtauz);

    dI_du1[n] = -(gx[n]*dWx_du1 + gy[n]*dWy_du1);
    dI_du2[n] = -(gx[n]*dWx_du2 + gy[n]*dWy_du2);
    dI_du3[n] = -(gx[n]*dWx_du3 + gy[n]*dWy_du3);
}









__global__ void fill_optimization_dI_depsilon_userotated(const float phix, const float phiy,
                                                         const float *  vx__, const float *  vy__, const float *  vz__, const float *  inv_vz2__,
                                                         const float *  R__,
                                                         const float *  gx, const float *  gy,
                                                         const float* __restrict__ REX, const float* __restrict__ REY, const float* __restrict__ REZ,
                                                         float *dI_depsilons,
                                                         const uint N_unique_pixels,
                                                         const uint Kepsilon,
                                                         const uint Nredundant)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N_unique_pixels*Kepsilon)
        return;

    const int i = n % N_unique_pixels;
    const int j = n / N_unique_pixels;


    float vx = vx__[i];
    float vy = vy__[i];
    float vz = vz__[i];

    float inv_vz2 = inv_vz2__[i];

    const uint cix = i+Nredundant*j;

    float nablaWxi_epsilonsj = phix*inv_vz2*(REX[cix]*vz - REZ[cix]*vx);
    float nablaWyi_epsilonsj = phiy*inv_vz2*(REY[cix]*vz - REZ[cix]*vy);

    dI_depsilons[i+Nrender_estimated*j] = -(gx[i]*nablaWxi_epsilonsj + gy[i]*nablaWyi_epsilonsj);
}














__global__ void fill_optimization_dI_dalpha(const float phix, const float phiy,
                                            const float *vx__, const float *vy__, const float *vz__,  const float *inv_vz2__,
                                            const float *R__,
                                            const float *gx, const float *gy,
                                            const float *RIX, const float *RIY, const float *RIZ,
                                            float *dI_dalpha,
                                            const uint N_unique_pixels,
                                            const uint Kalpha)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N_unique_pixels*Kalpha)
        return;

    const int i = n % N_unique_pixels;
    const int j = n / N_unique_pixels;

    __shared__ float R[9];

    if (colix < 9)
    {
        R[colix] = R__[colix];
    }

    __syncthreads();

    const float R00 = R[0];
    const float R10 = R[1];
    const float R20 = R[2];
    const float R01 = R[3];
    const float R11 = R[4];
    const float R21 = R[5];
    const float R02 = R[6];
    const float R12 = R[7];
    const float R22 = R[8];

    // The 3 vars below are accessed multiple times, it may be better to
    // declare them explicitly (so they are allocated at local register and accessed faster)
    float vx = vx__[i];
    float vy = vy__[i];
    float vz = vz__[i];

    float inv_vz2 = inv_vz2__[i];

    float RIXij = RIX[j+Kalpha*i];
    float RIYij = RIY[j+Kalpha*i];
    float RIZij = RIZ[j+Kalpha*i];

    float dvxi_dalphaj = RIXij*R00 + RIYij*R01 + RIZij*R02;
    float dvyi_dalphaj = RIXij*R10 + RIYij*R11 + RIZij*R12;
    float dvzi_dalphaj = RIXij*R20 + RIYij*R21 + RIZij*R22;

    float nablaWxi_alphaj = phix*inv_vz2*(dvxi_dalphaj*vz - dvzi_dalphaj*vx);
    float nablaWyi_alphaj = phiy*inv_vz2*(dvyi_dalphaj*vz - dvzi_dalphaj*vy);

    dI_dalpha[i+Nrender_estimated*j] = -(gx[i]*nablaWxi_alphaj + gy[i]*nablaWyi_alphaj);
}









__global__ void fill_optimization_dI_dbeta(
        const float *RTEX,
        const float *diffuse_comp,
        const float *L_intensity,
        float *dI_dbeta,
        const uint N_unique_pixels,
        const ushort Kbeta)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N_unique_pixels*Kbeta)
        return;

    const int i = n % N_unique_pixels;
    const int j = n / N_unique_pixels;

    dI_dbeta[i+Nrender_estimated*j] = RTEX[j+Kbeta*i] + L_intensity[0]*diffuse_comp[i]*RTEX[j+Kbeta*i];
}













__global__ void fill_optimization_auxiliary_variables_phase2_new(
        const float *dI_dbeta,
        const float *gx, const float *gy, const float *h,
        const ushort *kl_rel, const ushort *kr_rel, const ushort *ka_rel, const ushort *kb_rel,
        float *dgx_dtheta,
        float *dgy_dtheta,
        const int N_unique_pixels, const int Ktotal)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    const int i = n % Nrender_estimated;
    const int j = n / Nrender_estimated;

    if (n >= Nrender_estimated*Ktotal ) {
        return;
    }

    if (i>=N_unique_pixels ) {
        dgx_dtheta[i+Nrender_estimated*j] = 0;
        dgy_dtheta[i+Nrender_estimated*j] = 0;
        return;
    }


    float dIhor_dj = dI_dbeta[kr_rel[i] + Nrender_estimated*j]-dI_dbeta[kl_rel[i] + Nrender_estimated*j];
    float dIver_dj = dI_dbeta[kb_rel[i] + Nrender_estimated*j]-dI_dbeta[ka_rel[i] + Nrender_estimated*j];


    float diffx = 2*gx[i];
    float diffy = 2*gy[i];
    float precoef = 1.0f/(0.001 + h[i]*h[i]*h[i]);

    dgx_dtheta[i+Nrender_estimated*j] = -precoef*(-diffy)*((-diffy)*(-dIhor_dj) - (-diffx)*(-dIver_dj) );
    dgy_dtheta[i+Nrender_estimated*j] =  precoef*(-diffx)*((-diffy)*(-dIhor_dj) - (-diffx)*(-dIver_dj) );

    /*
    if (colix <= 20) {
        printf("KRREL  %d \n", kr_rel[i]);
        printf("KLREL  %d \n", kl_rel[i]);
        printf("KAREL  %d \n", ka_rel[i]);
        printf("KBREL  %d \n", kb_rel[i]);
    }
*/
}





/*

__global__ void fill_optimization_auxiliary_variables_phase2(
        const float *dI_dalpha,
        const float *gx, const float *gy, const float *h,
        const ushort *kl_rel, const ushort *kr_rel, const ushort *ka_rel, const ushort *kb_rel,
        float *dgx_dtheta,
        float *dgy_dtheta,
        const uint N_unique_pixels, const uint Ktotal)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N_unique_pixels*Ktotal ) {
        return;
    }

    const int i = n % N_unique_pixels;
    const int j = n / N_unique_pixels;


    float dIhor_dj = dI_dalpha[kr_rel[i] + Nrender_estimated*j]-dI_dalpha[kl_rel[i] + Nrender_estimated*j];
    float dIver_dj = dI_dalpha[kb_rel[i] + Nrender_estimated*j]-dI_dalpha[ka_rel[i] + Nrender_estimated*j];


    float diffx = 2*gx[i];
    float diffy = 2*gy[i];

    float precoef = 1.0f/(0.001 + h[i]*h[i]*h[i]);

    dgx_dtheta[i+Nrender_estimated*j] = -precoef*(-diffy)*((-diffy)*(-dIhor_dj) - (-diffx)*(-dIver_dj) );
    dgy_dtheta[i+Nrender_estimated*j] =  precoef*(-diffx)*((-diffy)*(-dIhor_dj) - (-diffx)*(-dIver_dj) );
}
*/








#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;



////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH,
        int pitch
        )
{
    // Handle to thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    cooperative_groups::sync(cta);
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float sum = 0;

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

extern "C" void convolutionRowsGPU(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH
        )
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolutionRowsKernel<<<blocks, threads>>>(
                                                 d_Dst,
                                                 d_Src,
                                                 imageW,
                                                 imageH,
                                                 imageW
                                                 );
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH,
        int pitch
        )
{
    // Handle to thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Compute and store results
    cooperative_groups::sync(cta);
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

extern "C" void convolutionColumnsGPU(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH
        )
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads>>>(
                                                    d_Dst,
                                                    d_Src,
                                                    imageW,
                                                    imageH,
                                                    imageW
                                                    );
}



// Calculates rotation matrix given euler angles.
cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<float>(3,3) <<
                   1,       0,              0,
                   0,       cos(theta[0]),   -sin(theta[0]),
            0,       sin(theta[0]),   cos(theta[0])
            );

    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<float>(3,3) <<
                   cos(theta[1]),    0,      sin(theta[1]),
            0,               1,      0,
            -sin(theta[1]),   0,      cos(theta[1])
            );

    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<float>(3,3) <<
                   cos(theta[2]),    -sin(theta[2]),      0,
            sin(theta[2]),    cos(theta[2]),       0,
            0,               0,                  1);


    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;

    return R;
}
