/*
 * newfuncs.h
 *
 *  Created on: Aug 8, 2020
 *      Author: root
 */

#ifndef NEWFUNCS_H_
#define NEWFUNCS_H_

#include "constants.h"
#include <stdio.h>
#include <opencv2/core.hpp>

//const int KERNEL_RADIUS = 2;
//#define KERNEL_RADIUS 2
//#define KERNEL_RADIUS 3 // <-
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setConvolutionKernel(float *h_Kernel);
extern "C" void convolutionRowsGPU( float *d_Dst, float *d_Src, int imageW, int imageH );
extern "C" void convolutionColumnsGPU( float *d_Dst,  float *d_Src, int  imageW,  int imageH);

__global__ void fill_diffuse_component_and_populate_texture_and_shape(const float *X0, const float *Y0, const float* Z0,
		const float *X, const float *Y, const float* Z,
		const uint* indices, const ushort *tl,  const ushort* triangle_idx,
		const float *alphas, const float *betas, const float *gammas, const float *Ztmp, float *Id_, const float *tex_mu,
		float *tex_torender, float *dI_dlambda, float *dI_dLintensity,
		float *vx, float *vy, float *vz, float *px, float *py, float *pz,
		float *inv_vz, float *inv_vz2, const float *R__, const float* tau__, const float *lambda__,  const float *Lintensity__, const uint N_unique_pixels,
		const uint Nrender_estimate, bool just_render, int N_TRIANGLES);

__global__ void fill_optimization_auxiliary_variables_phase1(const float phix, const float phiy,
		const float *vx__, const float *vy__, const float *vz__, const float *inv_vz, const float *inv_vz2,
		const float *px__, const float *py__, const float *pz__,
		const float *R__,  const float *dR_du1__, const float *dR_du2__, const float *dR_du3__,
		const float *gx, const float *gy,
		float *dI_dtaux, float *dI_dtauy, float *dI_dtauz,
		float *dI_du1, float *dI_du2, float *dI_du3,
		const uint N_unique_pixels);



__global__ void fill_optimization_dI_dalpha(const float phix, const float phiy,
		const float *vx__, const float *vy__, const float *vz__,const float *inv_vz2__,
		const float *R__,
		const float *gx, const float *gy,
		const float *RIX, const float *RIY, const float *RIZ,
		float *dI_dalpha,
		const uint N_unique_pixels,
		const uint Kalpha);


__global__ void fill_optimization_dI_depsilon_userotated(const float phix, const float phiy,
		const float * __restrict__ vx__, const float * __restrict__ vy__, const float * __restrict__ vz__, const float * __restrict__ inv_vz2__,
		const float * __restrict__ R__,
		const float * __restrict__ gx, const float * __restrict__ gy,
		const float* __restrict__ REX, const float* __restrict__ REY, const float* __restrict__ REZ,
		float *dI_depsilons,
		const uint N_unique_pixels,
        const uint Kepsilon,
        const uint Nredundant);


__global__ void fill_optimization_dI_dbeta(const float *RTEX,
                                           const float *diffuse_comp,
                                           const float *L_intensity,
                                           float *dI_dbeta,
                                           const uint N_unique_pixels,
                                           const ushort Kbeta);



__global__ void fill_optimization_auxiliary_variables_phase2_new(
		const float *dI_dbeta,
		const float *gx, const float *gy, const float *h,
		const ushort *kl_rel, const ushort *kr_rel, const ushort *ka_rel, const ushort *kb_rel,
		float *dgx_dtheta,
		float *dgy_dtheta,
		const int N_unique_pixels, const int Ktotal);
/*


__global__ void fill_optimization_auxiliary_variables_phase2(
		const float *dI_dalpha,
		const float *gx, const float *gy, const float *h,
		const ushort *kl_rel, const ushort *kr_rel, const ushort *ka_rel, const ushort *kb_rel,
		float *dgx_dtheta,
		float *dgy_dtheta,
		const uint N_unique_pixels, const uint Ktotal);

*/

cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta);


#endif /* NEWFUNCS_H_ */
