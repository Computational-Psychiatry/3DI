/*
 * rotation_header.h
 *
 *  Created on: Aug 5, 2020
 *      Author: root
 */

#ifndef ROTATION_HEADER_H_
#define ROTATION_HEADER_H_


#include "cuda.h"
#include "constants.h"
#include "funcs.h"

#include <stdio.h>

#include <cuda_runtime.h>





__global__ void fill_skew_matrix(float *skewmat, const float *u);
__global__ void eye3(float *I);
__global__ void fill_theta_etc(const float *u, float *theta_inv2, float *sin_theta, float *cos_inv_sum, float *u_skew, float *unorm_skew,
		float *R, float *IR, float *R1a, float *R1b, float *R1c, float *R2a, float *R2b, float *R2c, float *R3a, float *R3b, float *R3c,
		float *dR_u1, float *dR_u2, float *dR_u3);

__global__ void fill_theta_etc_without_derivatives(const float *u, float *theta_inv2, float *sin_theta, float *cos_inv_sum,
		float *u_skew, float *unorm_skew, float *R, float *R1a, float *R1b, float *R1c);

__device__ void fill_skew_matrix_dev4(float *sm1, float *sm2, float *sm3, float *sm4, const float *u);
__device__ void fill_skew_matrix_dev2(float *sm1, float *sm2, const float *u);
__device__ void multiply_scalar_matrix3x3(const float* scalar, float *A);
__device__ void sum_AB_3x3_inplace(const float* A, float* B);
__device__ void subtract_AtransB_3x3(const float* A, const float* B, float *C);

__global__ void eye3(float *I);

__device__ void fill_skew_matrix_dev(float *skewmat, const float *u);
__device__ float norm3d_dev(const float *x);

__global__ void print3(float *A);

__device__ void multiplyAB_3x3(const float *A, const float *B, float *C);


__device__ void sum_AB_3x3(const float* A, const float* B,  float *C);
__global__ void fill_skew_matrix(float *skewmat, const float *u);
__device__ void fill_skew_matrix_dev(float *skewmat, const float *u);
__device__ void fill_skew_matrix_dev2(float *sm1, float *sm2, const float *u);
__device__ void fill_skew_matrix_dev4(float *sm1, float *sm2, float *sm3, float *sm4, const float *u);
__device__ void sum_AB_3x3(const float* A, const float* B, float *C);
__device__ void subtract_AtransB_3x3(const float* A, const float* B, float *C);
__device__ void sum_AB_3x3_inplace(const float* A, float* B);
__device__ void multiply_scalar_matrix3x3(const float* scalar, float *A);



struct RotationComputer
{
	float *R, *R1a, *R1b, *R1c, *R2a, *R2b, *R2c, *R3a, *R3b, *R3c;
	float *dR_du1, *dR_du2, *dR_du3;
	float *u_skew, *unorm_skew;
	float *u, *unorm;

	float *theta_inv2, *sin_theta, *cos_inv_sum;

	float *I;
	float *IR;

	float *scalar_one;
	float *scalar_zero;
	float *scalar_minusone;

	float *u_ptr;

	uint size3x3, size3x1, size1x1;

	RotationComputer(float *_u_ptr = NULL);

	void process();

	void compute_euler_angles(float& yaw, float &pitch, float &roll);

	void compute_angle_idx(const float yaw, const float pitch, const float roll, int &yaw_idx, int& pitch_idx, int& roll_idx);

	void process_without_derivatives();

	void set_u_ptr(float *_u_ptr);

	~RotationComputer();
};







#endif /* ROTATION_HEADER_H_ */
