/*
 * orthographic_initializer.h
 *
 *  Created on: Sep 17, 2020
 *      Author: root
 */

#ifndef ORTHOGRAPHIC_INITIALIZER_H_
#define ORTHOGRAPHIC_INITIALIZER_H_

#include "cuda.h"
#include "constants.h"

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "rotation_header.h"
#include "solver.h"
#include "funcs.h"

#include "input_data.h"

#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>


__global__ void compute_orth_projections(const float *x_orthographic, const float *xl_bar, const float *yl_bar, const float *p0L_mat,
        const float *Rp, float *xproj, float *yproj);

__global__ void compute_gradient_hessian_obj_orthographic(const float *x_orthographic, const float *xl_bar, const float *yl_bar, const float *p0L_mat,
		const float *Rp, const float *dRp_du1, const float *dRp_du2, const float *dRp_du3, const bool eval_gradients, float *nablaW, float *err);


struct OrthographicInitializer
{
	float *d_x_orthographic;
	float *d_x_orthographic_linesearch;
	float *d_xl_bar, *d_yl_bar;
	float *d_p0L_mat;
	float *d_err;
	float *d_dg;
	float *d_obj;
	float *d_obj_tmp;
	float *d_JTJ;
	float *d_tmp;
	float *d_Rp, *d_dRp_du1, *d_dRp_du2, *d_dRp_du3, *d_nablaW;

    float yaw, pitch, roll;

	RotationComputer rc;
	RotationComputer rc_linesearch;

	Solver s;

	const uint MAXITER_OUTER = 100;
	const uint MAXITER_INNER = 100;

	OrthographicInitializer(cusolverDnHandle_t& handleDn);
	void set_landmarks(const float* xl, const float* yl);

	void reset_orthographic();

    void fit_model(cusolverDnHandle_t& handleDn, cublasHandle_t& handle,
                   const float* xl, const float* yl,
                   float *yaw_ptr = NULL, float *pitch_ptr = NULL, float *roll_ptr = NULL, bool reset_variables=true);

	~OrthographicInitializer();

};



struct LightPoseEstimator
{
    LightPoseEstimator() {}

    vector<vector<float> > estimate_poses(cusolverDnHandle_t& handleDn, cublasHandle_t& handle,
                        OrthographicInitializer& oi, LandmarkData& ld);

};


























#endif /* ORTHOGRAPHIC_INITIALIZER_H_ */
