/*
 * logbarrier_initializer.h
 *
 *  Created on: Sep 21, 2020
 *      Author: root
 */

#ifndef LOGBARRIER_INITIALIZER_H_
#define LOGBARRIER_INITIALIZER_H_

#include "cuda.h"
#include "constants.h"
#include "funcs.h"
#include "newfuncs.h"
#include "camera.h"
#include "Optimizer.h"
#include "solver.h"
#include "renderer.h"
#include "orthographic_initializer.h"
#include <opencv2/dnn.hpp>



__global__ void multiply_matrix_scalar(float *A, const float alpha, const uint Asize);
__global__ void multiply_vector_scalar(float *v, const float alpha, const uint vsize);

__global__ void update_diagonal_of_hessian_wbounds(const float *vec_lb, const float *vec_ub, float *A, uint Kvec, uint Asize, uint vec_offset);
__global__ void update_diagonal_of_hessian_wvector(const float *vec, float *A, uint Kvec, uint Asize, uint vec_offset);

__global__ void update_diags_and_offdiags_for_expdiffs(float *A, const int T, const int Keps, const int Ktotal, const float *weight);


__global__ void update_bottom_right_of_matrix(const float *A, float *B, const uint Asize, const uint offset);

__global__ void update_gradient(const float *vec_lb, const float *vec_ub, float *gradient, uint Kvec, uint vec_offset);
__global__ void neglogify(float *vec, const uint N);


__global__ void compute_xyproj(const float *varx, const float phix, const float phiy, const float cx, const float cy, const float *Rp, const float *xl, const float *yl, float *xproj, float *yproj);

struct Logbarrier_Initializer
{
	bool use_identity;
	bool use_texture;
	bool use_expression;

    bool use_temp_smoothing;
    bool use_exp_regularization;

    float CONFIDENCE;

	float *Gxs_ALL;

	float *Gx_minus_tmp;
	float *Gy_minus_tmp;
	float *Gx_plus_tmp;
	float *Gy_plus_tmp;

	float *inv_Gx_minus_tmp;
	float *inv_Gy_minus_tmp;
	float *inv_Gx_plus_tmp;
	float *inv_Gy_plus_tmp;

	float *nlog_Gx_minus;
	float *nlog_Gy_minus;
	float *nlog_Gx_plus;
	float *nlog_Gy_plus;

	float *xl;
    float *yl;

    float *bounds_ux;
    float *bounds_uy;
    float *bounds_lx;
    float *bounds_ly;

    float *bounds_lx_cur;
    float *bounds_ly_cur;
    float *bounds_ux_cur;
    float *bounds_uy_cur;

	float *AL; // Dictionary for Alpha (identity basis)
	float *EL; // Dictionary for Epsilon (expression basis)

	float *p;
    float *face_sizes;
	float *Rp;
	float *p0L_mat;

    uint *angle_idx;

	float *beta_lb, *beta_ub;
	float *alpha_lb, *alpha_ub;
	float *epsilon_lb, *epsilon_ub;

	float *f_beta_lb, *f_beta_ub;
	float *f_alpha_lb, *f_alpha_ub;
	float *f_epsilon_lb, *f_epsilon_ub;

    float *eps_l2weights, *eps_l2weights_x2;
    float *deps_l2weights, *deps_l2weights_x2;

    float *drigid_l2weights, *drigid_l2weights_x2;

    float *xmeans, *ymeans;

	bool fit_success;

	float *nablaWx, *nablaWy;
	float *for_nablaPhi_Gx_minus, *for_nablaPhi_Gy_minus, *for_nablaPhi_Gx_plus, *for_nablaPhi_Gy_plus;
	float *for_nabla2F_dsdc;

	float *nabla2F_dsdc;


	float *nabla2F, *nablaF;

    std::vector<std::vector<float> > angle_dictionary;

    std::vector<cv::dnn::Net> vec_bound_estimator;

    std::vector<float *> vec_bounds_ymin;
    std::vector<float *> vec_bounds_ymax;

	float *gradient;
	float *vecOnes;

	bool use_slack;

	float *xmean, *ymean;

	OrthographicInitializer oi;

	RotationComputer rc;
	Solver s;

	SolverQR s_qr;


    std::vector<Camera> *cams_ptr;

	uint T;

	uint Ktotal;
	uint Ktotal_base;

	uint Ktotal_landmarks;
	uint Ktotal_base_landmarks;

    Logbarrier_Initializer(std::vector<Camera> *_cams_ptr, OptimizationVariables* ov, cusolverDnHandle_t& handleDn,  float bdelta,
            bool use_identity_, bool use_texture_, bool use_expression_, Renderer &r, bool _use_slack=false, float _CONFIDENCE=0.3f,
                           bool _use_temp_smoothing=false, bool _use_exp_regularization=false);

	void set_minimal_slack(cublasHandle_t &handle, OptimizationVariables* ov);

	void get_minimal_slack_t(cublasHandle_t &handle, OptimizationVariables* ov, uint t);

	void compute_gradient_and_hessian(cublasHandle_t &handle, OptimizationVariables* ov, float *obj);

	void evaluate_objective_function(cublasHandle_t &handle, OptimizationVariables* ov, float *obj);

    void initialize_with_orthographic_t(cusolverDnHandle_t& handleDn, cublasHandle_t &handle, uint t, const float *h_xl_t, const float *h_yl_t, const float h_face_size, OptimizationVariables* ov, const std::vector<float> &xranges, const std::vector<float> &yranges);

	void copy_from_initialized(Logbarrier_Initializer& li_initialized);

	bool fit_model(cusolverDnHandle_t& handleDn, cublasHandle_t &handle, OptimizationVariables* ov, OptimizationVariables* ov_linesearch);

	void set_bounds_from_host( const float *h_bounds_lx, const float *h_bounds_ly, const float *h_bounds_ux, const float *h_bounds_uy);

	void set_landmarks_from_host(uint t, const float *h_xl_t, const float *h_yl_t);

	void set_dictionaries_from_host(const float *h_AL, const float *h_EL, const OptimizationVariables* ov);

	void compute_nonrigid_shape(cublasHandle_t &handle, const OptimizationVariables* ov, bool identity = true, bool expression = true);

    void compute_bounds(uint t, float yaw, float pitch, float roll, const std::vector<float> &xranges, const std::vector<float> &yranges, bool skip_this_frame=false);


	~Logbarrier_Initializer();
};



#endif /* LOGBARRIER_INITIALIZER_H_ */
