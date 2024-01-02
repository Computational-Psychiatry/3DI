/*
 * renderer.h
 *
 *  Created on: Aug 10, 2020
 *      Author: root
 */

#ifndef RENDERER_H_
#define RENDERER_H_

#include "constants.h"
#include "funcs.h"
#include "newfuncs.h"
#include "Optimizer.h"
#include "rotation_header.h"
#include "solver.h"
#include "camera.h"
#include "config.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>


#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



struct Renderer
{
	bool use_identity;
	bool use_texture;
	bool use_expression;

	float *d_IX, *d_IY, *d_IZ; // For now these pointers are managed in main.cu. They will be moved to here
	float *d_EX, *d_EY, *d_EZ; // For now these pointers are managed in main.cu. They will be moved to here

	float *d_EX_row_major, *d_EY_row_major, *d_EZ_row_major;
	float *d_IX_row_major, *d_IY_row_major, *d_IZ_row_major;
	float *d_TEX_row_major;

	float *d_TEX;

	float *d_mu_tex;

	float *ALL_VARS;
	float *X0_mean, *Y0_mean, *Z0_mean;
	float *X0, *Y0, *Z0;
	float *X, *Y, *Z;
	float *tmpX_ID, *tmpY_ID, *tmpZ_ID;

	float *d_REX, *d_REY, *d_REZ;
	float *d_RIX, *d_RIY, *d_RIZ;
	float *d_RTEX;

	float *d_grad;
	float *d_texIm;

	ushort T;

	ushort *x0_short, *y0_short;


	/*
	 * This is where we keep the Z coordinate of the face
	 * (i.e. the array has NPTS points)
	 */
	float *d_Z;

	/*
	 * This is where we'll keep Z values that will be used only
	 * for Z buffering (actually they won't be very accurate
	 * but sufficiently accurate for Z buffering)
	 */
	float *d_Ztmp;


	/*
	 * triangle_idx[i] will contain the index of the triangle
	 * from which the ith pixel will be rendered. Hence its size is Nredundant
	 */
	ushort *d_triangle_idx;


	/**
	 * xp, yp contain the face mesh points mapped onto 2d according by a perspective
	 * transformation. Hence each is an array of size NPTS.
	 */
	float *d_xp, *d_yp;

	/**
	 * A critical variable. Not all of Nredundant points that we store
	 * will be rendered. d_rend_flag[i] is a boolean that specifies
	 * if the ith point will be rendered. Hence d_rend_flag is of size
	 * Nredundant.
	 */
	bool 	*d_rend_flag, *d_rend_flag_tmp;


	/**
	 * When we do Z buffering we'll need the indices of the pixels that
	 * will be rendered (d_pixel_idx). This way, we can determine the pixels
	 * for which there are more than one candidate points to render, and we
	 * can eliminate the ones that are occluded (i.e. Z buffering -- we
	 * eliminate the pixels whose Z value is the largest)
	 */
	ushort	*d_pixel_idx, *d_pixel_idx2, *d_pixel_idx2_unique;

	/**
	 * The barycentric coordinates of each pixel that will be rendered.
	 */
	float *d_alphas_redundant, *d_betas_redundant, *d_gammas_redundant;

    float *d_Zmins;

	ushort Kalpha, Kbeta, Kepsilon;

	ushort *d_inner_idx, *d_inner_idx_unique;
    ushort *d_tl;

    uint *d_redundant_idx;

	size_t pitch;
	size_t pitch2;
	size_t pitch3;

	thrust::device_vector<float> d_ones; //(NPTS, 1.f);

    vector< vector<int> > tl_vector;


    /**
     * The structures below are for the texture objects
     * @BEGIN
     **/
    //! IDENTITY
    struct cudaResourceDesc ix_resDesc, iy_resDesc, iz_resDesc;
    struct cudaTextureDesc ix_texDesc, iy_texDesc, iz_texDesc;
    cudaTextureObject_t ix_tex, iy_tex, iz_tex;

    //! EXPRESSION
    struct cudaResourceDesc ex_resDesc, ey_resDesc, ez_resDesc;
    struct cudaTextureDesc ex_texDesc, ey_texDesc, ez_texDesc;
    cudaTextureObject_t ex_tex, ey_tex, ez_tex;

    //! TEXTURE
    struct cudaResourceDesc tex_resDesc;
    struct cudaTextureDesc tex_texDesc;
    cudaTextureObject_t tex_tex;
    /**
     * @END
     **/


	Renderer(uint T_, ushort _Kalpha, ushort _Kbeta, ushort _Kepsilon,
			bool use_identity_, bool use_texture_, bool use_expression_,
			float *h_X0 = NULL, float *h_Y0 = NULL, float *h_Z0 = NULL, float *h_tex_mu = NULL);

    void set_x0_short_y0_short(uint t, float *xp, float *yp, size_t array_size=NLANDMARKS_51, bool pad=config::PAD_RENDERING);

    void initialize_texture_memories();

	void render(uint t, Optimizer& o,  OptimizationVariables& ov,  const float *R, cublasHandle_t& handle,
            ushort *N_unique_pixels, float *d_cropped_face, float *d_buffer_face,  bool visualize = false, bool reset_texim=true);

	void render_for_illumination_only(uint t, Optimizer& o,  OptimizationVariables& ov,  const float *R, cublasHandle_t& handle, ushort *N_unique_pixels,
			float *d_cropped_face, float *d_buffer_face,  bool visualize = false);

	bool compute_nonrigid_shape2(cublasHandle_t &handle, const OptimizationVariables& ov, const float* R, const Camera& cam);

    void compute_nonrigid_shape_identityonly(cublasHandle_t &handle, const OptimizationVariables& ov);

    void  compute_nonrigid_shape_expression_and_rotation(cublasHandle_t &handle, const OptimizationVariables& ov,
                                                                const float* R, float* Xcur, float*  Ycur, float *Zcur);
    void  compute_nonrigid_shape_identity_and_rotation(cublasHandle_t &handle, const OptimizationVariables& ov,
                                                                const float* R, float* Xcur, float*  Ycur, float *Zcur);

	void compute_texture(cublasHandle_t &handle, const OptimizationVariables& ov, Optimizer &o);

    void print_obj(const std::string& obj_path);
    void print_obj_neutral(const std::string& obj_path);

    void print_mat_txt(const std::string& mat_path);

    void print_sparse_2Dpts(const std::string& pts_path, float _resize_coefl = 1.0f);
    void print_sparse_3Dpts(const std::string& pts_path);

	~Renderer();
};

#endif /* RENDERER_H_ */
