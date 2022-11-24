/*
 * model_fitter.h
 *
 *  Created on: Oct 5, 2020
 *      Author: root
 */

#ifndef MODEL_FITTER_H_
#define MODEL_FITTER_H_


#include "cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "constants.h"
#include "renderer.h"

#include "derivative_computer.h"
#include "logbarrier_initializer.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

extern float RESIZE_COEF;


bool fit_3DMM_shape_rigid(uint t, Renderer& r, Optimizer& o, Logbarrier_Initializer& li, cusolverDnHandle_t& handleDn,
        cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
        OptimizationVariables &ov, OptimizationVariables &ov_linesearch, std::vector<Camera> &cams, RotationComputer &rc,
        RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s,  float *d_tmp, bool visualize);


bool fit_to_single_image_autolandmarks(const std::string& im_path,
                                       std::vector<float>& xp_landmark, std::vector<float>& yp_landmark,
                                       std::vector<float>& xrange, std::vector<float>& yrange,
    const std::string& result_basepath,
    Renderer& r, Optimizer& o,
    cusolverDnHandle_t& handleDn,
    cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
    OptimizationVariables &ov, OptimizationVariables &ov_linesearch,
    OptimizationVariables &ov_lb, OptimizationVariables &ov_lb_linesearch,
    RotationComputer &rc,
    RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, Solver &s_lambda, float *d_tmp,
    float *search_dir_Lintensity, float *dg_Lintensity, std::vector<Camera> &cams, bool fit_landmarks_only = false);

bool fit_to_multi_images(std::vector<Camera> &cam, const std::vector<std::vector<float> >& all_xps, const std::vector<std::vector<float> >& all_yps,
                         const std::vector<std::vector<float> >& all_xranges, const std::vector<std::vector<float> >& all_yranges,
        const std::vector<cv::Mat>& frames,
        std::vector<std::string>* result_basepath,
        Renderer& r, Optimizer& o,
        cusolverDnHandle_t& handleDn,
        cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
        OptimizationVariables &ov, OptimizationVariables &ov_linesearch,
        OptimizationVariables &ov_lb, OptimizationVariables &ov_lb_linesearch,
        RotationComputer &rc,
        RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, Solver &s_lambda, float *d_tmp,
        float *search_dir_Lintensity, float *dg_Lintensity,
        float *h_X0, float *h_Y0, float *h_Z0, float *h_tex_mu);


void fit_3DMM_Lintensity(uint t, Renderer& r, Optimizer& o,
        cublasHandle_t& handle,   float *d_cropped_face, float *d_buffer_face,
        OptimizationVariables &ov, OptimizationVariables &ov_linesearch, Camera& cam, RotationComputer &rc,
        RotationComputer& rc_linesearch, DerivativeComputer &dc, float *search_dir_Lintensity, float *dg,  float *d_tmp,
        bool visualize, bool initialize_texture);


void fit_3DMM_lambdas(uint t, Renderer& r, Optimizer& o, cusolverDnHandle_t& handleDn,
        cublasHandle_t& handle,  float *d_cropped_face, float *d_buffer_face,
        OptimizationVariables &ov, OptimizationVariables &ov_linesearch, Camera& cam, RotationComputer &rc,
        RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s_lambda,  float *d_tmp,
        bool visualize, bool initialize_texture);

bool fit_3DMM_epsilon_finetune(uint t, Renderer& r, Optimizer& o, Logbarrier_Initializer& li, cusolverDnHandle_t& handleDn,
                          cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
                          OptimizationVariables &ov, OptimizationVariables &ov_linesearch, std::vector<Camera> & cams, RotationComputer &rc,
                          RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, float *d_tmp, bool visualize);

bool fit_3DMM_rigid_alone(uint t, Renderer& r, Optimizer& o, Logbarrier_Initializer& li, cusolverDnHandle_t& handleDn,
                          cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
                          OptimizationVariables &ov, OptimizationVariables &ov_linesearch, std::vector<Camera> & cams, RotationComputer &rc,
                          RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, float *d_tmp, bool visualize);

bool fit_to_multi_images_landmarks_only(const std::vector<std::vector<float> >& all_xps, const std::vector<std::vector<float> >& all_yps,
                                        const std::vector<std::vector<float> > &all_xranges, const std::vector<std::vector<float> > &all_yranges,
                                        Renderer& r, cusolverDnHandle_t& handleDn, cublasHandle_t& handle,
                                        RotationComputer &rc, RotationComputer& rc_linesearch, std::vector<Camera> &cams);

bool update_shape_single_resize_coef(float *xp_orig, float *yp_orig, std::vector<float>& xrange, std::vector<float>& yrange,
                        const cv::Mat& frame, std::vector<Camera> &cam, Renderer& r, Optimizer& o, cusolverDnHandle_t& handleDn,
                        cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
                        Logbarrier_Initializer &li, Logbarrier_Initializer &li_init,
                        OptimizationVariables &ov, OptimizationVariables &ov_linesearch,
                        OptimizationVariables &ov_lb, OptimizationVariables &ov_lb_linesearch, RotationComputer &rc,
                        RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, Solver &s_lambda, float *d_tmp,
                        float *search_dir_Lintensity, float *dg_Lintensity, const uint frame_id,
                        cv::VideoWriter *outputVideo,
                        const std::string& outputVideoPath,
                        float& ref_face_size, float cur_face_size, int *min_x, int *max_x, int* min_y, int* max_y, int FPS,
                                     float cur_resize_coef,
                                     std::vector<float> &exp_coeffs_combined, float *Xtmp, float *Ytmp, float *Ztmp,
                                     float* h_X0_cur, float* h_Y0_cur, float* h_Z0_cur,
                                     float* h_Xr_cur, float* h_Yr_cur, float* h_Zr_cur,
                                     vector<float>& X0cur_vec, vector<float>& Y0cur_vec, vector<float>& Z0cur_vec,
                                     vector<float>& Xrcur_vec, vector<float>& Yrcur_vec, vector<float>& Zrcur_vec);


bool fit_to_video_frame(float *xp, float *yp, std::vector<float> &xrange, std::vector<float> &yrange,
        const cv::Mat& inputImage_color,
        std::vector<Camera> &cams,
        Renderer& r, Optimizer& o,
        cusolverDnHandle_t& handleDn,
        cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
        Logbarrier_Initializer &li, Logbarrier_Initializer &li_init,
        OptimizationVariables &ov, OptimizationVariables &ov_linesearch,
        OptimizationVariables &ov_lb, OptimizationVariables &ov_lb_linesearch,
        RotationComputer &rc,
        RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, Solver &s_lambda, float *d_tmp,
        float *search_dir_Lintensity, float *dg_Lintensity, const uint frame_id, cv::VideoWriter *outputVideo,
                        const std::string& outputFilePath,
                        float &ref_face_size, float cur_face_size,
                        int *min_x = NULL, int *max_x = NULL, int *min_y = NULL, int *max_y = NULL, int FPS = 30);




#endif /* MODEL_FITTER_H_ */
