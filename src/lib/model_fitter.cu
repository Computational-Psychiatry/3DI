/*
 * model_fitter.cu
 *
 *  Created on: Oct 5, 2020
 *      Author: root
 */

#include "config.h"
#include "model_fitter.h"
#include "constants.h"
#include "funcs.h"
#include "preprocessing.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <experimental/filesystem>

#ifdef VISUALIZE_3D
#include "GLfuncs.h"
#endif

void get_obj_hessian_and_gradient_multiframe(Renderer& r, Optimizer& o, Logbarrier_Initializer& li,
                                             cublasHandle_t& handle,  float *d_cropped_face, float *d_buffer_face,
                                             OptimizationVariables &ov, std::vector<Camera> &cams, RotationComputer &rc, DerivativeComputer &dc,
                                             ushort &N_unique_pixels, bool visualize)
{

    cudaMemset(ov.grad_corr, 0,  sizeof(float));

    for (ushort t=0; t<ov.T; ++t)
    {
        // <!-- We'll probably need to change the below -->
        // Update camera at each frame

        ov.set_frame(t);
        rc.set_u_ptr(ov.u);

        rc.process();

        // <!-- We'll probably need to change the below --> use the right camera (e.g., cams[t])
        r.compute_nonrigid_shape2(handle, ov, rc.R, cams[t]);
        r.compute_texture(handle, ov, o);

#ifdef MEASURE_TIME
        cudaEvent_t start_t, stop_t;
        cudaEventCreate( &start_t);
        cudaEventCreate( &stop_t);
        cudaEventRecord( start_t, 0 );
#endif
        r.render(t, o, ov, rc.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, visualize);

#ifdef MEASURE_TIME
        cudaEventRecord( stop_t, 0 );
        cudaEventSynchronize( stop_t );

        float   elapsedTime_t;
        cudaEventElapsedTime( &elapsedTime_t, start_t, stop_t );
        printf( "TM_Render_Frame = %.2f ms\n", elapsedTime_t);
#endif


#ifdef MEASURE_TIME
        cudaEventCreate( &start_t );
        cudaEventCreate( &stop_t );
        cudaEventRecord( start_t, 0 );
#endif
        if (r.use_expression) {
            render_expression_basis_texture_colmajor_rotated2<<<N_unique_pixels, r.Kepsilon>>>(r.d_alphas_redundant, r.d_betas_redundant, r.d_gammas_redundant,  r.d_redundant_idx, N_unique_pixels,
                                                                                               r.d_tl, r.d_REX, r.d_REY, r.d_REZ, r.d_triangle_idx, rc.R);
        }

        if (r.use_identity) {
            render_identity_basis_texture<<<N_unique_pixels, r.Kalpha>>>(r.d_alphas_redundant, r.d_betas_redundant, r.d_gammas_redundant, r.d_redundant_idx, N_unique_pixels,
                                                                         r.d_tl, r.d_RIX, r.d_RIY, r.d_RIZ, r.d_triangle_idx, r.Kalpha);
        }

        if (r.use_texture) {
            render_texture_basis_texture<<<N_unique_pixels, r.Kbeta>>>(r.d_alphas_redundant, r.d_betas_redundant, r.d_gammas_redundant, r.d_redundant_idx, N_unique_pixels,
                                                                       r.d_tl, r.d_RTEX,  r.d_triangle_idx, r.Kbeta);
        }

#ifdef MEASURE_TIME
        cudaEventRecord( stop_t, 0 );
        cudaEventSynchronize( stop_t );

        cudaEventElapsedTime( &elapsedTime_t, start_t , stop_t );
        printf( "TM_Render_Bases = %.2f ms\n", elapsedTime_t);
#endif

        dc.compute_hessian_and_gradient(t, o, rc, r, cams[t], N_unique_pixels, handle, li);

    }
}




bool fit_3DMM_shape_rigid(uint t, Renderer& r, Optimizer& o, Logbarrier_Initializer& li, cusolverDnHandle_t& handleDn,
                          cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
                          OptimizationVariables &ov, OptimizationVariables &ov_linesearch, std::vector<Camera> & cams, RotationComputer &rc,
                          RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, float *d_tmp, bool visualize)
{
    ov.set_frame(t);
    ov_linesearch.set_frame(t);

    rc.set_u_ptr(ov.u);
    rc_linesearch.set_u_ptr(ov_linesearch.u);

    ushort N_unique_pixels;

    float plus_one_ = 1.0f;
    float h_searchdir_0;

    float yummymulti[] = {5.0f}; //  {10.0f};
    float yummy[1] = {5.0f}; // {10.0f};

    float alpha_ = 1.0f;
    float beta_ = 0.0f;

    cudaMemcpy(ov.tau_logbarrier, yummy, sizeof(float), cudaMemcpyHostToDevice);

    float *logbarrier_multi_coef;
    HANDLE_ERROR( cudaMalloc( (void**)&logbarrier_multi_coef, sizeof(float)) );
    cudaMemcpy(logbarrier_multi_coef, yummymulti, sizeof(float), cudaMemcpyHostToDevice);

    float tcoef_threshold = 0.075;
    if (ov.T > 1)
	    tcoef_threshold = 0.001;

    uint num_outer_iters = 0;
    uint num_inner_iters = 0;
    for (uint tau_iter = 0; tau_iter<1; ++tau_iter)
    {
        bool terminate = false;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha_, ov.tau_logbarrier, 1, logbarrier_multi_coef, 1, &beta_, ov.tau_logbarrier, 1);
        cudaMemcpy(ov_linesearch.tau_logbarrier, ov.tau_logbarrier, sizeof(float), cudaMemcpyDeviceToDevice);

        //        std::cout << "TAU_ITER" << tau_iter << std::endl;

        for (int i=0; i<100; ++i)
        {
            num_outer_iters++;
            if (terminate)
                break;

#ifdef MEASURE_TIME
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            get_obj_hessian_and_gradient_multiframe(r, o, li, handle,
                                                    d_cropped_face, d_buffer_face, ov, cams, rc, dc,
                                                    N_unique_pixels, false);
#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            float   elapsedTime_t;
            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_Hessian_computation= time %.2f ms\n", elapsedTime_t);
#endif

#ifdef MEASURE_TIME
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            copyMatsFloatToDouble<<<(s.n*s.n+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.JTJ, o.dG_dtheta, s.JTJ, s.dG_dtheta, s.n);
            bool solve_success = s.solve(handleDn);
#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_solve_system = time %.2f ms\n", elapsedTime_t);
#endif

            cudaMemcpy(&h_searchdir_0, s.search_dir, sizeof(float), cudaMemcpyDeviceToHost);

            if (isnan(h_searchdir_0)) {
                if (config::PRINT_WARNINGS)
                    std::cout << "was NAN at iteration# " << i << std::endl;
                //std::cout << "\t This is most likely caused by some denominators in compute_Gs() function being too close to zero." << std::endl;
                //std::cout << "\t However, this is doesn't affect results much and happens rarely enough to not make this a priority for now." << std::endl;
                terminate = true;
                break;
            }

            float t_coef = 1.0f;
            //float ALPHA = 0.2; // 0.4f; #TIME_MODIFICATIONS
            //float BETA = 0.33; // 0.5f; #TIME_MODIFICATIONS

            float ALPHA = 0.4f; // #TIME_MODIFICATIONS
            float BETA = 0.8f; //#TIME_MODIFICATIONS

            const uint MAX_INNER_ITERS = 100;
            uint inner_iter = 0;

            float obj[1], obj_tmp[1];
            cudaMemcpy(obj, ov.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);
            obj[0] = -obj[0];

#ifdef MEASURE_TIME
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            //		break;
            float lambda2[1];

            while (inner_iter < MAX_INNER_ITERS)
            {
                num_inner_iters++;
//                if (t_coef < 0.032f) {
                if (t_coef < tcoef_threshold) {
                    terminate = true;
                    break;
                }

                bool skip_because_nan = false;

                set_xtmp<<<(ov.Ktotal+NTHREADS-1)/NTHREADS, NTHREADS>>>(s.search_dir, ov.betas, t_coef, ov_linesearch.betas, ov_linesearch.Ktotal);
                cudaMemset(ov_linesearch.grad_corr, 0, sizeof(float));

                float cur_tz;

                cudaMemcpy(&cur_tz, ov_linesearch.tauz, sizeof(float), cudaMemcpyDeviceToHost);

                if (cur_tz<0.0f) {
                    if (config::PRINT_WARNINGS)
                        std::cout << "__cur_tz: " << cur_tz << std::endl;
                    break;
                }


                float *d_obj_ineqs;
                HANDLE_ERROR(cudaMalloc((void**)&d_obj_ineqs, sizeof(float)*1));
                cudaMemset(d_obj_ineqs, 0, sizeof(float));

                for (size_t t_inner=0; t_inner<ov.T; ++t_inner)
                {
                    if (skip_because_nan)
                        break;

                    ov_linesearch.set_frame(t_inner);
                    rc_linesearch.set_u_ptr(ov_linesearch.u);
                    rc_linesearch.process_without_derivatives();

                    // <!-- We'll probably need to change the below --> use the right camera (e.g., cams[t])
                    bool is_face_reasonable = r.compute_nonrigid_shape2(handle, ov_linesearch, rc_linesearch.R, cams[t_inner]);
                    if (!is_face_reasonable) {
                        if (config::PRINT_WARNINGS)
                           std::cout << "face is too large! " << std::endl;
                        skip_because_nan = true;
                        break;
                    }
                    // r.compute_texture(handle, ov_linesearch, o);
                    // r.render(t_inner, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, visualize); // #TIME_MODIFICATIONS - REMOVE/COMMENT THIS LINE

                    // We do this only if t_inner == 0 because the
                    // function li.evaluate_objective_function() takes care for all t=1:T
                    if (t_inner == 0 && dc.use_inequalities)
                    {
                        cudaMemset(d_obj_ineqs, 0, sizeof(float));
                        li.evaluate_objective_function(handle, &ov_linesearch, d_obj_ineqs);

                        if (dc.use_texture)
                        {
                            neglogify<<<(ov_linesearch.Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_ub, ov_linesearch.Kbeta);
                            neglogify<<<(ov_linesearch.Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_lb, ov_linesearch.Kbeta);

                            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov_linesearch.Kbeta, &plus_one_, li.vecOnes, 1, li.f_beta_lb, ov_linesearch.Kbeta, &plus_one_, d_obj_ineqs, 1);
                            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov_linesearch.Kbeta, &plus_one_, li.vecOnes, 1, li.f_beta_ub, ov_linesearch.Kbeta, &plus_one_, d_obj_ineqs, 1);
                        }

                        /* #TIME_MODIFICATIONS ? */
                        float h_obj_ineqs(0.0f);
                        cudaMemcpy(&h_obj_ineqs, d_obj_ineqs, sizeof(float), cudaMemcpyDeviceToHost);
                        if (isnan(h_obj_ineqs)) {
                            skip_because_nan = true;
                            break;
                        }
                        /* #TIME_MODIFICATIONS? ^ */

                        //add_to_scalar_negated<<<1,1>>>(ov_linesearch.grad_corr,  d_obj_ineqs); // #TIME_MODIFICATIONS - REMOVE/COMMENT THIS LINE
                        ov_linesearch.set_frame(t_inner);
                    }

                    if (t_inner == 0) 
                        r.compute_texture(handle, ov_linesearch, o);
                    // #TIME_MODIFICATIONS - BEGIN - PUT BACK THE FOLLOWING
                    r.render(t_inner, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, false);
                    if (t_inner == 0) {
                        add_to_scalar_negated<<<1,1>>>(ov_linesearch.grad_corr,  d_obj_ineqs);
                    }
                    /*
                    */
                }
                HANDLE_ERROR( cudaFree( d_obj_ineqs ) );

                if (skip_because_nan) {
                    t_coef = t_coef * BETA;
                    continue;
                }

                cudaMemcpy(obj_tmp, ov_linesearch.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);

                obj_tmp[0] = -obj_tmp[0];

                inner_iter++;
                cublasSdot(handle, ov.Ktotal, s.search_dir, 1, o.dG_dtheta, 1, d_tmp);

                cudaMemcpy(lambda2, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
                //                if (false) { // (-lambda2[0] < 0.0003) {
                if (-lambda2[0] < 0.0003) {
                    terminate = true;
                    break;
                }
                lambda2[0] *= ALPHA*t_coef;

                if (obj_tmp[0]  < obj[0] + lambda2[0]) {
                    cudaMemcpy(ov.betas, ov_linesearch.betas, sizeof(float)*ov.Ktotal, cudaMemcpyDeviceToDevice);
                    //std::cout << '#' << i << ": "<< t_coef << '\t' <<"obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
                    break;
                } else {
                    //std::cout << "\t\t t_coef:"  << t_coef <<"  obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
                }
                t_coef = t_coef * BETA;
                //			break;
            }

#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_linesearch = time %.2f ms\n", elapsedTime_t);
#endif
        }
    }

    if (visualize) {
        r.render(0, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face,  true);
    }

    HANDLE_ERROR( cudaFree(logbarrier_multi_coef) );

    return num_outer_iters > 6;
}





















bool fit_3DMM_epsilon_finetune(uint t, Renderer& r, Optimizer& o, Logbarrier_Initializer& li, cusolverDnHandle_t& handleDn,
                          cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
                          OptimizationVariables &ov, OptimizationVariables &ov_linesearch, std::vector<Camera> & cams, RotationComputer &rc,
                          RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, float *d_tmp, bool visualize)
{

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    HANDLE_ERROR( cudaMemcpy( li.epsilon_lb, li.epsilon_lb_finetune, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );
    HANDLE_ERROR( cudaMemcpy( li.epsilon_ub, li.epsilon_ub_finetune, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );


    ov.set_frame(t);
    ov_linesearch.set_frame(t);

    rc.set_u_ptr(ov.u);
    rc_linesearch.set_u_ptr(ov_linesearch.u);

    ushort N_unique_pixels;

    float plus_one_ = 1.0f;
    float h_searchdir_0;

    float yummymulti[] = {5.0f}; //  {10.0f};
    float yummy[1] = {5.0f}; // {10.0f};

    float alpha_ = 1.0f;
    float beta_ = 0.0f;

    cudaMemcpy(ov.tau_logbarrier, yummy, sizeof(float), cudaMemcpyHostToDevice);

    float *logbarrier_multi_coef;
    HANDLE_ERROR( cudaMalloc( (void**)&logbarrier_multi_coef, sizeof(float)) );
    cudaMemcpy(logbarrier_multi_coef, yummymulti, sizeof(float), cudaMemcpyHostToDevice);

    float tcoef_threshold = 0.075;
    if (ov.T > 1)
        tcoef_threshold = 0.001;

    uint num_outer_iters = 0;
    uint num_inner_iters = 0;
    for (uint tau_iter = 0; tau_iter<1; ++tau_iter)
    {
        bool terminate = false;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha_, ov.tau_logbarrier, 1, logbarrier_multi_coef, 1, &beta_, ov.tau_logbarrier, 1);
        cudaMemcpy(ov_linesearch.tau_logbarrier, ov.tau_logbarrier, sizeof(float), cudaMemcpyDeviceToDevice);

        //        std::cout << "TAU_ITER" << tau_iter << std::endl;

        for (int i=0; i<100; ++i)
        {
            num_outer_iters++;
            if (terminate)
                break;

#ifdef MEASURE_TIME
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            get_obj_hessian_and_gradient_multiframe(r, o, li, handle,
                                                    d_cropped_face, d_buffer_face, ov, cams, rc, dc,
                                                    N_unique_pixels, false);
#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            float   elapsedTime_t;
            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_Hessian_computation= time %.2f ms\n", elapsedTime_t);
#endif

#ifdef MEASURE_TIME
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            copyMatsFloatToDouble<<<(s.n*s.n+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.JTJ, o.dG_dtheta, s.JTJ, s.dG_dtheta, s.n);
            bool solve_success = s.solve(handleDn);
#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_solve_system = time %.2f ms\n", elapsedTime_t);
#endif

            cudaMemcpy(&h_searchdir_0, s.search_dir, sizeof(float), cudaMemcpyDeviceToHost);

            if (isnan(h_searchdir_0)) {
                if (config::PRINT_WARNINGS)
                    std::cout << "was NAN at iteration# " << i << std::endl;
                //std::cout << "\t This is most likely caused by some denominators in compute_Gs() function being too close to zero." << std::endl;
                //std::cout << "\t However, this is doesn't affect results much and happens rarely enough to not make this a priority for now." << std::endl;
                terminate = true;
                break;
            }

            float t_coef = 1.0f;
            //float ALPHA = 0.2; // 0.4f; #TIME_MODIFICATIONS
            //float BETA = 0.33; // 0.5f; #TIME_MODIFICATIONS

            float ALPHA = 0.4f; // #TIME_MODIFICATIONS
            float BETA = 0.8f; //#TIME_MODIFICATIONS

            const uint MAX_INNER_ITERS = 100;
            uint inner_iter = 0;

            float obj[1], obj_tmp[1];
            cudaMemcpy(obj, ov.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);
            obj[0] = -obj[0];

#ifdef MEASURE_TIME
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            //		break;
            float lambda2[1];

            while (inner_iter < MAX_INNER_ITERS)
            {
                num_inner_iters++;
//                if (t_coef < 0.032f) {
                if (t_coef < tcoef_threshold) {
                    terminate = true;
                    break;
                }

                bool skip_because_nan = false;

                set_xtmp<<<((ov.Ktotal-6)+NTHREADS-1)/NTHREADS, NTHREADS>>>(s.search_dir, ov.betas, t_coef, ov_linesearch.betas, ov_linesearch.Ktotal-6);
                cudaMemset(ov_linesearch.grad_corr, 0, sizeof(float));

                float cur_tz;

                cudaMemcpy(&cur_tz, ov_linesearch.tauz, sizeof(float), cudaMemcpyDeviceToHost);

                if (cur_tz<0.0f) {
                    std::cout << "__cur_tz: " << cur_tz << std::endl;
                    break;
                }


                float *d_obj_ineqs;
                HANDLE_ERROR(cudaMalloc((void**)&d_obj_ineqs, sizeof(float)*1));
                cudaMemset(d_obj_ineqs, 0, sizeof(float));

                for (size_t t_inner=0; t_inner<ov.T; ++t_inner)
                {
                    if (skip_because_nan)
                        break;

                    ov_linesearch.set_frame(t_inner);
                    rc_linesearch.set_u_ptr(ov_linesearch.u);
                    rc_linesearch.process_without_derivatives();

                    // <!-- We'll probably need to change the below --> use the right camera (e.g., cams[t])
                    bool is_face_reasonable = r.compute_nonrigid_shape2(handle, ov_linesearch, rc_linesearch.R, cams[t_inner]);
                    if (!is_face_reasonable) {
                        std::cout << "face is too large! " << std::endl;
                        skip_because_nan = true;
                        break;
                    }
                    // r.compute_texture(handle, ov_linesearch, o);
                    // r.render(t_inner, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, visualize); // #TIME_MODIFICATIONS - REMOVE/COMMENT THIS LINE

                    // We do this only if t_inner == 0 because the
                    // function li.evaluate_objective_function() takes care for all t=1:T
                    if (t_inner == 0 && dc.use_inequalities)
                    {
                        cudaMemset(d_obj_ineqs, 0, sizeof(float));
                        li.evaluate_objective_function(handle, &ov_linesearch, d_obj_ineqs);

                        if (dc.use_texture)
                        {
                            neglogify<<<(ov_linesearch.Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_ub, ov_linesearch.Kbeta);
                            neglogify<<<(ov_linesearch.Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_lb, ov_linesearch.Kbeta);

                            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov_linesearch.Kbeta, &plus_one_, li.vecOnes, 1, li.f_beta_lb, ov_linesearch.Kbeta, &plus_one_, d_obj_ineqs, 1);
                            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov_linesearch.Kbeta, &plus_one_, li.vecOnes, 1, li.f_beta_ub, ov_linesearch.Kbeta, &plus_one_, d_obj_ineqs, 1);
                        }

                        /* #TIME_MODIFICATIONS ? */
                        float h_obj_ineqs(0.0f);
                        cudaMemcpy(&h_obj_ineqs, d_obj_ineqs, sizeof(float), cudaMemcpyDeviceToHost);
                        if (isnan(h_obj_ineqs)) {
                            skip_because_nan = true;
                            break;
                        }
                        /* #TIME_MODIFICATIONS? ^ */

                        //add_to_scalar_negated<<<1,1>>>(ov_linesearch.grad_corr,  d_obj_ineqs); // #TIME_MODIFICATIONS - REMOVE/COMMENT THIS LINE
                        ov_linesearch.set_frame(t_inner);
                    }

                    if (t_inner == 0)
                        r.compute_texture(handle, ov_linesearch, o);
                    // #TIME_MODIFICATIONS - BEGIN - PUT BACK THE FOLLOWING
                    r.render(t_inner, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, false);
                    if (t_inner == 0) {
                        add_to_scalar_negated<<<1,1>>>(ov_linesearch.grad_corr,  d_obj_ineqs);
                    }
                    /*
                    */
                }
                HANDLE_ERROR( cudaFree( d_obj_ineqs ) );

                if (skip_because_nan) {
                    t_coef = t_coef * BETA;
                    continue;
                }

                cudaMemcpy(obj_tmp, ov_linesearch.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);

                obj_tmp[0] = -obj_tmp[0];

                inner_iter++;
                cublasSdot(handle, ov.Ktotal, s.search_dir, 1, o.dG_dtheta, 1, d_tmp);

                cudaMemcpy(lambda2, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
                //                if (false) { // (-lambda2[0] < 0.0003) {
                if (-lambda2[0] < 0.0003) {
                    terminate = true;
                    break;
                }
                lambda2[0] *= ALPHA*t_coef;

                if (obj_tmp[0]  < obj[0] + lambda2[0]) {
                    cudaMemcpy(ov.betas, ov_linesearch.betas, sizeof(float)*ov.Ktotal, cudaMemcpyDeviceToDevice);
                    //std::cout << '#' << i << ": "<< t_coef << '\t' <<"obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
                    break;
                } else {
                    //std::cout << "\t\t t_coef:"  << t_coef <<"  obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
                }
                t_coef = t_coef * BETA;
                //			break;
            }

#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_linesearch = time %.2f ms\n", elapsedTime_t);
#endif
        }
    }

    if (visualize) {
        r.render(0, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face,  true);
    }

    HANDLE_ERROR( cudaFree(logbarrier_multi_coef) );

    HANDLE_ERROR( cudaMemcpy( li.epsilon_lb, li.epsilon_lb_regular, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );
    HANDLE_ERROR( cudaMemcpy( li.epsilon_ub, li.epsilon_ub_regular, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );

    return num_outer_iters > 6;
}













bool fit_3DMM_rigid_alone(uint t, Renderer& r, Optimizer& o, Logbarrier_Initializer& li, cusolverDnHandle_t& handleDn,
                          cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
                          OptimizationVariables &ov, OptimizationVariables &ov_linesearch, std::vector<Camera> & cams, RotationComputer &rc,
                          RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, float *d_tmp, bool visualize)
{

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    HANDLE_ERROR( cudaMemcpy( li.epsilon_lb, li.epsilon_lb_finetune, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );
    HANDLE_ERROR( cudaMemcpy( li.epsilon_ub, li.epsilon_ub_finetune, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );


    ov.set_frame(t);
    ov_linesearch.set_frame(t);

    rc.set_u_ptr(ov.u);
    rc_linesearch.set_u_ptr(ov_linesearch.u);

    ushort N_unique_pixels;

    float plus_one_ = 1.0f;
    float h_searchdir_0;

    float yummymulti[] = {5.0f}; //  {10.0f};
    float yummy[1] = {5.0f}; // {10.0f};

    float alpha_ = 1.0f;
    float beta_ = 0.0f;

    cudaMemcpy(ov.tau_logbarrier, yummy, sizeof(float), cudaMemcpyHostToDevice);

    float *logbarrier_multi_coef;
    HANDLE_ERROR( cudaMalloc( (void**)&logbarrier_multi_coef, sizeof(float)) );
    cudaMemcpy(logbarrier_multi_coef, yummymulti, sizeof(float), cudaMemcpyHostToDevice);

    float tcoef_threshold = 0.075;
    if (ov.T > 1)
        tcoef_threshold = 0.001;

    uint num_outer_iters = 0;
    uint num_inner_iters = 0;
    for (uint tau_iter = 0; tau_iter<1; ++tau_iter)
    {
        bool terminate = false;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha_, ov.tau_logbarrier, 1, logbarrier_multi_coef, 1, &beta_, ov.tau_logbarrier, 1);
        cudaMemcpy(ov_linesearch.tau_logbarrier, ov.tau_logbarrier, sizeof(float), cudaMemcpyDeviceToDevice);

        //        std::cout << "TAU_ITER" << tau_iter << std::endl;

        for (int i=0; i<100; ++i)
        {
            num_outer_iters++;
            if (terminate)
                break;

#ifdef MEASURE_TIME
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            get_obj_hessian_and_gradient_multiframe(r, o, li, handle,
                                                    d_cropped_face, d_buffer_face, ov, cams, rc, dc,
                                                    N_unique_pixels, false);
#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            float   elapsedTime_t;
            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_Hessian_computation= time %.2f ms\n", elapsedTime_t);
#endif

#ifdef MEASURE_TIME
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            copyMatsFloatToDouble<<<(s.n*s.n+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.JTJ, o.dG_dtheta, s.JTJ, s.dG_dtheta, s.n);
            bool solve_success = s.solve(handleDn);
#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_solve_system = time %.2f ms\n", elapsedTime_t);
#endif

            cudaMemcpy(&h_searchdir_0, s.search_dir, sizeof(float), cudaMemcpyDeviceToHost);

            if (isnan(h_searchdir_0)) {
                if (config::PRINT_WARNINGS)
                    std::cout << "was NAN at iteration# " << i << std::endl;
                //std::cout << "\t This is most likely caused by some denominators in compute_Gs() function being too close to zero." << std::endl;
                //std::cout << "\t However, this is doesn't affect results much and happens rarely enough to not make this a priority for now." << std::endl;
                terminate = true;
                break;
            }

            float t_coef = 1.0f;
            //float ALPHA = 0.2; // 0.4f; #TIME_MODIFICATIONS
            //float BETA = 0.33; // 0.5f; #TIME_MODIFICATIONS

            float ALPHA = 0.4f; // #TIME_MODIFICATIONS
            float BETA = 0.8f; //#TIME_MODIFICATIONS

            const uint MAX_INNER_ITERS = 100;
            uint inner_iter = 0;

            float obj[1], obj_tmp[1];
            cudaMemcpy(obj, ov.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);
            obj[0] = -obj[0];

#ifdef MEASURE_TIME
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaEventRecord( start, 0 );
#endif
            //		break;
            float lambda2[1];

            while (inner_iter < MAX_INNER_ITERS)
            {
                num_inner_iters++;
//                if (t_coef < 0.032f) {
                if (t_coef < tcoef_threshold) {
                    terminate = true;
                    break;
                }

                bool skip_because_nan = false;

                set_xtmp<<<((6)+NTHREADS-1)/NTHREADS, NTHREADS>>>(s.search_dir+ov.Kepsilon, ov.betas+ov.Kepsilon, t_coef, ov_linesearch.betas+ov.Kepsilon, 6);
                cudaMemset(ov_linesearch.grad_corr, 0, sizeof(float));

                float cur_tz;

                cudaMemcpy(&cur_tz, ov_linesearch.tauz, sizeof(float), cudaMemcpyDeviceToHost);

                if (cur_tz<0.0f) {
                    if (config::PRINT_WARNINGS)
                       std::cout << "__cur_tz: " << cur_tz << std::endl;
                    break;
                }


                float *d_obj_ineqs;
                HANDLE_ERROR(cudaMalloc((void**)&d_obj_ineqs, sizeof(float)*1));
                cudaMemset(d_obj_ineqs, 0, sizeof(float));

                for (size_t t_inner=0; t_inner<ov.T; ++t_inner)
                {
                    if (skip_because_nan)
                        break;

                    ov_linesearch.set_frame(t_inner);
                    rc_linesearch.set_u_ptr(ov_linesearch.u);
                    rc_linesearch.process_without_derivatives();

                    // <!-- We'll probably need to change the below --> use the right camera (e.g., cams[t])
                    bool is_face_reasonable = r.compute_nonrigid_shape2(handle, ov_linesearch, rc_linesearch.R, cams[t_inner]);
                    if (!is_face_reasonable) {
                        std::cout << "face is too large! " << std::endl;
                        skip_because_nan = true;
                        break;
                    }
                    // r.compute_texture(handle, ov_linesearch, o);
                    // r.render(t_inner, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, visualize); // #TIME_MODIFICATIONS - REMOVE/COMMENT THIS LINE

                    // We do this only if t_inner == 0 because the
                    // function li.evaluate_objective_function() takes care for all t=1:T
                    if (t_inner == 0 && dc.use_inequalities)
                    {
                        cudaMemset(d_obj_ineqs, 0, sizeof(float));
                        li.evaluate_objective_function(handle, &ov_linesearch, d_obj_ineqs);

                        if (dc.use_texture)
                        {
                            neglogify<<<(ov_linesearch.Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_ub, ov_linesearch.Kbeta);
                            neglogify<<<(ov_linesearch.Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_lb, ov_linesearch.Kbeta);

                            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov_linesearch.Kbeta, &plus_one_, li.vecOnes, 1, li.f_beta_lb, ov_linesearch.Kbeta, &plus_one_, d_obj_ineqs, 1);
                            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov_linesearch.Kbeta, &plus_one_, li.vecOnes, 1, li.f_beta_ub, ov_linesearch.Kbeta, &plus_one_, d_obj_ineqs, 1);
                        }

                        /* #TIME_MODIFICATIONS ? */
                        float h_obj_ineqs(0.0f);
                        cudaMemcpy(&h_obj_ineqs, d_obj_ineqs, sizeof(float), cudaMemcpyDeviceToHost);
                        if (isnan(h_obj_ineqs)) {
                            skip_because_nan = true;
                            break;
                        }
                        /* #TIME_MODIFICATIONS? ^ */

                        //add_to_scalar_negated<<<1,1>>>(ov_linesearch.grad_corr,  d_obj_ineqs); // #TIME_MODIFICATIONS - REMOVE/COMMENT THIS LINE
                        ov_linesearch.set_frame(t_inner);
                    }

                    if (t_inner == 0)
                        r.compute_texture(handle, ov_linesearch, o);
                    // #TIME_MODIFICATIONS - BEGIN - PUT BACK THE FOLLOWING
                    r.render(t_inner, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, false);
                    if (t_inner == 0) {
                        add_to_scalar_negated<<<1,1>>>(ov_linesearch.grad_corr,  d_obj_ineqs);
                    }
                    /*
                    */
                }
                HANDLE_ERROR( cudaFree( d_obj_ineqs ) );

                if (skip_because_nan) {
                    t_coef = t_coef * BETA;
                    continue;
                }

                cudaMemcpy(obj_tmp, ov_linesearch.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);

                obj_tmp[0] = -obj_tmp[0];

                inner_iter++;
                cublasSdot(handle, ov.Ktotal, s.search_dir, 1, o.dG_dtheta, 1, d_tmp);

                cudaMemcpy(lambda2, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
                //                if (false) { // (-lambda2[0] < 0.0003) {
                if (-lambda2[0] < 0.0003) {
                    terminate = true;
                    break;
                }
                lambda2[0] *= ALPHA*t_coef;

                if (obj_tmp[0]  < obj[0] + lambda2[0]) {
                    cudaMemcpy(ov.betas, ov_linesearch.betas, sizeof(float)*ov.Ktotal, cudaMemcpyDeviceToDevice);
                    //std::cout << '#' << i << ": "<< t_coef << '\t' <<"obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
                    break;
                } else {
                    //std::cout << "\t\t t_coef:"  << t_coef <<"  obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
                }
                t_coef = t_coef * BETA;
                //			break;
            }

#ifdef MEASURE_TIME
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            cudaEventElapsedTime( &elapsedTime_t, start, stop );
            printf( "TM_linesearch = time %.2f ms\n", elapsedTime_t);
#endif
        }
    }

    if (visualize) {
        r.render(0, o, ov_linesearch, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face,  true);
    }

    HANDLE_ERROR( cudaFree(logbarrier_multi_coef) );

    HANDLE_ERROR( cudaMemcpy( li.epsilon_lb, li.epsilon_lb_regular, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );
    HANDLE_ERROR( cudaMemcpy( li.epsilon_ub, li.epsilon_ub_regular, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );

    return num_outer_iters > 6;
}








































void fit_3DMM_lambdas(uint t, Renderer& r, Optimizer& o, cusolverDnHandle_t& handleDn,
                      cublasHandle_t& handle,  float *d_cropped_face, float *d_buffer_face,
                      OptimizationVariables &ov, OptimizationVariables &ov_linesearch, Camera& cam, RotationComputer &rc,
                      RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s_lambda,  float *d_tmp,
                      bool visualize,
                      bool initialize_texture)
{
    ushort N_unique_pixels;

    cudaMemcpy(ov_linesearch.lambdas, ov.lambdas, ov.T*3*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ov_linesearch.Lintensity, ov.Lintensity, ov.T*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ov_linesearch.betas, ov.betas, ov.Ktotal*sizeof(float), cudaMemcpyDeviceToDevice);

    ov.set_frame(t);
    ov_linesearch.set_frame(t);

    rc.set_u_ptr(ov.u);
    rc_linesearch.set_u_ptr(ov_linesearch.u);

    if (initialize_texture)
    {
        rc.process();
        // <!-- We'll probably need to change the below --> use the right camera (e.g., cams[t])
        r.compute_nonrigid_shape2(handle, ov, rc.R, cam);
        r.compute_texture(handle, ov, o);

        r.render(t, o, ov, rc.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, false);
    }

    bool terminate = false;
    for (int i=0; i<10000; ++i)
    {
        if (terminate)
            break;

        r.render_for_illumination_only(t, o, ov, rc.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, visualize);

        if (i == 0)
            dc.compute_hessian_and_gradient_for_lambda(o, s_lambda, rc, N_unique_pixels, handleDn, handle, true);
        else
            dc.compute_hessian_and_gradient_for_lambda(o, s_lambda, rc, N_unique_pixels, handleDn, handle, false);

        float t_coef = 1.0f;
        float ALPHA = 0.5f;
        float BETA = 0.5f;

        const uint MAX_INNER_ITERS = 1000;
        uint inner_iter = 0;

        float obj[1], obj_tmp[1];
        cudaMemcpy(obj, ov.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);
        obj[0] = -obj[0];


        float lambda2[1];
        while (inner_iter < MAX_INNER_ITERS)
        {
            if (t_coef < 0.01f) {
                terminate = true;
                break;
            }

            set_xtmp<<<1, 3>>>(s_lambda.search_dir, ov.lambda, t_coef, ov_linesearch.lambda, 3);

            r.render_for_illumination_only(t, o, ov_linesearch, rc.R, handle,  &N_unique_pixels, d_cropped_face, d_buffer_face,   false);

            cudaMemcpy(obj_tmp, ov_linesearch.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);
            obj_tmp[0] = -obj_tmp[0];

            inner_iter++;
            cublasSdot(handle, 3, s_lambda.search_dir, 1, o.dG_dtheta_lambda, 1, d_tmp);

            cudaMemcpy(lambda2, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
            if (-lambda2[0] < 0.0003) {
                terminate = true;
                break;
            }
            lambda2[0] *= ALPHA*t_coef;


            if (obj_tmp[0]  < obj[0] + lambda2[0]) {
                cudaMemcpy(ov.lambda, ov_linesearch.lambda, sizeof(float)*3, cudaMemcpyDeviceToDevice);

                //                std::cout << '#' << i << ": "<< t_coef << '\t' <<"obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
                break;
            } else {
                //                std::cout << "\t\t" <<"obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
            }

            t_coef = t_coef * BETA;
        }
    }
}










void fit_3DMM_Lintensity(uint t, Renderer& r, Optimizer& o,
                         cublasHandle_t& handle,  float *d_cropped_face, float *d_buffer_face,
                         OptimizationVariables &ov, OptimizationVariables &ov_linesearch, Camera& cam, RotationComputer &rc,
                         RotationComputer& rc_linesearch, DerivativeComputer &dc, float *search_dir_Lintensity, float *dg,
                         float *d_tmp,
                         bool visualize,
                         bool initialize_texture)
{
    ov.set_frame(t);
    ov_linesearch.set_frame(t);

    rc.set_u_ptr(ov.u);
    rc_linesearch.set_u_ptr(ov_linesearch.u);


    ushort N_unique_pixels;

    cudaMemcpy(ov_linesearch.lambdas, ov.lambdas, ov.T*3*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ov_linesearch.Lintensity, ov.Lintensity, ov.T*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ov_linesearch.betas, ov.betas, ov.Ktotal*sizeof(float), cudaMemcpyDeviceToDevice);


    if (initialize_texture)
    {
        rc.process();
        // <!-- We'll probably need to change the below --> use the right camera (e.g., cams[t])
        r.compute_nonrigid_shape2(handle, ov, rc.R, cam);
        r.compute_texture(handle, ov, o);

        r.render(t, o, ov, rc.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face,  false);
    }

    bool terminate = false;
    for (int i=0; i<10000; ++i)
    {
        if (terminate)
            break;

        r.render_for_illumination_only(t, o, ov, rc.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, visualize);

        if (initialize_texture && i == 0)
            dc.compute_hessian_and_gradient_for_Lintensity(o,  N_unique_pixels, handle, search_dir_Lintensity, dg, true);
        else
            dc.compute_hessian_and_gradient_for_Lintensity(o, N_unique_pixels, handle, search_dir_Lintensity, dg, false);

        float t_coef = 1.0f;
        float ALPHA = 0.3f;
        float BETA = 0.5f;

        const uint MAX_INNER_ITERS = 1000;
        uint inner_iter = 0;

        float obj[1], obj_tmp[1];
        cudaMemcpy(obj, ov.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);
        obj[0] = -obj[0];

        float lambda2[1];
        while (inner_iter < MAX_INNER_ITERS)
        {
            if (t_coef < 0.1f) {
                terminate = true;
                break;
            }

            set_xtmp<<<1,1>>>(search_dir_Lintensity, ov.Lintensity, t_coef, ov_linesearch.Lintensity, 1);

            r.render_for_illumination_only(t, o, ov_linesearch, rc.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face,  false);



            cudaMemcpy(obj_tmp, ov_linesearch.grad_corr, sizeof(float), cudaMemcpyDeviceToHost);
            obj_tmp[0] = -obj_tmp[0];

            inner_iter++;
            cublasSdot(handle, 1, search_dir_Lintensity, 1, dg, 1, d_tmp);

            cudaMemcpy(lambda2, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);
            if (-lambda2[0] < 0.0025) {
                terminate = true;
                break;
            }
            lambda2[0] *= ALPHA*t_coef;

            if (obj_tmp[0]  < obj[0] + lambda2[0]) {
                cudaMemcpy(ov.Lintensity, ov_linesearch.Lintensity, sizeof(float), cudaMemcpyDeviceToDevice);

                //                std::cout << '#' << i << ": "<< t_coef << '\t' <<"obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
                break;
            } else {
                //                std::cout << "\t\t" <<"obj vs obj_tmp \t" << obj[0] << '\t' << obj_tmp[0] << std::endl;
            }

            t_coef = t_coef * BETA;
        }
    }

}
























bool fit_to_multi_images_landmarks_only(const std::vector<std::vector<float> >& all_xps, const std::vector<std::vector<float> >& all_yps,
        const std::vector<std::vector<float> >& all_xranges, const std::vector<std::vector<float> >& all_yranges,
        Renderer& r, cusolverDnHandle_t& handleDn, cublasHandle_t& handle,
        RotationComputer &rc, RotationComputer& rc_linesearch,
        std::vector<Camera> &cams)
{

    uint T = all_xps.size();
    OptimizationVariables ov_lb(T, r.Kalpha, r.Kbeta, r.Kepsilon, r.use_identity, r.use_texture, r.use_expression, true);
    OptimizationVariables ov_lb_linesearch(T, r.Kalpha, r.Kbeta, r.Kepsilon, r.use_identity, r.use_texture, r.use_expression, true);

    Logbarrier_Initializer li_init(&cams, &ov_lb, handleDn, 1.0f, r.use_identity, r.use_texture, r.use_expression, r, true);

    ov_lb.reset_tau_logbarrier();
    ov_lb_linesearch.reset_tau_logbarrier();


    // Pre-initialize -- initialize each frame
    for (size_t t=0; t<ov_lb.T; ++t)
    {
        //	 xp, yp contain the face mesh points mapped onto 2d according by a perspective
        //	 transformation. Hence each is an array of size NPTS.

        float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
        for (int i=0; i<NLANDMARKS_51; ++i)
        {
            xp[i] = cams[t].resize_coef*(all_xps[t][i] /*+ 1*/);
            yp[i] = cams[t].resize_coef*(all_yps[t][i] /*+ 1*/);
        }

        ////////////////////r.set_x0_short_y0_short(t, xp, yp);
        ///
        float face_size = compute_face_size(xp, yp);

        li_init.set_landmarks_from_host(t, xp, yp);
        li_init.initialize_with_orthographic_t(handleDn, handle, t, xp, yp, face_size, &ov_lb, all_xranges[t], all_yranges[t]);
    }

    li_init.set_minimal_slack(handle, &ov_lb);
    li_init.fit_model(handleDn, handle, &ov_lb, &ov_lb_linesearch);

    return li_init.fit_success;
}




bool fit_to_multi_images(std::vector<Camera> &cams,
                         const std::vector<std::vector<float> >& all_xps, const std::vector<std::vector<float> >& all_yps,
                         const std::vector<std::vector<float> >& all_xranges, const std::vector<std::vector<float> >& all_yranges,
                         const std::vector<cv::Mat>& frames,
                         std::vector<std::string>* result_basepaths,
                         Renderer& r, Optimizer& o,
                         cusolverDnHandle_t& handleDn,
                         cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
                         OptimizationVariables &ov, OptimizationVariables &ov_linesearch,
                         OptimizationVariables &ov_lb, OptimizationVariables &ov_lb_linesearch,
                         RotationComputer &rc,
                         RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, Solver &s_lambda, float *d_tmp,
                         float *search_dir_Lintensity, float *dg_Lintensity,
                         float *h_X0, float *h_Y0, float *h_Z0, float *h_tex_mu)
{
    cudaMemset(d_cropped_face, 0, sizeof(float)*DIMX*DIMY*ov.T);
    cudaMemset(d_buffer_face, 0, sizeof(float)*DIMX*DIMY);

    cudaMemset(ov_lb.betas, 0, ov_lb.Ktotal*sizeof(float));
    cudaMemset(ov_lb_linesearch.betas, 0, ov_lb_linesearch.Ktotal*sizeof(float));

    cudaMemset(ov.betas, 0, ov.Ktotal*sizeof(float));
    cudaMemset(ov_linesearch.betas, 0, ov_linesearch.Ktotal*sizeof(float));

    cudaMemset(ov_lb.slack, 0, sizeof(float));
    cudaMemset(ov_lb_linesearch.slack, 0, sizeof(float));

    //	Camera cam( (orig_width/2.0f)/0.5f,  (orig_height/2.0f)/0.5f, (orig_width)/2.0f, (orig_height)/2.0f);
    //	Camera cam(468.0640f, 468.4049f, (orig_width)/2.0f, (orig_height)/2.0f);


    Logbarrier_Initializer li_init(&cams, &ov_lb, handleDn, 1.0f, r.use_identity, r.use_texture, r.use_expression, r, true, config::CONFIDENCE_RANGE, config::USE_TEMP_SMOOTHING);
    Logbarrier_Initializer li(&cams, &ov, handleDn, 1.0f, r.use_identity, r.use_texture, r.use_expression, r, false, config::CONFIDENCE_RANGE, config::USE_TEMP_SMOOTHING);

    ov_lb.reset_tau_logbarrier();
    ov_lb_linesearch.reset_tau_logbarrier();
    ov.reset_tau_logbarrier();
    ov_linesearch.reset_tau_logbarrier();

    using std::vector;
    using std::string;

    // <!-- We'll probably need to change the below -->
    // Pre-initialize -- initialize each frame
    for (size_t t=0; t<ov.T; ++t)
    {
        //	 xp, yp contain the face mesh points mapped onto 2d according by a perspective
        //	 transformation. Hence each is an array of size NPTS.

        float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
        for (int i=0; i<NLANDMARKS_51; ++i)
        {
            xp[i] = cams[t].resize_coef*(all_xps[t][i] /*+ 1*/);
            yp[i] = cams[t].resize_coef*(all_yps[t][i] /*+ 1*/);
        }

        // <!-- We'll probably need to change the below --> if we do the right thing, face_size must be always the same for all frames
        float face_size = compute_face_size(xp, yp);

        r.set_x0_short_y0_short(t, xp, yp);
        li.set_landmarks_from_host(t, xp, yp);
        li_init.set_landmarks_from_host(t, xp, yp);
        li_init.initialize_with_orthographic_t(handleDn, handle, t, xp, yp, face_size, &ov_lb, all_xranges[t], all_yranges[t]);
    }

    li_init.set_minimal_slack(handle, &ov_lb);
    li_init.fit_model(handleDn, handle, &ov_lb, &ov_lb_linesearch);
    li.copy_from_initialized(li_init);

    if (!li_init.fit_success)
        return false;

    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);

    HANDLE_ERROR( cudaMemcpy( ov.taux, ov_lb.taux, sizeof(float)*6*ov.T, cudaMemcpyDeviceToDevice ) );
    HANDLE_ERROR( cudaMemcpy( ov.alphas, ov_lb.alphas, sizeof(float)*ov_lb.Kalpha, cudaMemcpyDeviceToDevice ) );

    for (uint t=0; t<ov.T; ++t) {
        HANDLE_ERROR( cudaMemcpy( ov.epsilons+t*ov.Kepsilon, ov_lb.epsilons+t*ov_lb.Kepsilon, sizeof(float)*ov_lb.Kepsilon, cudaMemcpyDeviceToDevice ) );
    }

    li.fit_model(handleDn, handle, &ov, &ov_linesearch);
    float h_lambdas[3] = {-7.3627f, 51.1364f, 100.1784f};
    float h_Lintensity = 0.005f;


    for (size_t t=0; t<ov.T; ++t)
    {
        // <!-- We'll probably need to change the below -->
        // Update camera at each frame

        ov.set_frame(t);
        ov_linesearch.set_frame(t);
        cv::Mat inputImage;

        cv::cvtColor(frames[t], inputImage, cv::COLOR_BGR2GRAY);

        if (config::PAINT_INNERMOUTH_BLACK)
            paint_innermouth_black(inputImage, all_xps[t], all_yps[t]);

        inputImage.convertTo(inputImage, CV_32FC1);
        inputImage = inputImage/255.0f;


        // <-- Critical part -->
        cv::resize(inputImage, inputImage, cv::Size(), cams[t].resize_coef, cams[t].resize_coef);
        cv::copyMakeBorder(inputImage, inputImage, 0, DIMY, 0, DIMX, cv::BORDER_CONSTANT, 0);

        cv::Mat cropped_face_upright = inputImage(cv::Rect(r.x0_short[t], r.y0_short[t], DIMX, DIMY)).clone();

        cv::Mat cropped_face_mat = cv::Mat(cropped_face_upright.t()).clone();
        HANDLE_ERROR( cudaMemcpy( d_cropped_face+t*(DIMX*DIMY), cropped_face_mat.data, sizeof(float)*DIMX*DIMY, cudaMemcpyHostToDevice ) );

        convolutionRowsGPU( d_buffer_face, d_cropped_face+t*(DIMX*DIMY), DIMX, DIMY );
        convolutionColumnsGPU(d_cropped_face+t*(DIMX*DIMY), d_buffer_face, DIMX, DIMY );

        cudaMemcpy(ov.lambda, h_lambdas, sizeof(float)*3, cudaMemcpyHostToDevice);
        cudaMemcpy(ov.Lintensity, &h_Lintensity, sizeof(float), cudaMemcpyHostToDevice);

        fit_3DMM_Lintensity(t, r, o, handle, d_cropped_face, d_buffer_face,
                            ov, ov_linesearch, cams[t], rc, rc_linesearch, dc, search_dir_Lintensity,
                            dg_Lintensity, d_tmp, false, true);

        fit_3DMM_lambdas(t, r, o, handleDn, handle, d_cropped_face, d_buffer_face,
                         ov, ov_linesearch, cams[t], rc, rc_linesearch, dc, s_lambda,  d_tmp, false, true);

        fit_3DMM_Lintensity(t, r, o,  handle, d_cropped_face, d_buffer_face,
                            ov, ov_linesearch, cams[t], rc, rc_linesearch, dc, search_dir_Lintensity,
                            dg_Lintensity, d_tmp, false, true);

        fit_3DMM_lambdas(t, r, o, handleDn, handle, d_cropped_face, d_buffer_face,
                         ov, ov_linesearch, cams[t], rc, rc_linesearch, dc, s_lambda,  d_tmp, false, true);
    }

    cudaMemcpy(ov_linesearch.lambdas, ov.lambdas, 3*ov.T*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ov_linesearch.Lintensities, ov.Lintensities, ov.T*sizeof(float), cudaMemcpyDeviceToDevice);

    //!std::cout << "NOW FITTING THE SHAPE " << std::endl;
    //!std::cout << "NOW FITTING THE SHAPE " << std::endl;

    ov.set_frame(0);
    ov_linesearch.set_frame(0);

    bool rigid_success = fit_3DMM_shape_rigid(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);

    ov.set_frame(0);
    rc.set_u_ptr(ov.u);
    rc.process();

    r.compute_texture(handle, ov, o);

    r.compute_nonrigid_shape_identityonly(handle, ov);

    cudaMemcpy(h_X0, r.X0, NPTS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y0, r.Y0, NPTS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z0, r.Z0, NPTS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tex_mu, o.d_tex, NPTS*sizeof(float), cudaMemcpyDeviceToHost);

    // <!-- We'll probably need to change the below --> use the right camera (e.g., cams[t])
    r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);


    if (result_basepaths != NULL)
    {
        //for (uint time_t=0; time_t<ov.T; ++time_t) {
        for (uint time_t=0; time_t<1; ++time_t) {
            ushort N_unique_pixels;
            ov.set_frame(time_t);


#ifdef WRITE_VARS_TO_DISK
            uint K_per_frame = ov.Kalpha+ov.Kbeta+ov.Kepsilon+6+4;
            float h_variables[K_per_frame];

            cudaMemcpy(h_variables, ov.alphas, ov.Kalpha*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_variables+ov.Kalpha, ov.betas, ov.Kbeta*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_variables+ov.Kalpha+ov.Kbeta, ov.taux, 6*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_variables+ov.Kalpha+ov.Kbeta+6, ov.epsilons, ov.Kepsilon*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_variables+ov.Kalpha+ov.Kbeta+6+ov.Kepsilon, ov.lambda, 3*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_variables+ov.Kalpha+ov.Kbeta+6+ov.Kepsilon+3, ov.Lintensity, 1*sizeof(float), cudaMemcpyDeviceToHost);

            writeArrFile<float>(h_variables, (*result_basepaths)[time_t]+".vars", 1, K_per_frame);
#endif

            rc_linesearch.set_u_ptr(ov.u);
            rc_linesearch.process_without_derivatives();

            // <!-- We'll probably need to change the below --> use the right camera (e.g., cams[t])
            r.compute_nonrigid_shape2(handle, ov, rc_linesearch.R, cams[0]);
            r.compute_texture(handle, ov, o);

            r.render(time_t, o, ov, rc_linesearch.R, handle, &N_unique_pixels, d_cropped_face, d_buffer_face, false);

            cv::Mat resized;
            cv::resize(frames[time_t], resized, cv::Size(), cams[time_t].resize_coef, cams[time_t].resize_coef);
            cv::copyMakeBorder(resized, resized, 0, DIMY, 0, DIMX, cv::BORDER_CONSTANT, 0);

            cv::Mat cropped_face_upright = resized(cv::Rect(r.x0_short[time_t], r.y0_short[time_t], DIMX, DIMY)).clone();

            // Note that we transpose the image here to make it column major as the rest
            cv::Mat cropped_face_mat = cv::Mat(cropped_face_upright.t()).clone();
            cv::cvtColor(cropped_face_mat, cropped_face_mat, cv::COLOR_BGR2GRAY);
            cropped_face_mat.convertTo(cropped_face_mat, CV_32FC1);
            cropped_face_mat = cropped_face_mat/255.0f;

            HANDLE_ERROR( cudaMemcpy( d_cropped_face, cropped_face_mat.data, sizeof(float)*DIMX*DIMY, cudaMemcpyHostToDevice ) );

#ifdef WRITE_OBJ_FILE
            imwrite_opencv(d_cropped_face, (*result_basepaths)[time_t]+"ori.png");
            imwrite_opencv(r.d_texIm, (*result_basepaths)[time_t]+"rec.png");

            r.compute_nonrigid_shape_identityonly(handle, ov);
            r.print_mat_txt((*result_basepaths)[time_t]+".mat_txt");
            r.print_obj_neutral((*result_basepaths)[time_t]+".obj");
#endif
        }
    }

    return rigid_success;
}















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
                                       float *search_dir_Lintensity, float *dg_Lintensity, std::vector<Camera> & cams, bool fit_landmarks_only)
{

    std::string obj_path(result_basepath+".obj");

    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ov.reset();
    ov_linesearch.reset();
    ov_lb.reset();
    ov_lb_linesearch.reset();

    int Kper_frame = 3 + 3 + ov.Kepsilon*r.use_expression;
    uint Ktotal__ = Kper_frame*ov.T + ov.Kalpha + ov.Kbeta;

    cudaMemset(ov.XALL, 0.0f, Ktotal__*sizeof(float));
    cudaMemset(o.d_TEX_ID_NREF, 0, sizeof(float)*Nrender_estimated*3850*ov.T);

    int Nksx = Nrender_estimated*19+NTOTAL_PIXELS*2;

    reset_ushort_array<<<(Nksx + NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_KSX, Nksx);

    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////




    cudaMemset(d_cropped_face, 0, sizeof(float)*DIMX*DIMY);
    cudaMemset(d_buffer_face, 0, sizeof(float)*DIMX*DIMY);

    //	std::cout << landmarks_path << std::endl;

    //	 xp, yp contain the face mesh points mapped onto 2d according by a perspective
    //	 transformation. Hence each is an array of size NPTS.

    float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
    for (int i=0; i<NLANDMARKS_51; ++i)
    {
        yp[i] = cams[0].resize_coef*(yp_landmark[i] /*+ 1*/);
        xp[i] = cams[0].resize_coef*(xp_landmark[i] /*+ 1*/);
    }

    float face_size = compute_face_size(xp, yp);

    r.set_x0_short_y0_short(0, xp, yp);

    cv::Mat inputImage = cv::imread(im_path, cv::IMREAD_GRAYSCALE);

    if (cams[0].cam_remap) {
        cv::Mat imageDistorted = inputImage.clone();
        cams[0].undistort(imageDistorted, inputImage);

        //cv::imshow("a", inputImage);
        //cv::waitKey(0);
    }

    cv::copyMakeBorder(inputImage, inputImage, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, cv::BORDER_CONSTANT, 0);

    inputImage.convertTo(inputImage, CV_32FC1);
    inputImage = inputImage/255.0f;

    cv::resize(inputImage, inputImage, cv::Size(), cams[0].resize_coef, cams[0].resize_coef);

    float orig_width = (float) inputImage.cols;
    float orig_height = (float) inputImage.rows;

    if (orig_width < DIMX*3)
        cv::copyMakeBorder(inputImage, inputImage, 0, DIMY, 0, DIMX, cv::BORDER_CONSTANT, 0);

    cudaMemset(ov_lb.slack, 0, sizeof(float));
    cudaMemset(ov_lb_linesearch.slack, 0, sizeof(float));

    //    Camera cam( (orig_width/2.0f)/tan(DEG2RAD(fovx/2.0f)),  (orig_height/2.0f)/tan(DEG2RAD(fovy/2.0f)), orig_width/2.0f, orig_height/2.0f);
    //    cam.update_camera(RESIZE_COEF);

    float CONFIDENCE_RANGE = config::CONFIDENCE_RANGE;
    Logbarrier_Initializer li_init(&cams, &ov_lb, handleDn, 1.0f, r.use_identity, r.use_texture, r.use_expression, r, true, CONFIDENCE_RANGE);
    Logbarrier_Initializer li(&cams, &ov, handleDn, 1.0f, r.use_identity, r.use_texture, r.use_expression, r, false, CONFIDENCE_RANGE);

    ov_lb.reset_tau_logbarrier();
    ov_lb_linesearch.reset_tau_logbarrier();
    ov.reset_tau_logbarrier();
    ov_linesearch.reset_tau_logbarrier();

    li.set_landmarks_from_host(0, xp, yp);
    li_init.set_landmarks_from_host(0, xp, yp);
    li_init.initialize_with_orthographic_t(handleDn, handle, 0, xp, yp, face_size, &ov_lb, xrange, yrange);
    li_init.set_minimal_slack(handle, &ov_lb);

    li_init.fit_model(handleDn, handle, &ov_lb, &ov_lb_linesearch);
    li.copy_from_initialized(li_init);

    /*
    for (uint i=0; i<51; ++i)
    {
        cv::Point2f ptOrig(xp[i], yp[i]);
        circle(inputImage, ptOrig, 1, cv::Scalar(255,255,255), cv::FILLED, 8, 0);
//        cv::rectangle(inputImage, cv::Rect(ptOrig.x+h_bounds_lx[i]*bbox_size,  ptOrig.y+h_bounds_ly[i]*bbox_size,
//                                           (h_bounds_ux[i]-h_bounds_lx[i])*bbox_size, (h_bounds_uy[i]-h_bounds_ly[i])*bbox_size), cv::Scalar::all(255));
    }

    cv::imshow("in_model_fitter", inputImage);
    cv::waitKey(0);*/

    HANDLE_ERROR( cudaMemcpy( ov.taux, ov_lb.taux, sizeof(float)*6*ov.T, cudaMemcpyDeviceToDevice ) );
    HANDLE_ERROR( cudaMemcpy( ov.alphas, ov_lb.alphas, sizeof(float)*ov_lb.Kalpha, cudaMemcpyDeviceToDevice ) );

    for (uint t=0; t<ov.T; ++t) {
        HANDLE_ERROR( cudaMemcpy( ov.epsilons+t*ov.Kepsilon, ov_lb.epsilons+t*ov_lb.Kepsilon, sizeof(float)*ov_lb.Kepsilon, cudaMemcpyDeviceToDevice ) );
    }

    cv::Mat cropped_face_upright = inputImage(cv::Rect(r.x0_short[0], r.y0_short[0], DIMX, DIMY)).clone();

    // Note that we transpose the image here to make it column major as the rest
    cv::Mat cropped_face_mat = cv::Mat(cropped_face_upright.t()).clone();

    HANDLE_ERROR( cudaMemcpy( d_cropped_face, cropped_face_mat.data, sizeof(float)*DIMX*DIMY, cudaMemcpyHostToDevice ) );



    convolutionRowsGPU( d_buffer_face, d_cropped_face, DIMX, DIMY );
    convolutionColumnsGPU(d_cropped_face, d_buffer_face, DIMX, DIMY );




    ushort N_unique_pixels;


    float h_lambdas[3] = {-7.3627f, 51.1364f, 100.1784f};
    float h_Lintensity = 0.005f;
    //	float h_Lintensity = 0.0018f;

    if (fit_landmarks_only) {
        return li_init.fit_success;
    }

    if (li_init.fit_success)
    {
        float yaw, pitch, roll;

        li_init.rc.compute_euler_angles(yaw, pitch, roll);

        std::cout << "\t (LB)yaw: " << RAD2DEG(yaw) << '\t' << " pitch: " << RAD2DEG(pitch) <<'\t' << " roll: " << RAD2DEG(roll) << std::endl;
        /**/

        cudaMemcpy(ov.lambda, h_lambdas, sizeof(float)*3, cudaMemcpyHostToDevice);
        cudaMemcpy(ov.Lintensity, &h_Lintensity, sizeof(float), cudaMemcpyHostToDevice);

        fit_3DMM_Lintensity(0, r, o, handle, d_cropped_face, d_buffer_face,
                            ov, ov_linesearch, cams[0], rc, rc_linesearch, dc, search_dir_Lintensity, dg_Lintensity, d_tmp, false, true);

        fit_3DMM_lambdas(0, r, o, handleDn, handle, d_cropped_face, d_buffer_face,
                         ov, ov_linesearch, cams[0], rc, rc_linesearch, dc, s_lambda,  d_tmp, false, true);

        cudaMemcpy(ov_linesearch.lambdas, ov.lambdas, ov.T*3*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ov_linesearch.Lintensity, ov.Lintensity, ov.T*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ov_linesearch.betas, ov.betas, ov.Ktotal*sizeof(float), cudaMemcpyDeviceToDevice);


        fit_3DMM_shape_rigid(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face,
                             ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);


#ifdef WRITE_OBJ_FILE
        imwrite_opencv(d_cropped_face, result_basepath+"ori.png");
        imwrite_opencv(r.d_texIm, result_basepath+"rec.png");
        r.print_obj(obj_path);
#endif

#ifdef WRITE_SPARSE_LANDMARKS
        r.print_sparse_2Dpts(result_basepath+".pts", 1.0f/cams[0].resize_coef);
#endif
        //r.print_mat_txt(result_basepath+".mat_txt");

#ifdef WRITE_VARS_TO_DISK
        uint K_per_frame = ov.Kalpha+ov.Kbeta+ov.Kepsilon+6+4;
        float h_variables[K_per_frame];

        cudaMemcpy(h_variables, ov.alphas, ov.Kalpha*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_variables+ov.Kalpha, ov.betas, ov.Kbeta*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_variables+ov.Kalpha+ov.Kbeta, ov.taux, 6*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_variables+ov.Kalpha+ov.Kbeta+6, ov.epsilons, ov.Kepsilon*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_variables+ov.Kalpha+ov.Kbeta+6+ov.Kepsilon, ov.lambda, 3*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_variables+ov.Kalpha+ov.Kbeta+6+ov.Kepsilon+3, ov.Lintensity, 1*sizeof(float), cudaMemcpyDeviceToHost);

        writeArrFile<float>(h_variables, (result_basepath)+".vars", 1, K_per_frame);
#endif

        return true;
    }

    return false;
}













bool update_shape_single_resize_coef(float *xp_orig, float *yp_orig, std::vector<float>& xrange, std::vector<float>& yrange,
                        const cv::Mat& frame, std::vector<Camera> &cams, Renderer& r, Optimizer& o, cusolverDnHandle_t& handleDn,
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
                                     std::vector<float>& exp_coeffs_combined,
                                     float *Xtmp, float *Ytmp, float *Ztmp,
                                     float* h_X0_cur, float* h_Y0_cur, float* h_Z0_cur,
                                     float* h_Xr_cur, float* h_Yr_cur, float* h_Zr_cur,
                                     vector<float>& X0cur_vec, vector<float>& Y0cur_vec, vector<float>& Z0cur_vec,
                                     vector<float>& Xrcur_vec, vector<float>& Yrcur_vec, vector<float>& Zrcur_vec)
{

    float exp_coeffs[config::K_EPSILON];

//    config::set_resize_coef(cur_resize_coef);
    cams[0].update_camera(cur_resize_coef);

    bool success = fit_to_video_frame(xp_orig, yp_orig, xrange, yrange, frame, cams, r, o, handleDn,  handle,
                                 d_cropped_face, d_buffer_face, li, li_init, ov, ov_linesearch,
                                 ov_lb, ov_lb_linesearch, rc, rc_linesearch, dc, s, s_lambda, d_tmp,
                                 search_dir_Lintensity, dg_Lintensity, frame_id, outputVideo,
                                 outputVideoPath,  ref_face_size, -1.0f,
                                 min_x, max_x, min_y, max_y, FPS);

    r.compute_nonrigid_shape_identity_and_rotation(handle, ov, rc.R, Xtmp, Ytmp, Ztmp);
    cudaMemcpy(exp_coeffs, ov.epsilons, sizeof(float)*config::K_EPSILON, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_X0_cur, r.X0, sizeof(float)*NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y0_cur, r.Y0, sizeof(float)*NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z0_cur, r.Z0, sizeof(float)*NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Xr_cur, Xtmp, sizeof(float)*NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Yr_cur, Ytmp, sizeof(float)*NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Zr_cur, Ztmp, sizeof(float)*NPTS, cudaMemcpyDeviceToHost);

    float denom = (float) config::NRES_COEFS;
    for (size_t pi=0; pi<NPTS; ++pi)
    {
        X0cur_vec[pi] += h_X0_cur[pi]/denom;
        Y0cur_vec[pi] += h_Y0_cur[pi]/denom;
        Z0cur_vec[pi] += h_Z0_cur[pi]/denom;
        Xrcur_vec[pi] += h_Xr_cur[pi]/denom;
        Yrcur_vec[pi] += h_Yr_cur[pi]/denom;
        Zrcur_vec[pi] += h_Zr_cur[pi]/denom;
    }

    for (size_t i=0; i<config::K_EPSILON; ++i)
        exp_coeffs_combined.push_back(exp_coeffs[i]);

    return success;

}











bool fit_to_video_frame(float *xp_orig, float *yp_orig, std::vector<float>& xrange, std::vector<float>& yrange,
                        const cv::Mat& inputImage_color, std::vector<Camera> &cams, Renderer& r, Optimizer& o, cusolverDnHandle_t& handleDn,
                        cublasHandle_t& handle, float *d_cropped_face, float *d_buffer_face,
                        Logbarrier_Initializer &li, Logbarrier_Initializer &li_init,
                        OptimizationVariables &ov, OptimizationVariables &ov_linesearch,
                        OptimizationVariables &ov_lb, OptimizationVariables &ov_lb_linesearch, RotationComputer &rc,
                        RotationComputer& rc_linesearch, DerivativeComputer &dc, Solver &s, Solver &s_lambda, float *d_tmp,
                        float *search_dir_Lintensity, float *dg_Lintensity, const uint frame_id,  cv::VideoWriter *outputVideo,
                        const std::string& outputFilePath,
                        float& ref_face_size, float cur_face_size, int *min_x, int *max_x, int* min_y, int* max_y, int FPS)
{


    bool success = false;
    bool we_have_good_landmarks = xp_orig[0] != 1.0f;

    cudaEvent_t     start_tavaturi, stop_tavaturi;
    cudaEventCreate( &start_tavaturi );
    cudaEventCreate( &stop_tavaturi );
    cudaEventRecord( start_tavaturi, 0 );
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    //    ov.reset();
    //    ov_linesearch.reset();

    ov.reset_tau_logbarrier();
    ov_linesearch.reset_tau_logbarrier();

    ov_lb.reset();
    ov_lb_linesearch.reset();

    int Kper_frame = 3 + 3 + ov.Kepsilon*r.use_expression;
    uint Ktotal__ = Kper_frame*ov.T + ov.Kalpha + ov.Kbeta;

    //    cudaMemset(ov.XALL, 0.0f, Ktotal__*sizeof(float));
    cudaMemset(o.d_TEX_ID_NREF, 0, sizeof(float)*Nrender_estimated*3850*ov.T);

    ////////////////int Nksx = Nrender_estimated*19+NTOTAL_PIXELS*2;

    ////////////////reset_ushort_array<<<(Nksx + NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_KSX, Nksx);

    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////

    cudaMemset(d_cropped_face, 0, sizeof(float)*DIMX*DIMY);
    cudaMemset(d_buffer_face, 0, sizeof(float)*DIMX*DIMY);

    cv::Mat inputImage;
    cv::cvtColor(inputImage_color, inputImage, cv::COLOR_BGR2GRAY);

    if (config::PAINT_INNERMOUTH_BLACK) {
        std::vector<float> xp_vec(xp_orig, xp_orig+NLANDMARKS_51);
        std::vector<float> yp_vec(yp_orig, yp_orig+NLANDMARKS_51);
        paint_innermouth_black(inputImage, xp_vec, yp_vec);
    }



    inputImage.convertTo(inputImage, CV_32FC1);
    inputImage = inputImage/255.0f;

    int orig_width = (int) inputImage.cols;
    int orig_height = (int) inputImage.rows;


    /*
    */
    cv::resize(inputImage, inputImage, cv::Size(), cams[0].resize_coef, cams[0].resize_coef);

    float resized_width = (int) inputImage.cols;
    float resized_height = (int) inputImage.rows;

    cv::copyMakeBorder(inputImage, inputImage, 0, DIMY, 0, DIMX, cv::BORDER_CONSTANT, 0);

    float xp[NLANDMARKS_51], yp[NLANDMARKS_51];

    if (we_have_good_landmarks)
    {
        for (int i=0; i<NLANDMARKS_51; ++i)
        {
            xp[i] = cams[0].resize_coef*(xp_orig[i]);
            yp[i] = cams[0].resize_coef*(yp_orig[i]);
        }

        float face_size = compute_face_size(xp, yp);

        cudaMemset(ov_lb.slack, 0, sizeof(float));
        cudaMemset(ov_lb_linesearch.slack, 0, sizeof(float));

        ov_lb.reset_tau_logbarrier();
        ov_lb_linesearch.reset_tau_logbarrier();
        ov.reset_tau_logbarrier();
        ov_linesearch.reset_tau_logbarrier();

        // --->***<--- // PLACE COMPUTATION OF IOD HERE
        li.set_landmarks_from_host(0, xp, yp);
        li_init.set_landmarks_from_host(0, xp, yp);
        li_init.initialize_with_orthographic_t(handleDn, handle, 0, xp, yp, face_size, &ov_lb, xrange, yrange);
        li_init.set_minimal_slack(handle, &ov_lb);
    }

    bool is_bb_OK = check_if_bb_OK(xp, yp);

    /*
    if (!is_bb_OK)
    {
        std::cout << "Skipping this ..." << std::endl;
        return success;
    }
*/


    /**
      * @START COPY
      * - Copying initialized camera view, identity (shape) and expression parameters
      */
    li_init.fit_model(handleDn, handle, &ov_lb, &ov_lb_linesearch);
    if (li_init.fit_success && is_bb_OK && we_have_good_landmarks)
    {
        li.copy_from_initialized(li_init);

        HANDLE_ERROR( cudaMemcpy( ov.taux, ov_lb.taux, sizeof(float)*6*ov.T, cudaMemcpyDeviceToDevice ) );
        HANDLE_ERROR( cudaMemcpy( ov.alphas, ov_lb.alphas, sizeof(float)*ov_lb.Kalpha, cudaMemcpyDeviceToDevice ) );

        for (uint t=0; t<ov.T; ++t) {
            HANDLE_ERROR( cudaMemcpy( ov.epsilons+t*ov.Kepsilon, ov_lb.epsilons+t*ov_lb.Kepsilon, sizeof(float)*ov_lb.Kepsilon, cudaMemcpyDeviceToDevice ) );
        }
        //        print_vector(ov.taux, 6, "rigid_params");
        r.set_x0_short_y0_short(0, xp, yp);

    }
    else
    {
        li.compute_bounds(0, 1.0, 1.0, 1.0, xrange, yrange, true);
        //        print_vector(ov.taux, 6, "rigid_params");
        //        print_vector(ov.epsilons, K_EPSILON, "expr_params");
    }

    /**
     * @END COPY
     */
    cv::Mat cropped_face_upright = inputImage(cv::Rect(r.x0_short[0], r.y0_short[0], DIMX, DIMY)).clone();
    // Note that we transpose the image here to make it column major as the rest
    cv::Mat cropped_face_mat = cv::Mat(cropped_face_upright.t()).clone();

    HANDLE_ERROR( cudaMemcpy( d_cropped_face, cropped_face_mat.data, sizeof(float)*DIMX*DIMY, cudaMemcpyHostToDevice ) );

    convolutionRowsGPU( d_buffer_face, d_cropped_face, DIMX, DIMY );
    convolutionColumnsGPU(d_cropped_face, d_buffer_face, DIMX, DIMY );

    /*
    float h_lambdas[3] = {-7.3627f, 51.1364f, 100.1784f};
    float h_Lintensity = 0.005f;
     */

    ushort N_unique_pixels;

    cv::Mat scratchMat(resized_height, resized_width, CV_32F, cv::Scalar::all(0));

    const int buffer = 0; //100;
    int facerect_width = ((*max_x+buffer)-(*min_x-buffer));
    int facerect_height =((*max_y+buffer)-(*min_y-buffer));

    //    cv::Size vid_size = cv::Size((int) 2*resized_width, (int) orig_height);
    cv::Size vid_size = cv::Size((int) 2*facerect_width, (int) facerect_height);
    cv::Size vid_size2 = cv::Size((int) 800, (int) 800);

    cudaEventRecord( stop_tavaturi, 0 );
    cudaEventSynchronize( stop_tavaturi );
    float   elapsedTime_tavaturi;
    cudaEventElapsedTime( &elapsedTime_tavaturi, start_tavaturi, stop_tavaturi );
    //    printf( "Tavaturi %.2f ms\n", elapsedTime_tavaturi);

    if (!outputVideo->isOpened())
    {
//        std::cout << "WASNT OPENED -- > NOW OPENING" << std::endl;
        outputVideo->open(outputFilePath, cv::VideoWriter::fourcc('M','J','P','G'), FPS, vid_size, false);
    }


    //    if (li_init.fit_success && li_init.face_sizes[0] >= 36)
    if (true)
    {
        /*
        cudaMemcpy(ov.lambda, h_lambdas, sizeof(float)*3, cudaMemcpyHostToDevice);
        cudaMemcpy(ov.Lintensity, &h_Lintensity, sizeof(float), cudaMemcpyHostToDevice);
         */
        if (frame_id % 50 == 0)
        {
            fit_3DMM_Lintensity(0, r, o, handle, d_cropped_face, d_buffer_face,
                                ov, ov_linesearch, cams[0], rc, rc_linesearch, dc, search_dir_Lintensity, dg_Lintensity, d_tmp, false, true);

            fit_3DMM_lambdas(0, r, o, handleDn, handle, d_cropped_face, d_buffer_face,
                             ov, ov_linesearch, cams[0], rc, rc_linesearch, dc, s_lambda,  d_tmp, false, true);
        }

        cudaMemcpy(ov_linesearch.lambdas, ov.lambdas, ov.T*3*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ov_linesearch.Lintensity, ov.Lintensity, ov.T*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ov_linesearch.betas, ov.betas, ov.Ktotal*sizeof(float), cudaMemcpyDeviceToDevice);

        fit_3DMM_shape_rigid(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face,
                             ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);

        if (false)
        {
            imshow_opencv(r.d_texIm, "TIM");
            cv::waitKey(5);
        }

        success = true;


        if (true) // #TIME_MODIFICATIONS - disable video writing to measure time
        {
            cv::Mat result(facerect_height, 2*facerect_width, CV_8U, cv::Scalar::all(0));
            try {
                create_cvmat_buffered(r.d_texIm,  r.x0_short[0], r.y0_short[0], resized_width, resized_height, scratchMat);
                //                cv::imshow("EHEEBEF", scratchMat);

                cv::resize(scratchMat, scratchMat, cv::Size(orig_width, orig_height));

                //////////////////			imshow_opencv_buffered(r.d_texIm,  r.x0_short[0], r.y0_short[0], orig_width, orig_height, "TIMSS");
                ////imshow_opencv(r.d_texIm, "TIM");
                //			imshow_opencv(d_cropped_face+t*(DIMX*DIMY), "INPUT");
                //		cv::waitKey(0);
                //                cv::imshow("EHEEAFT", scratchMat);

                cv::Mat scratchMat_uchar;
                cv::Mat inputImage_uchar;

                std::stringstream ss;
                ss << li.angle_idx[0];


                float h_bounds_lx[51], h_bounds_ux[51], h_bounds_ly[51], h_bounds_uy[51];
                cudaMemcpy(h_bounds_lx, li.bounds_lx_cur, 51*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_bounds_ux, li.bounds_ux_cur, 51*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_bounds_ly, li.bounds_ly_cur, 51*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_bounds_uy, li.bounds_uy_cur, 51*sizeof(float), cudaMemcpyDeviceToHost);
                double fcr = ref_face_size/cams[0].resize_coef;
                /*
                for (int i=0; i<NLANDMARKS_51; ++i)
                {
                    cv::Point2f ptOrig(xp[i]/config::RESIZE_COEF, yp[i]/config::RESIZE_COEF);
                    //                    circle(inputImage_color, ptOrig, 3, cv::Scalar(255,255,255), cv::FILLED, 8, 0);

                    //        cv::rectangle(inputImage_color, cv::Rect(ptOrig.x+h_bounds_lx[i]*fcr,  ptOrig.y+h_bounds_ly[i]*fcr,
                    //                                           (h_bounds_ux[i]-h_bounds_lx[i])*fcr, (h_bounds_uy[i]-h_bounds_ly[i])*fcr), cv::Scalar::all(255));
                }
                */

                cv::cvtColor(inputImage_color, inputImage_uchar, cv::COLOR_BGR2GRAY);


                scratchMat = 255*scratchMat;
                scratchMat.convertTo(scratchMat_uchar, CV_8U);


                scratchMat_uchar = scratchMat_uchar(cv::Rect((*min_x-buffer), (*min_y-buffer), facerect_width, facerect_height));
                inputImage_uchar = inputImage_uchar(cv::Rect((*min_x-buffer), (*min_y-buffer), facerect_width, facerect_height));

                scratchMat_uchar.copyTo(result(cv::Rect(0, 0, facerect_width, facerect_height)));
                inputImage_uchar.copyTo(result(cv::Rect(facerect_width, 0, facerect_width, facerect_height)));

                cv::putText(result,ss.str(),cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(255,255,255),2,false);

                success = true;

                //                cv::imshow("MERGED", result);
                //                        cv::waitKey(1);

            } catch (cv::Exception e) {
                std::cout << "Face too close to boundaries, skipping ..." << std::endl;
            }
            *outputVideo << result;
        }

    }
    else
    {
        std::cout << "failed to fit!" << std::endl;
    }

    if (we_have_good_landmarks && is_bb_OK && li.fit_success)
    {
        if (cur_face_size > -1.0f)
            ref_face_size = cur_face_size;
    }

    return success;
}

































