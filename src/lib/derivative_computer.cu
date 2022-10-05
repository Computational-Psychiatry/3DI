/*
 * derivative_computer.cu
 *
 *  Created on: Oct 2, 2020
 *      Author: root
 */



#include "derivative_computer.h"



__global__ void compute_Lintensity_search_dir(const float *JTJ, const float *dg, float *search_dir)
{
    search_dir[0] = -dg[0]/JTJ[0];
}




void DerivativeComputer::compute_hessian_and_gradient(ushort t,
                                                      Optimizer& o,
                                                      RotationComputer& rc,
                                                      Renderer& r,
                                                      Camera& cam,
                                                      ushort &N_unique_pixels,
                                                      cublasHandle_t &handle,
                                                      Logbarrier_Initializer& li)
{

#ifdef MEASURE_TIME
    cudaEvent_t start_t, stop_t;
    cudaEventCreate( &start_t);
    cudaEventCreate( &stop_t);
    cudaEventRecord( start_t, 0 );
#endif

    cudaMemset(o.dI_dbeta, 0, sizeof(float)*o.ov_ptr->Ktotal*Nrender_estimated);

    ////////////////////////////////
    // This is cum-sum
    thrust::inclusive_scan(thrust::device, o.d_M0, o.d_M0 + NTOTAL_PIXELS, o.d_cumM0); // in-place scan


    thrust::sequence(thrust::device, o.d_ks_sortidx, o.d_ks_sortidx + N_unique_pixels);
    thrust::sort_by_key(thrust::device, o.d_ks_sorted, o.d_ks_sorted + N_unique_pixels, o.d_ks_sortidx);

    thrust::sequence(thrust::device, o.d_ks_sortidx_sortidx, o.d_ks_sortidx_sortidx + N_unique_pixels);
    HANDLE_ERROR( cudaMemcpy( o.d_ks_sortidx_copy, o.d_ks_sortidx,  sizeof(ushort)*Nrender_estimated, cudaMemcpyDeviceToDevice) );

    thrust::sort_by_key(thrust::device, o.d_ks_sortidx_copy, o.d_ks_sortidx_copy + N_unique_pixels, o.d_ks_sortidx_sortidx);

    // We'll put back at least the two following functions (fill_ksides and fill_krels)
    ///////////////////////fill_ksides<<<N_unique_pixels, 1>>>(o.d_ks_sorted,  o.d_ks_left, o.d_ks_right, o.d_ks_above, o.d_ks_below);
    fill_ksides<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_ks_sorted,  o.d_ks_left, o.d_ks_right, o.d_ks_above, o.d_ks_below);

    fill_krels<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                      N_unique_pixels, o.d_ks_sorted,
                                                                      o.d_ks_sortidx,
                                                                      o.d_ks_sortidx_sortidx,
                                                                      o.d_ks_left, 		o.d_ks_right, 		o.d_ks_above, 		o.d_ks_below,
                                                                      o.d_kl_rel, 		o.d_kr_rel, 		o.d_ka_rel, 		o.d_kb_rel,
                                                                      o.d_kl_rel_sorted, 	o.d_kr_rel_sorted, 	o.d_ka_rel_sorted, 	o.d_kb_rel_sorted,
                                                                      o.d_cumM0);

    fill_krels2<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                       o.d_ks_unsorted,
                                                                       o.d_ks_sortidx,
                                                                       o.d_ks_sortidx_sortidx,
                                                                       o.d_kl_rel, 		o.d_kr_rel, 		o.d_ka_rel, 		o.d_kb_rel,
                                                                       o.d_kl_rel_sorted, 	o.d_kr_rel_sorted, 	o.d_ka_rel_sorted, 	o.d_kb_rel_sorted);

    uint c_offset = Nrender_estimated*6*t;
    fill_optimization_auxiliary_variables_phase1<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                                                        cam.h_phix, cam.h_phiy,
                                                                                                        o.vx, o.vy, o.vz, o.inv_vz, o.inv_vz2,
                                                                                                        o.px, o.py, o.pz,
                                                                                                        rc.R, rc.dR_du1, rc.dR_du2, rc.dR_du3,
                                                                                                        o.gx, o.gy,
                                                                                                        o.dI_dtaux + c_offset, o.dI_dtauy + c_offset, o.dI_dtauz + c_offset,
                                                                                                        o.dI_du1 + c_offset,   o.dI_du2 + c_offset,   o.dI_du3 + c_offset,
                                                                                                        N_unique_pixels);

    if (use_identity)
    {
        fill_optimization_dI_dalpha<<<(N_unique_pixels*o.Kalpha+NTHREADS-1)/NTHREADS, NTHREADS>>>(cam.h_phix, cam.h_phiy,
                                                                                                  o.vx, o.vy, o.vz, o.inv_vz2,
                                                                                                  rc.R,
                                                                                                  o.gx, o.gy,
                                                                                                  r.d_RIX, r.d_RIY, r.d_RIZ,
                                                                                                  o.dI_dalpha,
                                                                                                  N_unique_pixels,
                                                                                                  o.Kalpha);
    }

    if (use_expression)
    {
        uint epsilon_offset = Nrender_estimated*o.Kepsilon*t;
        fill_optimization_dI_depsilon_userotated<<<(N_unique_pixels*o.Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(cam.h_phix, cam.h_phiy,
                                                                                                                 o.vx, o.vy, o.vz, o.inv_vz2,
                                                                                                                 rc.R,
                                                                                                                 o.gx, o.gy,
                                                                                                                 r.d_REX, r.d_REY, r.d_REZ,
                                                                                                                 o.dI_depsilons + epsilon_offset,
                                                                                                                 N_unique_pixels,
                                                                                                                 o.Kepsilon);
    }

    if (use_texture)
    {
        fill_optimization_dI_dbeta<<<(N_unique_pixels*o.Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                                                  r.d_RTEX,
                                                                                                  o.d_Id_,
                                                                                                  o.ov_ptr->Lintensity,
                                                                                                  o.dI_dbeta,
                                                                                                  N_unique_pixels,
                                                                                                  o.Kbeta);
    }

    fill_optimization_auxiliary_variables_phase2_new<<<(Nrender_estimated*o.ov_ptr->Ktotal+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                                                                               o.dI_dbeta,
                                                                                                                               o.gx, o.gy, o.h,
                                                                                                                               o.d_kl_rel, o.d_kr_rel, o.d_ka_rel, o.d_kb_rel,
                                                                                                                               o.dgx_dtheta,
                                                                                                                               o.dgy_dtheta,
                                                                                                                               N_unique_pixels,
                                                                                                                               o.ov_ptr->Ktotal);

    float h_log_barrier;
    float plus_one_ = 1.0f;


    if (t == 0) {
        cudaMemset(o.JTJ, 0, o.ov_ptr->Ktotal*o.ov_ptr->Ktotal*sizeof(float));
        cudaMemset(o.dG_dtheta, 0, o.ov_ptr->Ktotal*sizeof(float));
    }


#ifdef MEASURE_TIME
    cudaEventRecord( stop_t, 0 );
    cudaEventSynchronize( stop_t );

    float   elapsedTime_t;
    cudaEventElapsedTime( &elapsedTime_t, start_t, stop_t );
    printf( "TM_OtherThanHTH = %.2f ms\n", elapsedTime_t);
#endif



#ifdef MEASURE_TIME
    cudaEventCreate( &start_t );
    cudaEventCreate( &stop_t );
    cudaEventRecord( start_t, 0 );
#endif
    cudaMemcpy(&h_log_barrier, o.ov_ptr->tau_logbarrier, sizeof(float), cudaMemcpyDeviceToHost);

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, o.ov_ptr->Ktotal, o.ov_ptr->Ktotal, Nrender_estimated, &h_log_barrier, o.dgx_dtheta, Nrender_estimated, o.dgx_dtheta, Nrender_estimated, &plus_one_, o.JTJ, o.ov_ptr->Ktotal);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, o.ov_ptr->Ktotal, o.ov_ptr->Ktotal, Nrender_estimated, &h_log_barrier, o.dgy_dtheta, Nrender_estimated, o.dgy_dtheta, Nrender_estimated, &plus_one_, o.JTJ, o.ov_ptr->Ktotal);

    h_log_barrier *= -1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, o.ov_ptr->Ktotal, Nrender_estimated, &h_log_barrier, o.gxs_norm, 1, o.dgx_dtheta, Nrender_estimated, &plus_one_, o.dG_dtheta, 1);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, o.ov_ptr->Ktotal, Nrender_estimated, &h_log_barrier, o.gys_norm, 1, o.dgy_dtheta, Nrender_estimated, &plus_one_, o.dG_dtheta, 1);

#ifdef MEASURE_TIME
    cudaEventRecord( stop_t, 0 );
    cudaEventSynchronize( stop_t );

    cudaEventElapsedTime( &elapsedTime_t, start_t , stop_t );
    printf( "TM_HTH = %.2f ms\n", elapsedTime_t);
#endif

    // We do this only if t  == 0 because the
    // function li.evaluate_objective_function() takes care for all t=1:T
    if (t == 0 && use_inequalities)
    {
        float *d_obj_ineqs;
        HANDLE_ERROR( cudaMalloc( (void**)&d_obj_ineqs, sizeof(float)) );

        li.compute_gradient_and_hessian(handle, o.ov_ptr, d_obj_ineqs);

        if (use_texture)
        {
            update_gradient<<<(o.ov_ptr->Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_lb, li.f_beta_ub, o.dG_dtheta, o.ov_ptr->Kbeta, 0);
            update_diagonal_of_hessian_wbounds<<<1, o.ov_ptr->Kbeta>>>(li.f_beta_lb, li.f_beta_ub, o.JTJ, o.ov_ptr->Kbeta, o.ov_ptr->Ktotal, 0);
        }

        const uint K_except_texture = o.ov_ptr->Ktotal - o.ov_ptr->Kbeta;

        // <!-- focus here -->
        update_bottom_right_of_matrix<<<(K_except_texture*K_except_texture+NTHREADS-1)/NTHREADS, NTHREADS>>>( li.nabla2F, o.JTJ, K_except_texture, o.ov_ptr->Kbeta );

        // <!-- focus here -->
        cublasSaxpy(handle, li.Ktotal, &plus_one_, li.gradient, 1, o.dG_dtheta+o.ov_ptr->Kbeta, 1);


        if (use_texture)
        {
            neglogify<<<(o.ov_ptr->Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_ub, o.ov_ptr->Kbeta);
            neglogify<<<(o.ov_ptr->Kbeta+NTHREADS-1)/NTHREADS, NTHREADS>>>(li.f_beta_lb, o.ov_ptr->Kbeta);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, o.ov_ptr->Kbeta, &plus_one_, li.vecOnes, 1, li.f_beta_lb, o.ov_ptr->Kbeta, &plus_one_, d_obj_ineqs, 1);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, o.ov_ptr->Kbeta, &plus_one_, li.vecOnes, 1, li.f_beta_ub, o.ov_ptr->Kbeta, &plus_one_, d_obj_ineqs, 1);
        }

        //////////multiply_and_add_to_scalar<<<1,1>>>(o.ov_ptr->grad_corr, &plus_one_, d_obj_ineqs);
        add_to_scalar_negated<<<1,1>>>(o.ov_ptr->grad_corr,  d_obj_ineqs);
        cudaFree(d_obj_ineqs);

        // need to reset frame no because li.compute_gradient_and_hessian() changes it
        o.ov_ptr->set_frame(t);
    }
}





















void DerivativeComputer::compute_hessian_and_gradient_for_lambda(Optimizer& o,
                                                                 Solver& s_lambda, RotationComputer& rc,
                                                                 ushort &N_unique_pixels,
                                                                 cusolverDnHandle_t& handleDn,
                                                                 cublasHandle_t &handle,bool set_ks)
{
    if (set_ks)
    {
        ////////////////////////////////
        thrust::inclusive_scan(thrust::device, o.d_M0, o.d_M0 + NTOTAL_PIXELS, o.d_cumM0); // in-place scan
        thrust::sequence(thrust::device, o.d_ks_sortidx, o.d_ks_sortidx + N_unique_pixels);
        thrust::sort_by_key(thrust::device, o.d_ks_sorted, o.d_ks_sorted + N_unique_pixels, o.d_ks_sortidx);

        thrust::sequence(thrust::device, o.d_ks_sortidx_sortidx, o.d_ks_sortidx_sortidx + N_unique_pixels);
        HANDLE_ERROR( cudaMemcpy( o.d_ks_sortidx_copy, o.d_ks_sortidx,  sizeof(ushort)*Nrender_estimated, cudaMemcpyDeviceToDevice) );

        thrust::sort_by_key(thrust::device, o.d_ks_sortidx_copy, o.d_ks_sortidx_copy + N_unique_pixels, o.d_ks_sortidx_sortidx);

        // We'll put back at least the two following functions (fill_ksides and fill_krels)
        ///////////////////////fill_ksides<<<N_unique_pixels, 1>>>(o.d_ks_sorted,  o.d_ks_left, o.d_ks_right, o.d_ks_above, o.d_ks_below);
        fill_ksides<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_ks_sorted,  o.d_ks_left, o.d_ks_right, o.d_ks_above, o.d_ks_below);

        //////////////////fill_krels<<<N_unique_pixels, 1>>>(N_unique_pixels, o.d_ks_sorted,
        fill_krels<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                          N_unique_pixels, o.d_ks_sorted,
                                                                          o.d_ks_sortidx,
                                                                          o.d_ks_sortidx_sortidx,
                                                                          o.d_ks_left, 		o.d_ks_right, 		o.d_ks_above, 		o.d_ks_below,
                                                                          o.d_kl_rel, 		o.d_kr_rel, 		o.d_ka_rel, 		o.d_kb_rel,
                                                                          o.d_kl_rel_sorted, 	o.d_kr_rel_sorted, 	o.d_ka_rel_sorted, 	o.d_kb_rel_sorted,
                                                                          o.d_cumM0);

        ///////////// <<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
        fill_krels2<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                           o.d_ks_unsorted,
                                                                           o.d_ks_sortidx,
                                                                           o.d_ks_sortidx_sortidx,
                                                                           o.d_kl_rel, 		o.d_kr_rel, 		o.d_ka_rel, 		o.d_kb_rel,
                                                                           o.d_kl_rel_sorted, 	o.d_kr_rel_sorted, 	o.d_ka_rel_sorted, 	o.d_kb_rel_sorted);
    }

    float zero_ = 0.0f;
    float plus_one_ = 1.0f;




    // d_dI_dlambda = d_dI_dlambda*R'
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Nrender_estimated, 	3, 3, &plus_one_,
                o.d_dI_dlambda, Nrender_estimated, rc.R, 3, &zero_, o.dId_dlambda, Nrender_estimated);

    fill_optimization_auxiliary_variables_phase2_new<<<(Nrender_estimated*3+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                                                                o.dId_dlambda,
                                                                                                                o.gx, o.gy, o.h,
                                                                                                                o.d_kl_rel, o.d_kr_rel, o.d_ka_rel, o.d_kb_rel,
                                                                                                                o.dgx_dlambda,
                                                                                                                o.dgy_dlambda,
                                                                                                                N_unique_pixels,
                                                                                                                3);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3, 3, Nrender_estimated, &alpha_, o.dgx_dlambda, Nrender_estimated, o.dgx_dlambda, Nrender_estimated, &beta_, o.JTJ_lambda, 3);

    beta_ = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3, 3, Nrender_estimated, &alpha_, o.dgy_dlambda, Nrender_estimated, o.dgy_dlambda, Nrender_estimated, &beta_, o.JTJ_lambda, 3);

    alpha_ = -1.0f;
    beta_ = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 3, Nrender_estimated, &alpha_, o.gxs_norm, 1, o.dgx_dlambda, Nrender_estimated, &beta_, o.dG_dtheta_lambda, 1);

    beta_ = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 3, Nrender_estimated, &alpha_, o.gys_norm, 1, o.dgy_dlambda, Nrender_estimated, &beta_, o.dG_dtheta_lambda, 1);

    copyMatsFloatToDouble<<<(s_lambda.n*s_lambda.n+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.JTJ_lambda, o.dG_dtheta_lambda, s_lambda.JTJ, s_lambda.dG_dtheta, s_lambda.n);

    s_lambda.solve(handleDn);
}





void DerivativeComputer::compute_hessian_and_gradient_for_Lintensity(Optimizer& o,
                                                                     ushort &N_unique_pixels,
                                                                     cublasHandle_t &handle,
                                                                     float *search_dir_Lintensity,
                                                                     float *dg_Lintensity,
                                                                     bool set_ks)
{
    if (set_ks)
    {
        ////////////////////////////////
        thrust::inclusive_scan(thrust::device, o.d_M0, o.d_M0 + NTOTAL_PIXELS, o.d_cumM0); // in-place scan
        thrust::sequence(thrust::device, o.d_ks_sortidx, o.d_ks_sortidx + N_unique_pixels);
        thrust::sort_by_key(thrust::device, o.d_ks_sorted, o.d_ks_sorted + N_unique_pixels, o.d_ks_sortidx);

        thrust::sequence(thrust::device, o.d_ks_sortidx_sortidx, o.d_ks_sortidx_sortidx + N_unique_pixels);
        HANDLE_ERROR( cudaMemcpy( o.d_ks_sortidx_copy, o.d_ks_sortidx,  sizeof(ushort)*Nrender_estimated, cudaMemcpyDeviceToDevice) );

        thrust::sort_by_key(thrust::device, o.d_ks_sortidx_copy, o.d_ks_sortidx_copy + N_unique_pixels, o.d_ks_sortidx_sortidx);

        // We'll put back at least the two following functions (fill_ksides and fill_krels)
        ///////////////////////fill_ksides<<<N_unique_pixels, 1>>>(o.d_ks_sorted,  o.d_ks_left, o.d_ks_right, o.d_ks_above, o.d_ks_below);
        fill_ksides<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_ks_sorted,  o.d_ks_left, o.d_ks_right, o.d_ks_above, o.d_ks_below);

        //////////////////fill_krels<<<N_unique_pixels, 1>>>(N_unique_pixels, o.d_ks_sorted,
        fill_krels<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                          N_unique_pixels, o.d_ks_sorted,
                                                                          o.d_ks_sortidx,
                                                                          o.d_ks_sortidx_sortidx,
                                                                          o.d_ks_left, 		o.d_ks_right, 		o.d_ks_above, 		o.d_ks_below,
                                                                          o.d_kl_rel, 		o.d_kr_rel, 		o.d_ka_rel, 		o.d_kb_rel,
                                                                          o.d_kl_rel_sorted, 	o.d_kr_rel_sorted, 	o.d_ka_rel_sorted, 	o.d_kb_rel_sorted,
                                                                          o.d_cumM0);

        ///////////// <<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
        fill_krels2<<<(N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                           o.d_ks_unsorted,
                                                                           o.d_ks_sortidx,
                                                                           o.d_ks_sortidx_sortidx,
                                                                           o.d_kl_rel, 		o.d_kr_rel, 		o.d_ka_rel, 		o.d_kb_rel,
                                                                           o.d_kl_rel_sorted, 	o.d_kr_rel_sorted, 	o.d_ka_rel_sorted, 	o.d_kb_rel_sorted);
    }

    float zero_ = 0.0f;
    float plus_one_ = 1.0f;

    fill_optimization_auxiliary_variables_phase2_new<<<(Nrender_estimated+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                                                              o.dI_dLintensity,
                                                                                                              o.gx, o.gy, o.h,
                                                                                                              o.d_kl_rel, o.d_kr_rel, o.d_ka_rel, o.d_kb_rel,
                                                                                                              o.dgx_dLintensity,
                                                                                                              o.dgy_dLintensity,
                                                                                                              N_unique_pixels,
                                                                                                              1);




    float *JTJ_Lintensity;

    HANDLE_ERROR( cudaMalloc( (void**)&JTJ_Lintensity, sizeof(float)) );



    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, Nrender_estimated, &alpha_, o.dgx_dLintensity, Nrender_estimated, o.dgx_dLintensity, Nrender_estimated, &beta_, JTJ_Lintensity, 1);

    beta_ = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, Nrender_estimated, &alpha_, o.dgy_dLintensity, Nrender_estimated, o.dgy_dLintensity, Nrender_estimated, &beta_, JTJ_Lintensity, 1);


    alpha_ = -1.0f;
    beta_ = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, Nrender_estimated, &alpha_, o.gxs_norm, 1, o.dgx_dLintensity, Nrender_estimated, &beta_, dg_Lintensity, 1);

    beta_ = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, Nrender_estimated, &alpha_, o.gys_norm, 1, o.dgy_dLintensity, Nrender_estimated, &beta_, dg_Lintensity, 1);


    compute_Lintensity_search_dir<<<1,1>>>(JTJ_Lintensity, dg_Lintensity, search_dir_Lintensity);

    HANDLE_ERROR( cudaFree( JTJ_Lintensity ) );
}








