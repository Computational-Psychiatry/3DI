/*
 * orthographic_initializer.cu
 *
 *  Created on: Sep 17, 2020
 *      Author: root
 */


#include "orthographic_initializer.h"





__global__ void compute_orth_projections(const float *x_orthographic, const float *xl_bar, const float *yl_bar, const float *p0L_mat,
                                         const float *Rp, float *xproj, float *yproj)
{
    const ushort rowix = blockIdx.x;
    const ushort colix = threadIdx.x;

    ushort n = colix + rowix*blockDim.x;

    if (n >= NLANDMARKS_51)
        return;

    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float sigmax[1];
    __shared__ float sigmay[1];

    if (colix == 0)
    {
        taux[0] = x_orthographic[0];
        tauy[0] = x_orthographic[1];
        sigmax[0] = x_orthographic[2];
        sigmay[0] = x_orthographic[3];
    }

    __syncthreads();

    const int i = n % NLANDMARKS_51;
    const int j = n / NLANDMARKS_51;

    const float r1p = Rp[i*3];
    const float r2p = Rp[i*3+1];

    if (j == 0)
    {
        xproj[i] = sigmax[0]*(r1p+taux[0]);
        yproj[i] = sigmay[0]*(r2p+tauy[0]);
    }
}



__global__ void compute_gradient_hessian_obj_orthographic(const float *x_orthographic, const float *xl_bar, const float *yl_bar, const float *p0L_mat,
                                                          const float *Rp, const float *dRp_du1, const float *dRp_du2, const float *dRp_du3, const bool eval_gradients, float *nablaW, float *err)
{
    const ushort rowix = blockIdx.x;
    const ushort colix = threadIdx.x;

    ushort n = colix + rowix*blockDim.x;

    if (n >= NLANDMARKS_51*7)
        return;

    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float sigmax[1];
    __shared__ float sigmay[1];

    if (colix == 0)
    {
        taux[0] = x_orthographic[0];
        tauy[0] = x_orthographic[1];
        sigmax[0] = x_orthographic[2];
        sigmay[0] = x_orthographic[3];
    }

    __syncthreads();

    const int i = n % NLANDMARKS_51;
    const int j = n / NLANDMARKS_51;

    const float r1p = Rp[i*3];
    const float r2p = Rp[i*3+1];

    if (eval_gradients)
    {
        const float dr1p_du1 = dRp_du1[i*3];
        const float dr2p_du1 = dRp_du1[i*3+1];

        const float dr1p_du2 = dRp_du2[i*3];
        const float dr2p_du2 = dRp_du2[i*3+1];

        const float dr1p_du3 = dRp_du3[i*3];
        const float dr2p_du3 = dRp_du3[i*3+1];

        if (j == 0) {
            nablaW[ i+j*2*NLANDMARKS_51 ] = 	sigmax[0];
            nablaW[ i+NLANDMARKS_51+j*2*NLANDMARKS_51 ] = 	0.0f;
        } else if (j == 1) {
            nablaW[ i+j*2*NLANDMARKS_51 ] = 	0.0f;
            nablaW[ i+NLANDMARKS_51+j*2*NLANDMARKS_51 ] = 	sigmay[0];
        } else if (j == 2) {
            nablaW[ i+j*2*NLANDMARKS_51 ] = 				taux[0]+r1p;
            nablaW[ i+NLANDMARKS_51+j*2*NLANDMARKS_51 ] = 	0.0f;
        } else if (j == 3) {
            nablaW[ i+j*2*NLANDMARKS_51 ] = 				0.0f;
            nablaW[ i+NLANDMARKS_51+j*2*NLANDMARKS_51 ] = 	tauy[0]+r2p;
        } else if (j == 4) {
            nablaW[ i+j*2*NLANDMARKS_51 ] = 				sigmax[0]*dr1p_du1;
            nablaW[ i+NLANDMARKS_51+j*2*NLANDMARKS_51 ] = 	sigmay[0]*dr2p_du1;
        } else if (j == 5) {
            nablaW[ i+j*2*NLANDMARKS_51 ] = 				sigmax[0]*dr1p_du2;
            nablaW[ i+NLANDMARKS_51+j*2*NLANDMARKS_51 ] = 	sigmay[0]*dr2p_du2;
        } else if (j == 6) {
            nablaW[ i+j*2*NLANDMARKS_51 ] = 				sigmax[0]*dr1p_du3;
            nablaW[ i+NLANDMARKS_51+j*2*NLANDMARKS_51 ] = 	sigmay[0]*dr2p_du3;
        }
    }

    if (j == 0)
    {
        const float diffx = sigmax[0]*(r1p+taux[0])-xl_bar[i];
        const float diffy = sigmay[0]*(r2p+tauy[0])-yl_bar[i];
        err[i] = diffx;
        err[i+NLANDMARKS_51] = diffy;
    }
}


void OrthographicInitializer::reset_orthographic()
{




    float *x_orthographic;
    x_orthographic = (float*) malloc( 7*sizeof(float) );
    x_orthographic[0] = 0.00f; //   0.1f;
    x_orthographic[1] = 0.00f; // 0.0f;
    x_orthographic[2] = 15.0f;
    x_orthographic[3] = 15.0f;
    x_orthographic[4] = 0.00f;
    x_orthographic[5] = 0.00f;
    x_orthographic[6] = 0.05f;



    HANDLE_ERROR( cudaMemcpy( d_x_orthographic, x_orthographic, sizeof(float)*7, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( d_x_orthographic_linesearch, x_orthographic, sizeof(float)*7, cudaMemcpyHostToDevice ) );

    free( x_orthographic );

}


OrthographicInitializer::OrthographicInitializer(cusolverDnHandle_t& handleDn) : s(handleDn, 7)
{


    float *p0L_mat;
    p0L_mat = (float*)malloc( 3*NLANDMARKS_51*sizeof(float) );


    vector< vector<float> > p0L_mat_vec  = read2DVectorFromFile<float>(config::P0L_PATH, NLANDMARKS_51, 3);



    for (size_t i=0; i<NLANDMARKS_51; ++i)
    {
        p0L_mat[i] = (float) p0L_mat_vec[i][0];
        p0L_mat[i + NLANDMARKS_51] = (float) p0L_mat_vec[i][1];
        p0L_mat[i + 2*NLANDMARKS_51] = (float) p0L_mat_vec[i][2];
    }

    HANDLE_ERROR( cudaMalloc( (void**)&d_x_orthographic, sizeof(float)*7) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_x_orthographic_linesearch, sizeof(float)*7) );

    reset_orthographic();



    HANDLE_ERROR( cudaMalloc( (void**)&d_xl_bar, sizeof(float)*NLANDMARKS_51) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_yl_bar, sizeof(float)*NLANDMARKS_51) );

    HANDLE_ERROR( cudaMalloc( (void**)&d_err, sizeof(float)*NLANDMARKS_51*2) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_dg, sizeof(float)*7) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_obj, sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_obj_tmp, sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_tmp, sizeof(float) ) );

    HANDLE_ERROR( cudaMalloc( (void**)&d_JTJ, sizeof(float)*7*7) );


    HANDLE_ERROR( cudaMalloc( (void**)&d_Rp, sizeof(float)*3*NLANDMARKS_51) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_dRp_du1, sizeof(float)*3*NLANDMARKS_51) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_dRp_du2, sizeof(float)*3*NLANDMARKS_51) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_dRp_du3, sizeof(float)*3*NLANDMARKS_51) );


    HANDLE_ERROR( cudaMalloc( (void**)&d_nablaW, sizeof(float)*7*2*NLANDMARKS_51) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_p0L_mat, sizeof(float)*NLANDMARKS_51*3) );


    HANDLE_ERROR( cudaMemcpy( d_p0L_mat, p0L_mat, sizeof(float)*NLANDMARKS_51*3, cudaMemcpyHostToDevice ) );



    rc.set_u_ptr(d_x_orthographic+4);
    rc_linesearch.set_u_ptr(d_x_orthographic_linesearch+4);

    free( p0L_mat );

}


void OrthographicInitializer::set_landmarks(const float* xl, const float* yl)
{
    float xmean = thrust::reduce(xl, xl+NLANDMARKS_51, 0.0f, thrust::plus<float>())/(float)NLANDMARKS_51;
    float ymean = thrust::reduce(yl, yl+NLANDMARKS_51, 0.0f, thrust::plus<float>())/(float)NLANDMARKS_51;

    float xl_bar[NLANDMARKS_51], yl_bar[NLANDMARKS_51];

    for (size_t i=0; i<NLANDMARKS_51; ++i)
    {
        xl_bar[i] = xl[i]-xmean;
        yl_bar[i] = yl[i]-ymean;
    }

    HANDLE_ERROR( cudaMemcpy( d_xl_bar, xl_bar, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( d_yl_bar, yl_bar, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice ) );
}








void OrthographicInitializer::fit_model(cusolverDnHandle_t& handleDn, cublasHandle_t& handle, const float* xl, const float* yl,
                                        float *yaw_ptr, float *pitch_ptr, float *roll_ptr, bool reset_variables)
{
    if (reset_variables)
        reset_orthographic();

    set_landmarks(xl, yl);

    bool terminate = false;
    int q;

    for (q=0; q<MAXITER_OUTER; ++q)
    {
        if (terminate)
            break;

        float alpha_ = 1.0f;
        float beta_ = 0.0f;

        rc.process();
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, NLANDMARKS_51, 3, &alpha_, rc.R,      3, d_p0L_mat, NLANDMARKS_51, &beta_, d_Rp, 3);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, NLANDMARKS_51, 3, &alpha_, rc.dR_du1, 3, d_p0L_mat, NLANDMARKS_51, &beta_, d_dRp_du1, 3);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, NLANDMARKS_51, 3, &alpha_, rc.dR_du2, 3, d_p0L_mat, NLANDMARKS_51, &beta_, d_dRp_du2, 3);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, NLANDMARKS_51, 3, &alpha_, rc.dR_du3, 3, d_p0L_mat, NLANDMARKS_51, &beta_, d_dRp_du3, 3);

        compute_gradient_hessian_obj_orthographic<<<1, NLANDMARKS_51*7>>>(d_x_orthographic, d_xl_bar, d_yl_bar, d_p0L_mat, d_Rp, d_dRp_du1, d_dRp_du2, d_dRp_du3, true, d_nablaW, d_err );

        //////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////
        /*
        cv::Mat emptyFrame(1000, 1000, CV_32FC3, cv::Scalar::all(255));

        float *d_xproj, *d_yproj;
        float h_xproj[NLANDMARKS_51], h_yproj[NLANDMARKS_51];
        float h_xl_bar[NLANDMARKS_51], h_yl_bar[NLANDMARKS_51];
        HANDLE_ERROR( cudaMalloc( (void**)&d_xproj, sizeof(float)*NLANDMARKS_51) );
        HANDLE_ERROR( cudaMalloc( (void**)&d_yproj, sizeof(float)*NLANDMARKS_51) );

        compute_orth_projections<<<NLANDMARKS_51, 1>>>(d_x_orthographic, d_xl_bar, d_yl_bar, d_p0L_mat, d_Rp, d_xproj, d_yproj);

        cudaMemcpy(h_xproj, d_xproj, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_yproj, d_yproj, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_xl_bar, d_xl_bar, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_yl_bar, d_yl_bar, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost);


        for (uint ui=0; ui<NLANDMARKS_51; ++ui) {
            double offset=400;
            cv::Point2f ptd(offset+h_xl_bar[ui], offset+h_yl_bar[ui]);
            cv::Point2f ptp(offset+h_xproj[ui], offset+h_yproj[ui]);
            cv::circle(emptyFrame, ptd, 3, cv::Scalar(0,0,255), cv::FILLED, 8, 0);
            cv::circle(emptyFrame, ptp, 3, cv::Scalar(255,0,0), cv::FILLED, 8, 0);
        }

        print_vector(d_x_orthographic, 7, "CUR_ITERATION");


        cv::imshow("emptyFrame", emptyFrame);
        cv::waitKey(0);

        cudaFree( d_xproj );
        cudaFree( d_yproj );
        */
        //////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 7, 7, 2*NLANDMARKS_51, &alpha_, d_nablaW, 2*NLANDMARKS_51, d_nablaW, 2*NLANDMARKS_51, &beta_, d_JTJ, 7);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 7, 1, 2*NLANDMARKS_51, &alpha_, d_nablaW, 2*NLANDMARKS_51, d_err, 2*NLANDMARKS_51, &beta_, d_dg, 7);

        cublasSdot(handle, 2*NLANDMARKS_51, d_err, 1, d_err, 1, d_obj);
        copyMatsFloatToDouble<<<(s.n*s.n+NTHREADS-1)/NTHREADS, NTHREADS>>>(d_JTJ, d_dg, s.JTJ, s.dG_dtheta, s.n);


        s.solve(handleDn);

        float obj, obj_tmp, tmp;
        cudaMemcpy(&obj, d_obj, sizeof(float), cudaMemcpyDeviceToHost);

        float t_coef = 1.0f;
        float ALPHA = 0.4f;
        float BETA = 0.5f;


        const int MAX_INNER_ITERS = 1000;
        int inner_iter = 0;

        while (inner_iter < MAX_INNER_ITERS)
        {
            if (t_coef < 0.1f) {
                terminate = true;
                break;
            }

            set_xtmp<<<(s.n+NTHREADS-1)/NTHREADS, NTHREADS>>>(s.search_dir, d_x_orthographic, t_coef, d_x_orthographic_linesearch, s.n);
            rc_linesearch.process_without_derivatives();
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 3, NLANDMARKS_51, 3, &alpha_, rc_linesearch.R, 3, d_p0L_mat, NLANDMARKS_51, &beta_, d_Rp, 3);
            compute_gradient_hessian_obj_orthographic<<<1, NLANDMARKS_51*7>>>(d_x_orthographic_linesearch, d_xl_bar, d_yl_bar, d_p0L_mat, d_Rp, d_dRp_du1, d_dRp_du2, d_dRp_du3, false, d_nablaW, d_err );
            cublasSdot(handle, 2*NLANDMARKS_51, d_err, 1, d_err, 1, d_obj_tmp);

            cudaMemcpy(&obj_tmp, d_obj_tmp, sizeof(float), cudaMemcpyDeviceToHost);

            //                            std::cout << "\t OBJ is " << obj << " OBJ_TMP IS:" << obj_tmp << " t_coef is " << t_coef << std::endl;

            cublasSdot(handle, s.n, s.search_dir, 1, d_dg, 1, d_tmp);
            cudaMemcpy(&tmp, d_tmp, sizeof(float), cudaMemcpyDeviceToHost);

            tmp *= ALPHA*t_coef;

            if (obj_tmp < obj+tmp) {
                cudaMemcpy(d_x_orthographic, d_x_orthographic_linesearch, sizeof(float)*s.n, cudaMemcpyDeviceToDevice);
                break;
            }

            t_coef = t_coef * BETA;
            inner_iter++;
        }
    }



    if (yaw_ptr != NULL && pitch_ptr != NULL && roll_ptr != NULL)
    {
        float yaw, pitch, roll;
        rc_linesearch.compute_euler_angles(yaw, pitch, roll);
        *yaw_ptr = RAD2DEG(yaw);
        *pitch_ptr = RAD2DEG(pitch);
        *roll_ptr = RAD2DEG(roll);
    }

    /*
    std::cout << "\t (OI) yaw: " << RAD2DEG(yaw) << '\t' << " pitch: " << RAD2DEG(pitch) << '\t' << " roll: " << RAD2DEG(roll) << std::endl;
    */
}



OrthographicInitializer::~OrthographicInitializer()
{


    HANDLE_ERROR( cudaFree( d_tmp) );
    HANDLE_ERROR( cudaFree( d_obj_tmp) );
    HANDLE_ERROR( cudaFree( d_obj ) );
    HANDLE_ERROR( cudaFree( d_dg ) );
    HANDLE_ERROR( cudaFree( d_err ) );
    HANDLE_ERROR( cudaFree( d_JTJ ) );

    HANDLE_ERROR( cudaFree( d_nablaW ) );

    HANDLE_ERROR( cudaFree( d_dRp_du3 ) );
    HANDLE_ERROR( cudaFree( d_dRp_du2 ) );
    HANDLE_ERROR( cudaFree( d_dRp_du1 ) );
    HANDLE_ERROR( cudaFree( d_Rp ) );

    HANDLE_ERROR( cudaFree( d_xl_bar ) );
    HANDLE_ERROR( cudaFree( d_yl_bar ) );
    HANDLE_ERROR( cudaFree( d_x_orthographic ) );

    HANDLE_ERROR( cudaFree( d_p0L_mat ) );

}





vector<vector<float> > LightPoseEstimator::estimate_poses(cusolverDnHandle_t &handleDn, cublasHandle_t &handle,
                                        OrthographicInitializer &oi, LandmarkData &ld)
{
    vector<vector<float> > all_poses;
    for (size_t t=0; t<ld.get_num_frames(); ++t)
    {
        std::vector<float> xp_vec = ld.get_xpvec(t);
        std::vector<float> yp_vec = ld.get_ypvec(t);

        float yaw, pitch, roll;

        oi.fit_model(handleDn, handle,
                     &xp_vec[0], &yp_vec[0],
                &yaw, &pitch, &roll);

        std::cout << std::setw(4) << t << ": " << yaw << '\t' << pitch << '\t' << roll << std::endl;
    }

    return all_poses;
}

