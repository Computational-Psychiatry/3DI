/*
 * solver.h
 *
 *  Created on: Aug 14, 2020
 *      Author: root
 */

#include "constants.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include <iostream>
#include "config.h"






#ifndef SOLVER_H_
#define SOLVER_H_



__global__ void copyMatsFloatToDouble( const float *f_A, const float *f_b, double *d_A, double *d_b, const int n);

__global__ void copyVectorDoubleToFloat( const double *d_x, float *f_x);

__global__ void copyVectorDoubleToFloatNegated(const double *d_x, float *f_x, int size_vec);

struct Solver
{
	double *JTJ;
	double *dG_dtheta;
	double *x;

    double *buffer = NULL;

    double *MEM_ALL;

    float *search_dir;

    int bufferSize = 0;

    cusolverDnHandle_t *handle;


	int n;
	cublasFillMode_t uplo;
    Solver(cusolverDnHandle_t& handle_NOTUSED, int _n) : n(_n), uplo(CUBLAS_FILL_MODE_LOWER) // , handle(_handle)
    {
        cusolverDnHandle_t handle_local;
        cusolverDnCreate(&handle_local);
		HANDLE_ERROR(cudaMalloc((void**)&MEM_ALL, sizeof(double)*(n*n + n + n)));

		JTJ = MEM_ALL;
		dG_dtheta = JTJ + n*n;
		x = dG_dtheta + n;

		HANDLE_ERROR(cudaMalloc((void**)&search_dir, sizeof(float)*n));

		//buffer = x + n;

        cusolverDnDpotrf_bufferSize(handle_local, uplo, n, JTJ, n, &bufferSize);
        cudaMalloc(&buffer, sizeof(double)*bufferSize);
        cusolverDnDestroy(handle_local);


	}


	bool solve(cusolverDnHandle_t& handle )
	{
        int *info; //, *info2;
        int h_info = 0, h_info2 = 0;

        cudaMalloc(&info, sizeof(int));
//        cudaMalloc(&info2, sizeof(int));
        cusolverStatus_t statusD = cusolverDnDpotrf(handle, uplo, n, JTJ, n, buffer, bufferSize, info);
        cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

	    cudaMemcpy(x, dG_dtheta, sizeof(double)*n, cudaMemcpyDeviceToDevice);
        cusolverStatus_t statusS = cusolverDnDpotrs(handle, uplo, n, 1, JTJ, n, x, n, NULL);
//        cudaMemcpy(&h_info2, info2, sizeof(int), cudaMemcpyDeviceToHost);

	    cudaDeviceSynchronize();
	
	    bool result = true;
        if (h_info != 0) {
            int asdasd=1;
            if (config::PRINT_WARNINGS)
                std::cout << "Unsuccessful potrf execution " << "devInfo = " << h_info << std::endl;
    	    result = false;
        }

        /*
        if (h_info2 != 0) {
            int asdasd=1;
            std::cout << "Unsuccessful potrs execution\n\n" << "devInfo = " << h_info2 << "\n\n";
        }
        */

        copyVectorDoubleToFloatNegated<<<(n+NTHREADS-1)/NTHREADS, NTHREADS>>>(x, search_dir, n);
        cudaFree(info);
//        cudaFree(info2);
	    return result;
	}



	~Solver()
	{
//		cusolverDnDestroy(handle);
		cudaFree( buffer );
		cudaFree( search_dir );
		cudaFree( MEM_ALL );
	}
};







struct SolverQR
{
	double *JTJ;
	double *dG_dtheta;
	double *x;

    double *buffer = NULL;

    double *MEM_ALL;

    float *search_dir;

    int bufferSize = 0;

    cusolverDnHandle_t *handle;

	int n;
	cublasFillMode_t uplo;
	SolverQR(cusolverDnHandle_t& handle, int _n) : n(_n), uplo(CUBLAS_FILL_MODE_LOWER) // , handle(_handle)
	{
//	    cusolverDnCreate(&handle);
		HANDLE_ERROR(cudaMalloc((void**)&MEM_ALL, sizeof(double)*(n*n + n + n)));

		JTJ = MEM_ALL;
		dG_dtheta = JTJ + n*n;
		x = dG_dtheta + n;

		HANDLE_ERROR(cudaMalloc((void**)&search_dir, sizeof(float)*n));


/*
		HANDLE_ERROR(cudaMalloc((void**)&JTJ, sizeof(double)*n*n));
		HANDLE_ERROR(cudaMalloc((void**)&dG_dtheta, sizeof(double)*n));
		HANDLE_ERROR(cudaMalloc((void**)&x, sizeof(double)*n));
		HANDLE_ERROR(cudaMalloc(&buffer, sizeof(double)*bufferSize));
		*/
	}


	void solve(cusolverDnHandle_t& handleDn, cublasHandle_t& handle )
	{
	    int *info = NULL;
	    int *info2 = NULL;
	    int h_info = 0;

	    int infoint = 123;
	    int info2int = 456;

	    int lda = n;

	    int bufferSize = 0;
	    int bufferSize_geqrf = 0;
	    int bufferSize_ormqr = 0;
	    double *buffer = NULL;
	    double *A = NULL;
	    double *tau = NULL;
	    double start, stop;
	    double time_solve;
	    const double one = 1.0;

	    cusolverDnDgeqrf_bufferSize( handleDn, n, n, JTJ, lda, &bufferSize_geqrf);
	    cusolverDnDormqr_bufferSize( handleDn, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, JTJ, lda, NULL,x, n, &bufferSize_ormqr);


	    bufferSize = (bufferSize_geqrf > bufferSize_ormqr)? bufferSize_geqrf : bufferSize_ormqr ;

	    cudaMalloc(&info, sizeof(int));
	    cudaMalloc(&buffer, sizeof(double)*bufferSize);
	    cudaMalloc((void**)&tau, sizeof(double)*n);

	    cudaMemset(info, 0, sizeof(int));


	// compute QR factorization
	    cusolverDnDgeqrf(handleDn, n, n, JTJ, lda, tau, buffer, bufferSize, info);

	    cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);


	    cudaMemcpy(x, dG_dtheta, sizeof(double)*n, cudaMemcpyDeviceToDevice);

	    cusolverDnDormqr( handleDn, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, JTJ, lda, tau, x, n, buffer, bufferSize, info);

	    cublasDtrsm( handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, 1, &one, JTJ, lda, x, n);
	    cudaDeviceSynchronize();


	    if (info  ) { cudaFree(info  ); }
	    if (buffer) { cudaFree(buffer); }
	    if (tau   ) { cudaFree(tau); }


        copyVectorDoubleToFloatNegated<<<(n+NTHREADS-1)/NTHREADS, NTHREADS>>>(x, search_dir, n);



	}



	~SolverQR()
	{
//		cusolverDnDestroy(handle);
		cudaFree( search_dir );
		cudaFree( MEM_ALL );
/*
		cudaFree( x );
		cudaFree( JTJ );
		cudaFree( dG_dtheta );
		cudaFree( buffer );
		*/
	}
};




#endif /* SOLVER_H_ */
