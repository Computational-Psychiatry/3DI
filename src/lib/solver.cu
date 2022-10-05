/*
 * solver.cu
 *
 *  Created on: Aug 14, 2020
 *      Author: root
 */


#include "solver.h"

__global__ void copyMatsFloatToDouble( const float *f_A, const float *f_b, double *d_A, double *d_b, const int n)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= n*n)
        return;

    d_A[x] = (double) f_A[x];


    if (x < n) {
        d_b[x] = (double) f_b[x];
    }

    /*

    __syncthreads();

    if (x == 0)
    {
        printf("===============JTJ===============\n");
        for (size_t i=0; i<n; ++i)
        {
            for (size_t j=0; j<n; ++j)
            {
                printf("%.3f\t", d_A[j+n*i]);
            }
            printf("\n");
        }
        printf("===============JTJ===============\n");
    }


    if (x == 0)
    {
        printf("===============G===============\n");
            for (size_t j=0; j<n; ++j)
            {
                printf("%.3f\t", d_b[j]);
            }
        printf("===============G===============\n");
    }

    */
}





__global__ void copyVectorDoubleToFloat( const double *d_x, float *f_x)
{

    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    const int n = colix + rowix*blockDim.x;

    f_x[n] = (float) d_x[n];

}



__global__ void copyVectorDoubleToFloatNegated( const double *d_x, float *f_x, int size_vec)
{
//    const int n = threadIdx.x;

    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    const int n = colix + rowix*blockDim.x;

    if (n >= size_vec)
        return;




    f_x[n] = (float) -d_x[n];

}





