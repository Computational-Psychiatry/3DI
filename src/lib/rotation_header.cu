/*
 * rotation_header.cu
 *
 *  Created on: Aug 10, 2020
 *      Author: root
 */


#include "rotation_header.h"


__global__ void print3(float *A)
{
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            printf("%.04f ", A[i*3+j]);
        }
        printf("\n");
    }
}



__device__ void multiplyAB_3x3(const float *A, const float *B, float *C)
{
    C[0] = A[0]*B[0] + A[3]*B[1] + A[6]*B[2];
    C[1] = A[1]*B[0] + A[4]*B[1] + A[7]*B[2];
    C[2] = A[2]*B[0] + A[5]*B[1] + A[8]*B[2];

    C[3] = A[0]*B[3] + A[3]*B[4] + A[6]*B[5];
    C[4] = A[1]*B[3] + A[4]*B[4] + A[7]*B[5];
    C[5] = A[2]*B[3] + A[5]*B[4] + A[8]*B[5];

    C[6] = A[0]*B[6] + A[3]*B[7] + A[6]*B[8];
    C[7] = A[1]*B[6] + A[4]*B[7] + A[7]*B[8];
    C[8] = A[2]*B[6] + A[5]*B[7] + A[8]*B[8];
}



__device__ void sum_AB_3x3(const float* A, const float* B,  float *C);




__global__ void fill_theta_etc(const float *u, float *theta_inv2, float *sin_theta, float *cos_inv_sum,
                               float *u_skew, float *unorm_skew,
                               float *R, float *IR, float *R1a, float *R1b, float *R1c, float *R2a, float *R2b, float *R2c, float *R3a, float *R3b, float *R3c,
                               float *dR_u1, float *dR_u2, float *dR_u3)
{
    /*
    u[0] = -0.1980f;
    u[1] = 0.7241f;
    u[2] = 0.7886f;
    */
    float theta = sqrtf(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

    float unorm[3];
    unorm[0] = u[0]/theta;
    unorm[1] = u[1]/theta;
    unorm[2] = u[2]/theta;


    fill_skew_matrix_dev4(u_skew, R1a, R1b, R1c, u);
    fill_skew_matrix_dev(unorm_skew, unorm);

    *sin_theta = sinf(theta);
    *cos_inv_sum = 1.0f-cosf(theta);


    *theta_inv2 = 1.0f/(theta*theta);

    multiply_scalar_matrix3x3(u+0, R1a);
    multiply_scalar_matrix3x3(u+1, R1b);
    multiply_scalar_matrix3x3(u+2, R1c);

    float unorm_skew2[9];

    multiplyAB_3x3(unorm_skew, unorm_skew, unorm_skew2);


    R[0] = IR[0] = (*cos_inv_sum)*unorm_skew2[0] + 1.0f + (*sin_theta)*unorm_skew[0];
    R[1] = IR[1] = (*cos_inv_sum)*unorm_skew2[1]         + (*sin_theta)*unorm_skew[1];
    R[2] = IR[2] = (*cos_inv_sum)*unorm_skew2[2]         + (*sin_theta)*unorm_skew[2];

    R[3] = IR[3] = (*cos_inv_sum)*unorm_skew2[3]         + (*sin_theta)*unorm_skew[3];
    R[4] = IR[4] = (*cos_inv_sum)*unorm_skew2[4] + 1.0f + (*sin_theta)*unorm_skew[4];
    R[5] = IR[5] = (*cos_inv_sum)*unorm_skew2[5]         + (*sin_theta)*unorm_skew[5];

    R[6] = IR[6] = (*cos_inv_sum)*unorm_skew2[6]        + (*sin_theta)*unorm_skew[6];
    R[7] = IR[7] = (*cos_inv_sum)*unorm_skew2[7]        + (*sin_theta)*unorm_skew[7];
    R[8] = IR[8] = (*cos_inv_sum)*unorm_skew2[8] + 1.0f + (*sin_theta)*unorm_skew[8];


    float minusone = -1.0f;
    multiply_scalar_matrix3x3(&minusone, IR);
    IR[0] += 1.0f;
    IR[4] += 1.0f;
    IR[8] += 1.0f;


    R2a[0] = u[0]*IR[0];
    R2a[1] = u[0]*IR[1];
    R2a[2] = u[0]*IR[2];
    R2a[3] = u[1]*IR[0];
    R2a[4] = u[1]*IR[1];
    R2a[5] = u[1]*IR[2];
    R2a[6] = u[2]*IR[0];
    R2a[7] = u[2]*IR[1];
    R2a[8] = u[2]*IR[2];

    R2b[0] = u[0]*IR[3];
    R2b[1] = u[0]*IR[4];
    R2b[2] = u[0]*IR[5];
    R2b[3] = u[1]*IR[3];
    R2b[4] = u[1]*IR[4];
    R2b[5] = u[1]*IR[5];
    R2b[6] = u[2]*IR[3];
    R2b[7] = u[2]*IR[4];
    R2b[8] = u[2]*IR[5];

    R2c[0] = u[0]*IR[6];
    R2c[1] = u[0]*IR[7];
    R2c[2] = u[0]*IR[8];
    R2c[3] = u[1]*IR[6];
    R2c[4] = u[1]*IR[7];
    R2c[5] = u[1]*IR[8];
    R2c[6] = u[2]*IR[6];
    R2c[7] = u[2]*IR[7];
    R2c[8] = u[2]*IR[8];


    sum_AB_3x3_inplace(R2a, R1a);
    sum_AB_3x3_inplace(R2b, R1b);
    sum_AB_3x3_inplace(R2c, R1c);

    subtract_AtransB_3x3(R1a, R2a, R3a);
    subtract_AtransB_3x3(R1b, R2b, R3b);
    subtract_AtransB_3x3(R1c, R2c, R3c);

    multiplyAB_3x3(R3a, R, dR_u1);
    multiplyAB_3x3(R3b, R, dR_u2);
    multiplyAB_3x3(R3c, R, dR_u3);

    multiply_scalar_matrix3x3(theta_inv2, dR_u1);
    multiply_scalar_matrix3x3(theta_inv2, dR_u2);
    multiply_scalar_matrix3x3(theta_inv2, dR_u3);

}




__global__ void fill_theta_etc_without_derivatives(const float *u, float *theta_inv2, float *sin_theta, float *cos_inv_sum,
                                                   float *u_skew, float *unorm_skew, float *R, float *R1a, float *R1b, float *R1c)
{
    float theta = sqrtf(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

    float unorm[3];
    unorm[0] = u[0]/theta;
    unorm[1] = u[1]/theta;
    unorm[2] = u[2]/theta;

    fill_skew_matrix_dev4(u_skew, R1a, R1b, R1c, u);
    fill_skew_matrix_dev(unorm_skew, unorm);

    *sin_theta = sinf(theta);
    *cos_inv_sum = 1.0f-cosf(theta);

    *theta_inv2 = 1.0f/(theta*theta);

    multiply_scalar_matrix3x3(u+0, R1a);
    multiply_scalar_matrix3x3(u+1, R1b);
    multiply_scalar_matrix3x3(u+2, R1c);

    float unorm_skew2[9];

    multiplyAB_3x3(unorm_skew, unorm_skew, unorm_skew2);

    R[0] = (*cos_inv_sum)*unorm_skew2[0] + 1.0f + (*sin_theta)*unorm_skew[0];
    R[1] = (*cos_inv_sum)*unorm_skew2[1]     	+ (*sin_theta)*unorm_skew[1];
    R[2] = (*cos_inv_sum)*unorm_skew2[2]     	+ (*sin_theta)*unorm_skew[2];

    R[3] = (*cos_inv_sum)*unorm_skew2[3]     	+ (*sin_theta)*unorm_skew[3];
    R[4] = (*cos_inv_sum)*unorm_skew2[4] + 1.0f + (*sin_theta)*unorm_skew[4];
    R[5] = (*cos_inv_sum)*unorm_skew2[5]     	+ (*sin_theta)*unorm_skew[5];

    R[6] = (*cos_inv_sum)*unorm_skew2[6]     	+ (*sin_theta)*unorm_skew[6];
    R[7] = (*cos_inv_sum)*unorm_skew2[7]     	+ (*sin_theta)*unorm_skew[7];
    R[8] = (*cos_inv_sum)*unorm_skew2[8] + 1.0f + (*sin_theta)*unorm_skew[8];
}




__global__ void fill_skew_matrix(float *skewmat, const float *u)
{
    skewmat[0] = 0.0f;
    skewmat[1] = u[2];
    skewmat[2] = -u[1];

    skewmat[3] = -u[2];
    skewmat[4] = 0.0f;
    skewmat[5] = u[0];

    skewmat[6] = u[1];
    skewmat[7] = -u[0];
    skewmat[8] = 0.0f;
}


__device__ void fill_skew_matrix_dev(float *skewmat, const float *u)
{
    skewmat[0] = 0.0f;
    skewmat[1] = u[2];
    skewmat[2] = -u[1];

    skewmat[3] = -u[2];
    skewmat[4] = 0.0f;
    skewmat[5] = u[0];

    skewmat[6] = u[1];
    skewmat[7] = -u[0];
    skewmat[8] = 0.0f;
}


__device__ void fill_skew_matrix_dev2(float *sm1, float *sm2, const float *u)
{
    sm1[0] = sm2[0] = 0.0f;
    sm1[1] = sm2[1] = u[2];
    sm1[2] = sm2[2] = -u[1];

    sm1[3] = sm2[3] = -u[2];
    sm1[4] = sm2[4] = 0.0f;
    sm1[5] = sm2[5] = u[0];

    sm1[6] = sm2[6] = u[1];
    sm1[7] = sm2[7] = -u[0];
    sm1[8] = sm2[8] = 0.0f;
}


__device__ void fill_skew_matrix_dev4(float *sm1, float *sm2, float *sm3, float *sm4, const float *u)
{
    sm1[0] = sm2[0] = sm3[0] = sm4[0] = 0.0f;
    sm1[1] = sm2[1] = sm3[1] = sm4[1] = u[2];
    sm1[2] = sm2[2] = sm3[2] = sm4[2] = -u[1];

    sm1[3] = sm2[3] = sm3[3] = sm4[3] = -u[2];
    sm1[4] = sm2[4] = sm3[4] = sm4[4] = 0.0f;
    sm1[5] = sm2[5] = sm3[5] = sm4[5] = u[0];

    sm1[6] = sm2[6] = sm3[6] = sm4[6] = u[1];
    sm1[7] = sm2[7] = sm3[7] = sm4[7] = -u[0];
    sm1[8] = sm2[8] = sm3[8] = sm4[8] = 0.0f;
}


__device__ void sum_AB_3x3(const float* A, const float* B, float *C)
{
    C[0] = A[0] + B[0];
    C[1] = A[1] + B[1];
    C[2] = A[2] + B[2];

    C[3] = A[3] + B[3];
    C[4] = A[4] + B[4];
    C[5] = A[5] + B[5];

    C[6] = A[6] + B[6];
    C[7] = A[7] + B[7];
    C[8] = A[8] + B[8];
}



__device__ void subtract_AtransB_3x3(const float* A, const float* B, float *C)
{
    C[0] = A[0] - B[0];
    C[1] = A[1] - B[3];
    C[2] = A[2] - B[6];
    C[3] = A[3] - B[1];
    C[4] = A[4] - B[4];
    C[5] = A[5] - B[7];
    C[6] = A[6] - B[2];
    C[7] = A[7] - B[5];
    C[8] = A[8] - B[8];
}




__device__ void sum_AB_3x3_inplace(const float* A, float* B)
{
    B[0] += A[0];
    B[1] += A[1];
    B[2] += A[2];

    B[3] += A[3];
    B[4] += A[4];
    B[5] += A[5];

    B[6] += A[6];
    B[7] += A[7];
    B[8] += A[8];
}


__device__ void multiply_scalar_matrix3x3(const float* scalar, float *A)
{
    A[0] *= *scalar;
    A[1] *= *scalar;
    A[2] *= *scalar;
    A[3] *= *scalar;
    A[4] *= *scalar;
    A[5] *= *scalar;
    A[6] *= *scalar;
    A[7] *= *scalar;
    A[8] *= *scalar;
}


__global__ void eye3(float *I)
{
    for (ushort x=0; x<9; ++x)
    {
        ushort i = x % 3;
        ushort j = x/3;

        if (i == j)
            I[x] = 1.0f;
        else
            I[x] = 0.0f;
    }


}












RotationComputer::RotationComputer(float *_u_ptr) :
    size3x3( sizeof(float)*9 ), size3x1( sizeof(float)*3 ), size1x1( sizeof(float) ), u_ptr(_u_ptr)
{
    cudaMalloc((void **) &R1a, size3x3);
    cudaMalloc((void **) &R1b, size3x3);
    cudaMalloc((void **) &R1c, size3x3);
    cudaMalloc((void **) &R2a, size3x3);
    cudaMalloc((void **) &R2b, size3x3);
    cudaMalloc((void **) &R2c, size3x3);
    cudaMalloc((void **) &R3a, size3x3);
    cudaMalloc((void **) &R3b, size3x3);
    cudaMalloc((void **) &R3c, size3x3);

    cudaMalloc((void **) &dR_du1, size3x3);
    cudaMalloc((void **) &dR_du2, size3x3);
    cudaMalloc((void **) &dR_du3, size3x3);

    cudaMalloc((void **) &u_skew, size3x3);
    cudaMalloc((void **) &unorm_skew, size3x3);

    cudaMalloc((void **) &u, size3x1);
    cudaMalloc((void **) &unorm, size3x1);

    cudaMalloc((void **) &theta_inv2, size1x1);
    cudaMalloc((void **) &sin_theta, size1x1);
    cudaMalloc((void **) &cos_inv_sum, size1x1);
    cudaMalloc((void **) &scalar_one, size1x1);
    cudaMalloc((void **) &scalar_zero, size1x1);
    cudaMalloc((void **) &scalar_minusone, size1x1);

    cudaMalloc((void **) &I, size3x3);
    cudaMalloc((void **) &R, size3x3);
    cudaMalloc((void **) &IR, size3x3);


    //	    float c_u[3] = {1,0,0};

    //        cudaMemcpy(u, c_u, size3x1, cudaMemcpyHostToDevice);


    float c_u_skew[9];

    float c_scalar_one = 1.0f;
    float c_scalar_zero = 0.0f;
    float c_scalar_minusone = -1.0f;
    cudaMemcpy(scalar_one, 			&c_scalar_one, size1x1, cudaMemcpyHostToDevice);
    cudaMemcpy(scalar_zero, 		&c_scalar_zero, size1x1, cudaMemcpyHostToDevice);
    cudaMemcpy(scalar_minusone,	 	&c_scalar_minusone, size1x1, cudaMemcpyHostToDevice);



    cudaMemcpy(c_u_skew, u_skew, size3x3, cudaMemcpyDeviceToHost);


    eye3<<<1,1>>>(I);


    /*
    for (size_t i=0; i<3; ++i) {
        for (size_t j=0; j<3; ++j) {
            std::cout << c_u_skew[j*3+i] << ' ';
        }
        std::cout << std::endl;
    }

    */
}


#include <iostream>

void RotationComputer::compute_euler_angles(float& yaw, float &pitch, float &roll)
{
    float h_R[9];

    cudaMemcpy(&h_R, R, 9*sizeof(float), cudaMemcpyDeviceToHost);

    //    float cosine_for_pitch = sqrt(h_R[0]*h_R[0] + h_R[1]*h_R[1]);
    float cosine_for_pitch = sqrt(h_R[5]*h_R[5] + h_R[8]*h_R[8]);
    bool is_singular = cosine_for_pitch < 1e-6;

    if (!is_singular)
    {
        roll = atan2(h_R[1], h_R[0]);
        yaw = atan2(-h_R[2], cosine_for_pitch);
        pitch = atan2(h_R[5], h_R[8]);
        //	    std::cout << yaw << '\t' << pitch << '\t' << roll << std::endl;
    }
    else
    {
        roll= atan2(-h_R[7], h_R[4]);
        yaw  = atan2(-h_R[2], cosine_for_pitch);
        pitch = 0.0f;
        //	    std::cout << yaw << '\t' << pitch << '\t' << roll << std::endl;
    }

    if (isnan(yaw)) {
        //print_vector(R, 9, "RRRR");
    }
}

void RotationComputer::compute_angle_idx(const float yaw, const float pitch, const float roll, int &yaw_idx, int& pitch_idx, int  &roll_idx)
{
    yaw_idx = -1;
    pitch_idx = -1;
    roll_idx = -1;

    double angle_step = ANGLE_STEP;
    double angle_min = ANGLE_MIN;
    double angle_max = ANGLE_MAX;

    uint num_angles = (int) (angle_max-angle_min)/angle_step;

    double yaw_deg = RAD2DEG(yaw);
    double pitch_deg = RAD2DEG(pitch);
    double roll_deg = RAD2DEG(roll);

    for (uint i=0; i<num_angles; ++i) {
        if (yaw_deg < angle_min + i*angle_step) {
            yaw_idx = i-1;
            break;
        }
    }

    for (uint i=0; i<num_angles; ++i) {
        if (pitch_deg < angle_min + i*angle_step) {
            pitch_idx = i-1;
            break;
        }
    }

    for (uint i=0; i<num_angles; ++i) {
        if (roll_deg < angle_min + i*angle_step) {
            roll_idx = i-1;
            break;
        }
    }

    if (!isnan(yaw) && !isnan(pitch) && !isnan(roll)) {
        if (yaw_idx == -1) {
            yaw_idx = 0;
            pitch_idx = 0;
            roll_idx = 0;
        }
    }
}




void RotationComputer::process()
{
    fill_theta_etc<<<1,1>>>(u_ptr, theta_inv2, sin_theta, cos_inv_sum, u_skew, unorm_skew, R, IR, R1a, R1b, R1c, R2a, R2b, R2c, R3a, R3b, R3c, dR_du1, dR_du2, dR_du3);
}



void RotationComputer::process_without_derivatives()
{
    fill_theta_etc_without_derivatives<<<1,1>>>(u_ptr, theta_inv2, sin_theta, cos_inv_sum, u_skew, unorm_skew, R, R1a, R1b, R1c);
}

void RotationComputer::set_u_ptr(float * _u_ptr)
{
    u_ptr = _u_ptr;
}


RotationComputer::~RotationComputer()
{
    cudaFree(scalar_minusone);
    cudaFree(scalar_zero);
    cudaFree(scalar_one);
    cudaFree(IR);
    cudaFree(R);

    cudaFree(R1a);
    cudaFree(R1b);
    cudaFree(R1c);
    cudaFree(R2a);
    cudaFree(R2b);
    cudaFree(R2c);
    cudaFree(R3a);
    cudaFree(R3b);
    cudaFree(R3c);

    cudaFree(dR_du1);
    cudaFree(dR_du2);
    cudaFree(dR_du3);

    cudaFree(theta_inv2);
    cudaFree(sin_theta);
    cudaFree(cos_inv_sum);


    cudaFree(u_skew);
    cudaFree(unorm_skew);

    cudaFree(u);
    cudaFree(unorm);


    cudaFree(I);
}



























