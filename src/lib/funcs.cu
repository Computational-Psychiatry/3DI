/*
 * funcs.cu
 *
 *  Created on: Aug 8, 2020
 *      Author: root
 */

#include "funcs.h"
#include <vector>
#include <string>
#include <sstream>
#include <numeric>


/*
extern texture<float,2> EX_texture;
extern texture<float,2> EY_texture;
extern texture<float,2> EZ_texture;
*/

using std::string;

std::string base_name(const std::string& path, const std::string& delims)
{
    return path.substr(path.find_last_of(delims) + 1);
}


std::string remove_extension(const std::string &filename)
{
    typename std::string::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != std::string::npos ? filename.substr(0, p) : filename;
}

bool is_float(const std::string &myString ) {
    std::istringstream iss(myString);
    float f;
    iss >> std::noskipws >> f; // noskipws considers leading whitespace invalid
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail();
}


float compute_face_size(const float* xp, const float* yp)
{
    float minx = *std::min_element(xp, xp + NLANDMARKS_51);
    float miny = *std::min_element(yp, yp + NLANDMARKS_51);

    float maxx = *std::max_element(xp, xp + NLANDMARKS_51);
    float maxy = *std::max_element(yp, yp + NLANDMARKS_51);

    return std::max<float>(maxx-minx, maxy-miny);
}

float compute_face_diagonal(const float* xp, const float* yp)
{
    float minx = *std::min_element(xp, xp + NLANDMARKS_51);
    float miny = *std::min_element(yp, yp + NLANDMARKS_51);

    float maxx = *std::max_element(xp, xp + NLANDMARKS_51);
    float maxy = *std::max_element(yp, yp + NLANDMARKS_51);

    return std::sqrt((maxx-minx)*(maxx-minx)+(maxy-miny)*(maxy-miny));
}

vector<string> str_split (const string &s, char delim) {
    vector<string> result;
    std::stringstream ss (s);
    string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}




__global__ void fill_ks_and_M0(
        const ushort *pixel_idx,
        const uint* redundant_idx,
        ushort *ks,
        ushort *ks_unsorted,
        ushort *M0,
        const uint N_unique_pixels)
{
    // The number of pixels that we'll render is N_unique_pixels
    ////////////////////////const uint n = blockIdx.x;


    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    const uint n = colix + rowix*blockDim.x;

    if (n >= N_unique_pixels)
        return;

    ks[n] = pixel_idx[redundant_idx[n]];

    ks_unsorted[n] = ks[n];

    M0[ks[n]] = 1;
}



__global__ void fill_ksides(
        const ushort *ks,
        ushort* ks_left, ushort* ks_right, ushort* ks_above, ushort* ks_below)
{
    // The number of pixels that we'll render is N_unique_pixels
    ////////////const int n = blockIdx.x;


    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    uint n = colix + rowix*blockDim.x;

    ushort y = ks[n] % (DIMY);
    ushort x = ks[n]/(DIMY);


    ks_right[n] = (x < DIMX-1) ? (x+1)*(DIMY) + y : 0;
    ks_left[n]  = (x > 0) ? (x-1)*(DIMY) + y : 0;
    ks_above[n] = (y > 0) ? x*(DIMY) + y-1 : 0;
    ks_below[n] = (y < DIMY-1) ? x*(DIMY) + y+1 : 0;
}


__global__ void reset_bool_flag(bool *flag, int N)
{
    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    uint n = colix + rowix*blockDim.x;

    if (n >= N)
        return;

    flag[n] = false;
}


__global__ void reset_ushort_array(ushort *flag, int N)
{
    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    uint n = colix + rowix*blockDim.x;

    if (n >= N)
        return;

    flag[n] = false;
}




__global__ void render_texture_basis_texture_via_object(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int N1, const ushort* __restrict__ tl,
        float* __restrict__ RTEX, const ushort* __restrict__ triangle_idx,
        const ushort Kbeta, const int N_TRIANGLES, cudaTextureObject_t tex_tex)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    const int rel_index = indices[rowix];

    //! Important! We fill REX, ... in a ROW-MAJOR order. This way it will be easier to extract a submatrix of REX that ignores the bottom of REX
    const int idx = threadIdx.x + Kbeta*blockIdx.x;

    const int tl_i1 = triangle_idx[rel_index];
    const int tl_i2 = tl_i1 + N_TRIANGLES;
    const int tl_i3 = tl_i2 + N_TRIANGLES;

    RTEX[idx] = tex2D<float>(tex_tex,colix,tl[tl_i1])*alphas[rel_index] + tex2D<float>(tex_tex,colix,tl[tl_i2])*betas[rel_index] + tex2D<float>(tex_tex,colix,tl[tl_i3])*gammas[rel_index];
}



__global__ void render_identity_basis_texture_via_object(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int N1, const ushort* __restrict__ tl,
        float* __restrict__ RIX, float* __restrict__ RIY, float* __restrict__ RIZ,
        const ushort* __restrict__ triangle_idx, const ushort Kalpha, const int N_TRIANGLES,
        cudaTextureObject_t ix_tex, cudaTextureObject_t iy_tex, cudaTextureObject_t iz_tex)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    const int rel_index = indices[rowix];

    //! Important! We fill REX, ... in a ROW-MAJOR order. This way it will be easier to extract a submatrix of REX that ignores the bottom of REX
    const int idx = threadIdx.x + Kalpha*blockIdx.x;

    const int tl_i1 = triangle_idx[rel_index];
    const int tl_i2 = tl_i1 + N_TRIANGLES;
    const int tl_i3 = tl_i2 + N_TRIANGLES;

    RIX[idx] = tex2D<float>(ix_tex,colix,tl[tl_i1])*alphas[rel_index] + tex2D<float>(ix_tex,colix,tl[tl_i2])*betas[rel_index] + tex2D<float>(ix_tex,colix,tl[tl_i3])*gammas[rel_index];
    RIY[idx] = tex2D<float>(iy_tex,colix,tl[tl_i1])*alphas[rel_index] + tex2D<float>(iy_tex,colix,tl[tl_i2])*betas[rel_index] + tex2D<float>(iy_tex,colix,tl[tl_i3])*gammas[rel_index];
    RIZ[idx] = tex2D<float>(iz_tex,colix,tl[tl_i1])*alphas[rel_index] + tex2D<float>(iz_tex,colix,tl[tl_i2])*betas[rel_index] + tex2D<float>(iz_tex,colix,tl[tl_i3])*gammas[rel_index];
}














__global__ void render_expression_basis_texture_colmajor_rotated2_via_object(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int Nunique_pixels, const ushort* __restrict__ tl,
        float*  __restrict__ REX, float*  __restrict__ REY, float*  __restrict__ REZ,
        const ushort* __restrict__ triangle_idx, const float* R__,
        int N_TRIANGLES, uint Nredundant,
        cudaTextureObject_t ex_tex, cudaTextureObject_t ey_tex, cudaTextureObject_t ez_tex)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;
    __shared__ float R[9];

    if (colix < 9) {
        R[colix] = R__[colix];
    }
    __syncthreads();

    const float R00 = R[0]; const float R10 = R[1]; const float R20 = R[2];
    const float R01 = R[3]; const float R11 = R[4]; const float R21 = R[5];
    const float R02 = R[6]; const float R12 = R[7]; const float R22 = R[8];

    const int rel_index = indices[rowix];

    const int idx = threadIdx.x*Nredundant + blockIdx.x;

    const int tl_i1 = triangle_idx[rel_index];
    const int tl_i2 = tl_i1 + N_TRIANGLES;
    const int tl_i3 = tl_i2 + N_TRIANGLES;


    //if (tl_i1 >= Nunique_pixels || tl_i2 >= Nunique_pixels || tl_i3 >= Nunique_pixels)
    //    return;

    /*
    const float tmpx = tex2D(EX_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EX_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EX_texture,colix,tl[tl_i3])*gammas[rel_index];
    const float tmpy = tex2D(EY_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EY_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EY_texture,colix,tl[tl_i3])*gammas[rel_index];
    const float tmpz = tex2D(EZ_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EZ_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EZ_texture,colix,tl[tl_i3])*gammas[rel_index];
*/
    const float tmpx = tex2D<float>(ex_tex,colix,tl[tl_i1])*alphas[rel_index] + tex2D<float>(ex_tex,colix,tl[tl_i2])*betas[rel_index] + tex2D<float>(ex_tex,colix,tl[tl_i3])*gammas[rel_index];
    const float tmpy = tex2D<float>(ey_tex,colix,tl[tl_i1])*alphas[rel_index] + tex2D<float>(ey_tex,colix,tl[tl_i2])*betas[rel_index] + tex2D<float>(ey_tex,colix,tl[tl_i3])*gammas[rel_index];
    const float tmpz = tex2D<float>(ez_tex,colix,tl[tl_i1])*alphas[rel_index] + tex2D<float>(ez_tex,colix,tl[tl_i2])*betas[rel_index] + tex2D<float>(ez_tex,colix,tl[tl_i3])*gammas[rel_index];


    REX[idx] = tmpx*R00 + tmpy*R01 + tmpz*R02;
    REY[idx] = tmpx*R10 + tmpy*R11 + tmpz*R12;
    REZ[idx] = tmpx*R20 + tmpy*R21 + tmpz*R22;
}

















__global__ void fill_krels(
        const uint N_pixels,
        const ushort *ks,
        const ushort *ks_sortidx,
        const ushort *ks_sortidx_sortidx,
        const ushort* ks_left, 	const ushort* ks_right, const ushort* ks_above, const ushort* ks_below,
        ushort* kl_rel, 	ushort* kr_rel, 	ushort* ka_rel, 	ushort* kb_rel,
        ushort* kl_rel_sorted, 	ushort* kr_rel_sorted, 	ushort* ka_rel_sorted, 	ushort* kb_rel_sorted,
        const ushort *cumM0)
{
    // The number of pixels that we'll render is N_unique_pixels
    //	const ushort n = blockIdx.x;


    const ushort rowix = blockIdx.x;
    const ushort colix = threadIdx.x;

    const ushort n = colix + rowix*blockDim.x;



    if (n == 0 || n == N_pixels-1)
        return;

    kl_rel_sorted[n] = (n > (cumM0[ks[n]] - cumM0[ks_left[n]])) ? (n - (cumM0[ks[n]] - cumM0[ks_left[n]])) : 0;
    kr_rel_sorted[n] = n + cumM0[ks_right[n]] - cumM0[ks[n]];

    ka_rel_sorted[n] = n - 1;
    kb_rel_sorted[n] = n + 1;

    if (kl_rel_sorted[n] >= N_pixels-1)
        kl_rel_sorted[n] = N_pixels-1;
}



__global__ void fill_krels2(
        const ushort *ks,
        const ushort *ks_sortidx,
        const ushort *ks_sortidx_sortidx,
        ushort* kl_rel, 	ushort* kr_rel, 	ushort* ka_rel, 	ushort* kb_rel,
        const ushort* kl_rel_sorted, 	const ushort* kr_rel_sorted, 	const ushort* ka_rel_sorted, 	const ushort* kb_rel_sorted)
{
    const ushort rowix = blockIdx.x;
    const ushort colix = threadIdx.x;

    ushort n = colix + rowix*blockDim.x;

    kl_rel[n] = ks_sortidx[kl_rel_sorted[ks_sortidx_sortidx[n]]];
    kr_rel[n] = ks_sortidx[kr_rel_sorted[ks_sortidx_sortidx[n]]];
    ka_rel[n] = ks_sortidx[ka_rel_sorted[ks_sortidx_sortidx[n]]];
    kb_rel[n] = ks_sortidx[kb_rel_sorted[ks_sortidx_sortidx[n]]];
}






__global__ void rotate_3d_pts(float* X, float* Y, float* Z, const float* R__, const int NPTS)
{
    const ushort rowix = blockIdx.x;
    const ushort colix = threadIdx.x;

    ushort n = colix + rowix*blockDim.x;

    if (n >= NPTS)
        return;

    __shared__ float R[9];
    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float tauz[1];
    __shared__ float phix[1];
    __shared__ float phiy[1];
    __shared__ float cx[1];
    __shared__ float cy[1];

    if (colix < 9)
    {
        R[colix] = R__[colix];
    }

    __syncthreads();

    const float R00 = R[0];
    const float R10 = R[1];
    const float R20 = R[2];
    const float R01 = R[3];
    const float R11 = R[4];
    const float R21 = R[5];
    const float R02 = R[6];
    const float R12 = R[7];
    const float R22 = R[8];

    float xtmp = R00*X[n] + R01*Y[n] + R02*Z[n];
    float ytmp = R10*X[n] + R11*Y[n] + R12*Z[n];
    float ztmp = R20*X[n] + R21*Y[n] + R22*Z[n];

    X[n] = xtmp;
    Y[n] = ytmp;
    Z[n] = ztmp;
}




__global__ void view_transform_3d_pts_and_render_2d(const float* X0, const float* Y0, const float* Z0,
                                                    const float* R__, const float *taux__, const float *tauy__, const float *tauz__,
                                                    const float* phix__, const float* phiy__, const float* cx__, const float* cy__,
                                                    float *X, float *Y, float *Z, float *xp, float *yp, const int NPTS)
{
    const ushort rowix = blockIdx.x;
    const ushort colix = threadIdx.x;

    ushort n = colix + rowix*blockDim.x;

    if (n >= NPTS)
        return;

    __shared__ float R[9];
    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float tauz[1];
    __shared__ float phix[1];
    __shared__ float phiy[1];
    __shared__ float cx[1];
    __shared__ float cy[1];

    if (colix < 9)
    {
        R[colix] = R__[colix];

        if (colix == 0) {
            taux[0] = taux__[0];
            tauy[0] = tauy__[0];
            tauz[0] = tauz__[0];
        }

        if (colix == 1) {
            phix[0] = phix__[0];
            phiy[0] = phiy__[0];
            cx[0] = cx__[0];
            cy[0] = cy__[0];
        }
    }

    __syncthreads();

    const float R00 = R[0];
    const float R10 = R[1];
    const float R20 = R[2];
    const float R01 = R[3];
    const float R11 = R[4];
    const float R21 = R[5];
    const float R02 = R[6];
    const float R12 = R[7];
    const float R22 = R[8];

    float X3 = X[n] = R00*X0[n] + R01*Y0[n] + R02*Z0[n] + taux[0];
    float Y3 = Y[n] = R10*X0[n] + R11*Y0[n] + R12*Z0[n] + tauy[0];
    float Z3 = Z[n] = R20*X0[n] + R21*Y0[n] + R22*Z0[n] + tauz[0];

    xp[n] = phix[0]*X3/Z3 + cx[0];
    yp[n] = phiy[0]*Y3/Z3 + cy[0];
}


__global__ void set_xtmp(const float* search_dir, const float *x,  float t_coef__, float *xtmp, const uint Ktotal)
{

    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    const uint n = colix + rowix*blockDim.x;

    if (n >= Ktotal)
        return;

    __shared__ float t_coef[1];

    if (colix == 0)
        t_coef[0] = t_coef__;

    __syncthreads();

    xtmp[n] = x[n] + t_coef[0]*search_dir[n];
    /*
    if (n == 0)
    {
        for (uint i=0; i<6; ++i)
        {
            printf("%.2f\n", xtmp[i]);
        }
    }
*/
}





void print_vector( void *d_vec, int Nsize, const std::string& title)
{
    if (title.size() > 0) {
        std:: cout << title << " :::";

        if (Nsize > 1) {
            std::cout << std::endl;
        }
    }
    float *h_vec;
    h_vec = (float*)malloc( Nsize*sizeof(float) );

    cudaMemcpy(h_vec, d_vec, sizeof(float)*Nsize, cudaMemcpyDeviceToHost);

    for (uint i=0; i< Nsize; ++i) {
        //		printf("%.4f ", h_vec[i]);
        std::cout << h_vec[i] << ' ';

    }
    //	printf("\n");
    std::cout << std::endl;
    free(h_vec);
}



void print_vector_double( void *d_vec, int Nsize, const std::string& title)
{
    if (title.size() > 0) {
        std:: cout << title << " :::";

        if (Nsize > 1) {
            std::cout << std::endl;
        }
    }
    double *h_vec;
    h_vec = (double*)malloc( Nsize*sizeof(double) );

    cudaMemcpy(h_vec, d_vec, sizeof(double)*Nsize, cudaMemcpyDeviceToHost);

    for (uint i=0; i< Nsize; ++i) {
        //		printf("%.4f ", h_vec[i]);
        std::cout << h_vec[i] << ' ';

    }
    //	printf("\n");
    std::cout << std::endl;
    free(h_vec);
}






void print_vector_bool( void *d_vec, int Nsize, const std::string& title)
{
    std:: cout << title << " :::" << std::endl;
    bool *h_vec;
    h_vec = (bool*)malloc( Nsize*sizeof(bool) );

    cudaMemcpy(h_vec, d_vec, sizeof(bool)*Nsize, cudaMemcpyDeviceToHost);

    for (uint i=0; i< Nsize; ++i) {
        printf("%d", h_vec[i]);
    }
    printf("\n");
    free(h_vec);
}





void print_vector_uint( void *d_vec, int Nsize, const std::string& title)
{
    std:: cout << title << " :::" << std::endl;
    uint *h_vec;
    h_vec = (uint*)malloc( Nsize*sizeof(uint) );

    cudaMemcpy(h_vec, d_vec, sizeof(uint)*Nsize, cudaMemcpyDeviceToHost);

    for (uint i=0; i< Nsize; ++i) {
        printf("%d ", h_vec[i]);
    }
    printf("\n");
    free(h_vec);
}




bool check_if_bb_OK(float *xp, float *yp)
{
    float *x0f = thrust::min_element(thrust::host, xp, xp+NLANDMARKS_51);
    float *y0f = thrust::min_element(thrust::host, yp, yp+NLANDMARKS_51);

    float *xff = thrust::max_element(thrust::host, xp, xp+NLANDMARKS_51);
    float *yff = thrust::max_element(thrust::host, yp, yp+NLANDMARKS_51);

    float xsize = *xff-*x0f;
    float ysize = *yff-*y0f;


    bool is_bb_ok = true;

    if (config::PAD_RENDERING)
    {
        if (*x0f-xsize/2 <= 0) {
            is_bb_ok = false;
            //        std::cout << "failed x test" << std::endl;
        }

        if (*y0f-ysize/2 <= 0) {
            //        std::cout << "failed y test" << std::endl;
            is_bb_ok = false;
        }
    }

    return is_bb_ok;
}



bool check_if_face_in_frame(float *xp, float *yp, int imcols, int imrows)
{
    float *x0f = thrust::min_element(thrust::host, xp, xp+NLANDMARKS_51);
    float *y0f = thrust::min_element(thrust::host, yp, yp+NLANDMARKS_51);

    float *xff = thrust::max_element(thrust::host, xp, xp+NLANDMARKS_51);
    float *yff = thrust::max_element(thrust::host, yp, yp+NLANDMARKS_51);
    
    if (*x0f < 0 || *xff >= imcols || *y0f < 0 || *yff >= imrows)
        return false;
    
    return true;
}

void print_matrix( void *d_A, int Nrows, int Ncols)
{
    float *h_A;
    h_A = (float*)malloc( Nrows*Ncols*sizeof(float) );

    cudaMemcpy(h_A, d_A, sizeof(float)*Nrows*Ncols, cudaMemcpyDeviceToHost);

    for (uint i=0; i<Nrows; ++i) {
        for (uint j=0; j<Ncols; ++j) {
            printf("%.4f ", h_A[i+j*Nrows]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    free(h_A);
}





void print_matrix_double( void *d_A, int Nrows, int Ncols)
{
    double *h_A;
    h_A = (double*)malloc( Nrows*Ncols*sizeof(double) );

    cudaMemcpy(h_A, d_A, sizeof(double)*Nrows*Ncols, cudaMemcpyDeviceToHost);

    for (uint i=0; i<Nrows; ++i) {
        for (uint j=0; j<Ncols; ++j) {
            printf("%.4f ", h_A[i+j*Nrows]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    free(h_A);
}



void write_identity(const std::string& path, float* x, float *y, float *z)
{
    std::ofstream DataFile;
    DataFile.open(path.c_str());
    for (int i = 0; i < config::NPTS; i++) {
        DataFile << x[i] << ' ' << y[i] << ' ' << z[i] << std::endl;
    }
    DataFile.close();
}




void write_texture(const std::string& path, float* x)
{
    std::ofstream DataFile;
    DataFile.open(path.c_str());
    for (int i = 0; i < config::NPTS; i++) {
        DataFile << x[i] << std::endl;
    }
    DataFile.close();
}



void writeLandmarks(const std::string& path, std::vector<float>& x, std::vector<float>& y)
{

    std::ofstream DataFile;
    DataFile.open(path.c_str());
    for (int i = 0; i < NLANDMARKS_51; i++) {
        DataFile << x[i]-config::PAD_SINGLE_IMAGE << '\t' << y[i]-config::PAD_SINGLE_IMAGE << std::endl;
    }
    DataFile.close();
}


void write_matrix_to_file_uint( const void *d_A, int Nrows, int Ncols, const std::string &filepath)
{
    uint *h_A;
    h_A = (uint*)malloc( Nrows*Ncols*sizeof(uint) );

    cudaMemcpy(h_A, d_A, sizeof(uint)*Nrows*Ncols, cudaMemcpyDeviceToHost);


    writeArrFile<uint>(h_A, filepath, Nrows, Ncols);



    free(h_A);
}







void write_matrix_to_file_bool( const void *d_A, int Nrows, int Ncols, const std::string &filepath)
{
    bool *h_A;
    h_A = (bool*)malloc( Nrows*Ncols*sizeof(bool) );

    cudaMemcpy(h_A, d_A, sizeof(bool)*Nrows*Ncols, cudaMemcpyDeviceToHost);


    writeArrFile<bool>(h_A, filepath, Nrows, Ncols);



    free(h_A);
}





void write_matrix_to_file( const void *d_A, int Nrows, int Ncols, const std::string &filepath)
{
    float *h_A;
    h_A = (float*)malloc( Nrows*Ncols*sizeof(float) );

    cudaMemcpy(h_A, d_A, sizeof(float)*Nrows*Ncols, cudaMemcpyDeviceToHost);

    writeArrFile<float>(h_A, filepath, Nrows, Ncols);

    free(h_A);
}








void write_matrix_to_file_ushort( void *d_A, int Nrows, int Ncols, const std::string &filepath)
{
    ushort *h_A;
    h_A = (ushort*)malloc( Nrows*Ncols*sizeof(ushort) );

    cudaMemcpy(h_A, d_A, sizeof(ushort)*Nrows*Ncols, cudaMemcpyDeviceToHost);


    writeArrFile<ushort>(h_A, filepath, Nrows, Ncols);



    free(h_A);
}




__global__ void elementwise_vector_multiplication(float *vec_out, const float *vec_in, const int vec_len )
{

    // The number of pixels that we'll render is N_unique_pixels
    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    const uint n = colix + rowix*blockDim.x;

    if (n>=vec_len)
        return;

    vec_out[n] *= vec_in[n];
}




__global__ void fill_tex_im1(
        const ushort *ks,
        const float *tex,
        float *tex_im,
        const uint N_unique_pixels)
{
    // The number of pixels that we'll render is N_unique_pixels
    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    const uint n = colix + rowix*blockDim.x;

    if (n>=N_unique_pixels)
        return;


    tex_im[ks[n]] = tex[n];
}


__global__ void multiply_and_add_to_scalar(float *a, const float* alpha, const float *b)
{
    a[0] = alpha[0]*a[0]+b[0];
}



__global__ void add_to_scalar(float *a, const float *b)
{
    a[0] = a[0]+b[0];
}



__global__ void add_to_scalar_negated(float *a, const float *b)
{
    a[0] = a[0]+(-b[0]);
}




__global__ void fill_grads(
        const ushort *ks,
        const float *tex_im,
        const float *source_im,
        float *gx, float *gy, float *gx_norm, float *gy_norm, float *h,
        float *gxs, float *gys, float *gxs_norm, float *gys_norm, float *hs,
        const uint N_unique_pixels,
        const uint Nrender_estimated)
{
    // The number of pixels that we'll render is N_unique_pixels
    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    const uint n = colix + rowix*blockDim.x;

    if (n >= Nrender_estimated) {
        return;
    }

    if (n>=N_unique_pixels) {
        gx[n] = 0;
        gy[n] = 0;

        gxs[n] = 0;
        gys[n] = 0;

        gxs_norm[n] = 0;
        gys_norm[n] = 0;
        return;
    }

    ushort y = ks[n] % (DIMY);
    ushort x = ks[n]/(DIMY);

    ushort ks_right = (x < DIMX-1) ? (x+1)*(DIMY) + y : 0;
    ushort ks_left  = (x > 0) ? (x-1)*(DIMY) + y : 0;
    ushort ks_above = (y > 0) ? x*(DIMY) + y-1 : 0;
    ushort ks_below = (y < DIMY-1) ? x*(DIMY) + y+1 : 0;


    gx[n] = 0.5f*(tex_im[ks_right]-tex_im[ks_left]);
    gy[n] = 0.5f*(tex_im[ks_below]-tex_im[ks_above]);

    h[n] = 1e-16f + sqrtf(4*gx[n]*gx[n] + 4*gy[n]*gy[n]);

    gx_norm[n] = 2.0f*gx[n]/h[n];
    gy_norm[n] = 2.0f*gy[n]/h[n];

    gxs[n] = 0.5f*(source_im[ks_right]-source_im[ks_left]);
    gys[n] = 0.5f*(source_im[ks_below]-source_im[ks_above]);

    hs[n] = 1e-16f + sqrtf(4*gxs[n]*gxs[n] + 4*gys[n]*gys[n]);

    gxs_norm[n] = 2.0f*gxs[n]/hs[n];
    gys_norm[n] = 2.0f*gys[n]/hs[n];
}


__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}



/**
 * pixel_idx: is the 1d index of the pixel on the image. this is necessary for z-buffering
 *
 * This kernel processes triangles separately
 */
__global__ void get_pixels_to_render(const ushort *tl, const float *xp, const float *yp,
                                     bool *rend_flag, ushort *pixel_idx,
                                     float *alphas_redundant, float *betas_redundant, float *gammas_redundant,
                                     ushort* triangle_idx,
                                     const float *Z, float *Ztmp,
                                     const ushort x0, const ushort y0, float* Zmins, uint* redundant_idx,
                                     int N_TRIANGLES, uint Nredundant)
{
    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    /**
     * The parallelization for rendering is done in terms of the triangle. That is,
     * when the kernel runs, each node processes one triangle of the 3DMM.
     */
    const uint idx = colix + rowix*blockDim.x;

    if (idx >= Nredundant)
        return;

    redundant_idx[idx] = idx;

    if (idx >= N_TRIANGLES)
        return;

    float bx = xp[tl[idx]];
    float cx = xp[tl[N_TRIANGLES*1+idx]];
    float ax = xp[tl[N_TRIANGLES*2+idx]];

    float by = yp[tl[idx]];
    float cy = yp[tl[N_TRIANGLES*1+idx]];
    float ay = yp[tl[N_TRIANGLES*2+idx]];


    uint ridx = 0;
    for (ushort x=floorf(fmin(fmin(bx,cx),ax)); x<ceilf(fmax(fmax(bx,cx),ax)); ++x) {
        for (ushort y=floorf(fmin(fmin(by,cy),ay)); y<ceilf(fmax(fmax(by,cy),ay)); ++y) {

            /**
             * Below we are computing the barymetric coordinates, which will determine
             * if the pixel (x,y) is within the triangle tl[idx]. The barymetric coordinates will also
             * give us the correct mix for interpolation that is needed while rendering this pixel.
             *
             * This was probably taken from:
             * https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
             */
            float v0x = bx-ax;
            float v1x = cx-ax;
            float v2x = x-ax;

            float v0y = by-ay;
            float v1y = cy-ay;
            float v2y = y-ay;

            float d00 = v0x*v0x+v0y*v0y;
            float d01 = v0x*v1x+v0y*v1y;
            float d11 = v1x*v1x+v1y*v1y;
            float d20 = v0x*v2x+v0y*v2y;
            float d21 = v1x*v2x+v1y*v2y;

            float denom = d00*d11-d01*d01;
            float alpha = (d11*d20-d01*d21)/denom;
            float beta  = (d00*d21-d01*d20)/denom;

            if (alpha>0 && beta>0 && (alpha+beta<=1)) {
                if (ridx < NTMP)
                {
                    uint cidx = ridx*N_TRIANGLES+idx;

                    float gamma = 1-alpha-beta;

                    rend_flag[cidx] = true;

                    alphas_redundant[cidx] = alpha;
                    betas_redundant[cidx] = beta;
                    gammas_redundant[cidx] = gamma;

                    pixel_idx[cidx] = (y-y0)+(x-x0)*(DIMY);

//                    atomicAdd((unsigned int *)&cnt_per_pixel[pixel_idx[cidx]], 1);
                    atomicMinFloat(&Zmins[pixel_idx[cidx]], Z[tl[idx]]);

//                    atomicMinFloat()

                    triangle_idx[cidx] = idx;
                    Ztmp[cidx] = Z[tl[/*N_TRIANGLES*0+*/idx]];

                    ridx++;

                }
            } else {
                //				Ztmp[cidx] = INFINITY;
            }
        }
    }
}





__global__ void keep_only_minZ(const float *Zmins, const float* Zs, const ushort* pixel_idx, uint* redundant_idx, int N)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N)
        return;

//    if (fabsf(Zs[n]-Zmins[pixel_idx[n]]) >= 0.0000001f)
    if (Zs[n]!=Zmins[pixel_idx[n]])
        redundant_idx[n] = 0;
}










__global__ void fill_float_arr(float *arr, float val, int N)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N)
        return;

    arr[n] = val;
}


__global__ void populate_pixel_idx2(const ushort *pixel_idx, const uint* indices, ushort *pixel_idx2, const uint N1)
{

    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    const uint idx = colix + rowix*blockDim.x;

    if (idx >= N1)
        return;

    pixel_idx2[idx] = pixel_idx[indices[idx]];
}



__global__ void zbuffer_update(const ushort* sorted_pixel_idx,
                               const ushort* unique_pixel_locations,
                               const float *Ztmp,
                               const uint *indices,
                               const ushort *inner_indices,
                               bool* rend_flag_tmp, const uint Nredundant, const ushort N_unique_pixels, const uint N1)
{
    const uint rowix = blockIdx.x;
    const uint colix = threadIdx.x;

    const uint idx = colix + rowix*blockDim.x;

    if (idx >= N_unique_pixels)
        return;

    ///////////////const uint idx = blockIdx.x;
    ushort cur_pixel_idx_location = unique_pixel_locations[idx];
    ushort cur_pixel_idx = sorted_pixel_idx[ cur_pixel_idx_location ];

    if (cur_pixel_idx_location >= N1) {
        return;
    }

    float minz = Ztmp[indices[inner_indices[cur_pixel_idx_location]]];
    uint minz_idx = inner_indices[cur_pixel_idx_location];

    /*
    if (!rend_flag_tmp[inner_indices[cur_pixel_idx_location]]) {
        printf("ERROR! At location %d we have bool: %d for Z %.4f \n", blockIdx.x, rend_flag_tmp[inner_indices[cur_pixel_idx_location]],
                Ztmp[indices[inner_indices[cur_pixel_idx_location]]]);
    }
     */

    for (uint i=1; i<100; ++i) {
        if (cur_pixel_idx_location+i >= N1) {
            break;
        }
        if (inner_indices[cur_pixel_idx_location+i] >= Nredundant)
            break;
        if (sorted_pixel_idx[ cur_pixel_idx_location+i ] == cur_pixel_idx) {
            float z_tmp = Ztmp[indices[inner_indices[cur_pixel_idx_location+i]]];

            if (z_tmp < minz) {
                /*
                if (!rend_flag_tmp[inner_indices[cur_pixel_idx_location+i]])
                    printf("ERROR!\n");
                    */

                rend_flag_tmp[minz_idx] = false;
                minz = z_tmp;
                minz_idx = inner_indices[cur_pixel_idx_location+i];
            } else {
                /*
                if (!rend_flag_tmp[inner_indices[cur_pixel_idx_location+i]])
                    printf("ERROR!\n");
                */
                rend_flag_tmp[inner_indices[cur_pixel_idx_location+i]] = false;
            }
        }
        else
            break;
    }
}




void imshow_opencv(const float *d_im, const std::string win_name)
{
    cv::Mat srcMat(DIMX, DIMY, CV_32F);

    cudaMemcpy(srcMat.data, d_im, sizeof(float)*NTOTAL_PIXELS, cudaMemcpyDeviceToHost);

    double minSrc, maxSrc;
    cv::minMaxLoc(srcMat, &minSrc, &maxSrc);
    srcMat = (srcMat-minSrc)/(maxSrc-minSrc);

    cv::imshow(win_name,    srcMat.t());
}








void imshow_opencv_buffered(const float *d_im, const int x_offset, const int y_offset, const int width, const int height, const std::string win_name)
{
    cv::Mat rectMat(DIMX, DIMY, CV_32F);

    cv::Mat srcMat(height, width, CV_32F, cv::Scalar::all(0));

    cudaMemcpy(rectMat.data, d_im, sizeof(float)*NTOTAL_PIXELS, cudaMemcpyDeviceToHost);
    cv::transpose(rectMat, rectMat);

    rectMat.copyTo(srcMat(cv::Rect(x_offset, y_offset, DIMX, DIMY)));

    double minSrc, maxSrc;
    cv::minMaxLoc(srcMat, &minSrc, &maxSrc);
    srcMat = (srcMat-minSrc)/(maxSrc-minSrc);

    cv::imshow(win_name,    srcMat);
}






void create_cvmat_buffered(const float *d_im, const int x_offset, const int y_offset, const int face_width, const int face_height, cv::Mat &dstMat)
{
    cv::Mat rectMat(DIMX, DIMY, CV_32F);

    //    cv::Mat srcMat(height, width, CV_32F, cv::Scalar::all(0));

    dstMat.setTo(cv::Scalar::all(0));

    cudaMemcpy(rectMat.data, d_im, sizeof(float)*NTOTAL_PIXELS, cudaMemcpyDeviceToHost);
    cv::transpose(rectMat, rectMat);

    int rect_width = DIMX;
    int rect_height = DIMY;
    if (dstMat.rows - y_offset < DIMY) {
        rect_height = dstMat.rows - y_offset-1;
    }

    if (dstMat.cols - x_offset < DIMX) {
        rect_width = dstMat.cols - x_offset-1;
    }
    /*
*/
    rectMat = rectMat(cv::Rect(0,0,rect_width, rect_height));

    rectMat.copyTo(dstMat(cv::Rect(x_offset, y_offset, rect_width, rect_height)));

    double minSrc, maxSrc;
    cv::minMaxLoc(dstMat, &minSrc, &maxSrc);
    dstMat = (dstMat-minSrc)/(maxSrc-minSrc);
}









void imshow_opencv_cpu(cv::Mat &srcMat, const std::string win_name)
{
    double minSrc, maxSrc;
    cv::minMaxLoc(srcMat, &minSrc, &maxSrc);
    srcMat = (srcMat-minSrc)/(maxSrc-minSrc);

    cv::imshow(win_name,    srcMat.t());

}




void imwrite_opencv(const float *d_im, const std::string filepath)
{
    cv::Mat srcMat(DIMX, DIMY, CV_32F);

    cudaMemcpy(srcMat.data, d_im, sizeof(float)*NTOTAL_PIXELS, cudaMemcpyDeviceToHost);

    double minSrc, maxSrc;
    cv::minMaxLoc(srcMat, &minSrc, &maxSrc);
    srcMat = 255.0f*(srcMat-minSrc)/(maxSrc-minSrc);

    cv::imwrite(filepath,  srcMat.t());
}




/*
__global__ void render_expression_basis_texture(const float *EX, const float *EY, const float *EZ,
        const float *alphas, const float *betas, const float *gammas, const uint *indices, const int N1,
        ushort *tl, float *REX, float *REY, float *REZ, const ushort* triangle_idx)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    const int rel_index = indices[rowix];

    //! Important! We fill REX, ... in a ROW-MAJOR order. This way it will be easier to extract a submatrix of REX that ignores the bottom of REX
    const int idx = threadIdx.x + NEXP_COEFS*blockIdx.x;

    const int tl_i1 = triangle_idx[rel_index];
    const int tl_i2 = tl_i1 + N_TRIANGLES;
    const int tl_i3 = tl_i2 + N_TRIANGLES;


    REX[idx] = tex2D(EX_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EX_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EX_texture,colix,tl[tl_i3])*gammas[rel_index];
    REY[idx] = tex2D(EY_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EY_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EY_texture,colix,tl[tl_i3])*gammas[rel_index];
    REZ[idx] = tex2D(EZ_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EZ_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EZ_texture,colix,tl[tl_i3])*gammas[rel_index];
}
*/

