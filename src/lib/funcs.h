/*
 * funcs.h
 *
 *  Created on: Aug 1, 2020
 *      Author: sariyanide
 */

#ifndef FUNCS_H_
#define FUNCS_H_

#include "constants.h"


#include <vector>
#include <deque>
#include <iomanip>
#include <map>
#include <string>
#include <complex>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <regex>

#include <stdio.h>
#include <numeric>

#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// This will be defined in main.cu because it uses texture memory -- and I'm too lazy right now to make texture
// memory accessible by files other than main.cu and measure if we lose efficiency while doing this



template <typename T> T compute_std(std::vector<T> &v)
{
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    return std::sqrt(sq_sum / v.size() - mean * mean);
}

std::string base_name(const std::string& path, const std::string& delims = "/\\");

std::string remove_extension(const std::string& filename);


bool is_float(const std::string& myString );


__global__ void fill_tex_im(
        const ushort *xrend,
        const ushort *yrend,
        const uint* redundant_idx,
        const float *tex,
        float *tex_im);


__global__ void fill_ks_and_M0(
        const ushort *pixel_idx,
        const uint* redundant_idx,
        ushort *ks,
        ushort *ks_unsorted,
        ushort *M0,
        const uint N_unique_pixels);


__global__ void fill_ksides(
        const ushort *ks,
        ushort* ks_left, ushort* ks_right, ushort* ks_above, ushort* ks_below);

__global__ void reset_bool_flag(bool *flag, int N);

__global__ void reset_ushort_array(ushort *flag, int N);

__global__ void fill_krels(
        const uint N_pixels,
        const ushort *ks,
        const ushort *ks_sortidx,
        const ushort *ks_sortidx_sortidx,
        const ushort* ks_left, 	const 	ushort* ks_right, const 	ushort* ks_above, const ushort* ks_below,
        ushort* kl_rel, 				ushort* kr_rel, 			ushort* ka_rel, 		ushort* kb_rel,
        ushort* kl_rel_sorted, 			ushort* kr_rel_sorted, 		ushort* ka_rel_sorted, 	ushort* kb_rel_sorted,
        const ushort *cumM0);


__global__ void fill_krels2(
        const ushort *ks,
        const ushort *ks_sortidx,
        const ushort *ks_sortidx_sortidx,
        ushort* kl_rel, 	ushort* kr_rel, 	ushort* ka_rel, 	ushort* kb_rel,
        const ushort* kl_rel_sorted, 	const ushort* kr_rel_sorted, 	const ushort* ka_rel_sorted, 	const ushort* kb_rel_sorted);


void print_matrix( void *d_A, int Nrows, int Ncols);
void print_matrix_double( void *d_A, int Nrows, int Ncols);

void write_matrix_to_file( const void *d_A, int Nrows, int Ncols, const std::string &filepath);
void write_matrix_to_file_ushort( void *d_A, int Nrows, int Ncols, const std::string &filepath);
void write_matrix_to_file_bool( const void *d_A, int Nrows, int Ncols, const std::string &filepath);
void write_matrix_to_file_uint( const void *d_A, int Nrows, int Ncols, const std::string &filepath);

template <typename T> void write_2d_vector(const std::string& filepath, const std::vector<std::vector<T> >& vec) {
    std::ofstream DataFile;
    DataFile.open(filepath.c_str());
    for (int i = 0; i < vec.size(); i++)
    {
        for (int j = 0; j < vec[0].size(); j++)
        {
            DataFile << std::setprecision(std::numeric_limits<float>::digits10 + 1)  << (T) vec[i][j] << " ";
        }
        DataFile << std::endl;
    }
    DataFile.close();
}


template <typename T> void write_3d_reconstruction(const std::string& filepath, const std::vector<T>& X, const std::vector<T>& Y, const std::vector<T>& Z) {
    std::ofstream DataFile;
    DataFile.open(filepath.c_str());
    for (int i = 0; i < X.size(); i++)
    {
        DataFile << std::setprecision(std::numeric_limits<float>::digits10 + 1)  << (T) X[i] << " " << (T) Y[i] << " " << (T) Z[i];
        DataFile << std::endl;
    }
    DataFile.close();
}

std::vector<std::string> str_split (const std::string &s, char delim);


bool check_if_bb_OK(float *xp, float *yp);


__global__ void fill_tex_im1(
        const ushort *ks,
        const float *tex,
        float *tex_im,
        const uint N_unique_pixels);


__global__ void add_to_scalar(float *a, const float *b);

__global__ void multiply_and_add_to_scalar(float *a, const float* alpha, const float *b);

__global__ void add_to_scalar_negated(float *a, const float *b);

void imshow_opencv_cpu(cv::Mat &srcMat, const std::string win_name="CPU_image");

void imwrite_opencv(const float *d_im, const std::string filepath);

void write_identity(const std::string& path, float* x, float *y, float *z);
void write_texture(const std::string& path, float* x);

void writeLandmarks(const std::string& path, std::vector<float>& x, std::vector<float>& y);


__global__ void fill_grads(
        const ushort *ks,
        const float *tex_im,
        const float *source_im,
        float *gx, float *gy, float *gx_norm, float *gy_norm, float *h,
        float *gxs, float *gys, float *gxs_norm, float *gys_norm, float *hs,
        const uint N_unique_pixels,
        const uint N_render_pixels);



__global__ void view_transform_3d_pts_and_render_2d(const float* X0, const float* Y0, const float* Z0, const float* R__,
                                                    const float *taux__, const float *tauy__, const float *tauz__,
                                                    const float *phix, const float *phiy, const float *cx, const float *cy,
                                                    float *X, float *Y, float *Z, float *xp, float *yp);

__global__ void rotate_3d_pts(float* X, float* Y, float* Z, const float* R__);


__global__ void set_xtmp(const float* search_dir, const float *x, float t_coef__, float *xtmp, const uint Ktotal);


void print_vector( void *d_vec, int Nsize, const std::string& title = "THE TITLE IS MISSING");
void print_vector_double( void *d_vec, int Nsize, const std::string& title = "TITLE MISSING");
void print_vector_bool( void *d_vec, int Nsize, const std::string& title= "THE TITLE IS MISSING");
void print_vector_uint( void *d_vec, int Nsize, const std::string& title="TITLE MISSING");

template <class T>
T GetMax (T a, T b) {
    return (a>b?a:b);
}

template <class T>
void writeArrFile(const T* data, std::string FileName, int nrows, int ncols, bool is_row_major = false)

{
    std::ofstream DataFile;
    DataFile.open(FileName.c_str());
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++)
        {
            int idx;
            if (is_row_major)
                idx = j+i*ncols;
            else
                idx = i+j*nrows;


            DataFile <<   std::setprecision(std::numeric_limits<float>::digits10 + 1)  << (T) data[idx] << " ";
        }
        DataFile << std::endl;
    }
    DataFile.close();
}




template <class T>
void save_2d_cuda_array(void* d_arr, int Nrows, int Ncols, const std::string& fname)
{
    T *h_arr;
    h_arr = (T*)malloc( Nrows*Ncols*sizeof(T) );


    cudaMemcpy(h_arr, d_arr, sizeof(T)*Nrows*Ncols, cudaMemcpyDeviceToHost);
    writeArrFile<T>(h_arr, fname, Nrows, Ncols, false);


    free(h_arr);

}



using std::vector;

/*
template <class T>
T* vec2arr(vector<vector<T> > &vals, int nrows, int ncols, bool is_row_major = false);

template <class T>
vector< vector<T> > read2DVectorFromFile(const std::string& FileName,  int rows, int cols);
*/

__device__ __forceinline__ float atomicMinFloat (float * addr, float value);

/**
 * pixel_idx: is the 1d index of the pixel on the image. this is necessary for z-buffering
 */
__global__ void get_pixels_to_render(const ushort *tl, const float *xp, const float *yp,
                                     bool *rend_flag, ushort *pixel_idx,
                                     float *alphas_redundant, float *betas_redundant, float *gammas_redundant,
                                     ushort* triangle_idx,
                                     const float *Z, float *Ztmp,
                                     const ushort x0, const ushort y0, float *Zmins, uint *redundant_idx);



__global__ void populate_pixel_idx2(const ushort *pixel_idx, const uint* indices, ushort *pixel_idx2, const uint N1);

__global__ void zbuffer_update(const ushort* sorted_pixel_idx, const ushort* unique_pixel_locations, const float *Ztmp,
                               const uint *indices, const ushort *inner_indices, bool* rend_flag_tmp, const uint Nredundant,  const ushort N_unique_pixels , const uint N1);

__global__ void fill_float_arr(float *arr, float val, int N);
__global__ void keep_only_minZ(const float *Zmins, const float* Zs, const ushort* pixel_idx, uint* redundant_idx, int N);

__global__ void render_expression_basis_texture(const float *EX, const float *EY, const float *EZ,
                                                const float *alphas, const float *betas, const float *gammas, const uint *indices, const int N1,
                                                ushort *tl, float *REX, float *REY, float *REZ, const ushort* triangle_idx);

void imshow_opencv(const float *d_im, const std::string win_name = "im");
void imshow_opencv_buffered(const float *d_im, const int x_offset, const int y_offset, const int width, const int height, const std::string win_name = "im");

void create_cvmat_buffered(const float *d_im, const int x_offset, const int y_offset, const int width, const int height, cv::Mat &dstMat);


template <class T>
T* vec2arr(vector<vector<T> > &vals, int nrows, int ncols, bool is_row_major = false, float alpha = 1.0f)
{
    T* temp;
    temp = (T*)malloc( nrows*ncols*sizeof(T) );
    for(uint i=0; i < nrows; i++)
        for(uint j=0; j < ncols; j++) {
            uint idx;

            if (is_row_major)
                idx = j+ncols*i;
            else
                idx = i+j*nrows;

            temp[idx] = alpha*vals[i][j];
        }

    return temp;
}



template <class T>
int ReadNumbers( const std::string & s, std::vector<T> & v ) {
    std::string scp(s);

    scp = std::regex_replace(scp, std::regex("nan"), "0"); // replace 'def' -> 'klm'
    std::istringstream is( scp );
    T n;




    while( is >> n ) {
        v.push_back( n );
    }
    return v.size();
}

template <class T>
void import_matrix_from_txt_file(const std::string& filepath, vector <T>& v, int& rows, int& cols) {
    std::ifstream file_X;
    std::string line;

    file_X.open(filepath);
    if (file_X.is_open())
    {
        int i=0;
        getline(file_X, line);

        cols =ReadNumbers<float>( line, v );
        for ( i=1;i<32767;i++){
            if ( !getline(file_X, line) ) break;
            ReadNumbers<float>( line, v );
        }

        rows=i;

        if(rows >32766)
            std::cout<< "N must be smaller than MAX_INT";

        file_X.close();
    }
    else{
        std::cout << "file open failed";
    }
}





template <class T>
std::vector< std::vector<T> > read2DVectorFromFile_unknown_size(const std::string& FileName) {

    int rows, cols;
    std::vector<T> tmp;
    import_matrix_from_txt_file<T>(FileName, tmp, rows, cols);

    using std::vector;
    vector<vector<T>> tl;

    std::ifstream DataFile(FileName.c_str());
    if (DataFile.is_open()) {
        for (uint i = 0; i < rows; i++) {
            tl.push_back(vector<T>());
            for (uint j = 0; j < cols; j++) {
                float tmp;
                if (!(DataFile >> std::setprecision(16) >> tmp)) {
                    tl[i].push_back(NAN);
                    DataFile.clear();
                    DataFile.ignore(3);
                    continue;
                }

                tl[i].push_back((T)tmp);
            }
        }
        DataFile.close();
    }
    return tl;
}



template <class T>
std::vector< std::vector<T> > read2DVectorFromFile(const std::string& FileName,  int rows, int cols) {

    using std::vector;
    vector<vector<T>> tl;

    std::ifstream DataFile(FileName.c_str());
    if (DataFile.is_open()) {
        for (uint i = 0; i < rows; i++) {
            tl.push_back(vector<T>());
            for (uint j = 0; j < cols; j++) {
                float tmp;
                DataFile >> std::setprecision(16) >> tmp;
                tl[i].push_back((T)tmp);
            }
        }
        DataFile.close();
    }

    return tl;
}


float compute_face_size(const float *xp, const float *yp);


__global__ void render_expression_basis_texture(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int N1, const ushort* __restrict__ tl,
        float* __restrict__ REX, float* __restrict__ REY, float* __restrict__ REZ,
        const ushort* __restrict__ triangle_idx);




__global__ void render_expression_basis_texture_colmajor_rotated2(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int Nunique_pixels, const ushort* __restrict__ tl,
        float*  __restrict__ REX, float*  __restrict__ REY, float*  __restrict__ REZ,
        const ushort* __restrict__ triangle_idx, const float* R__);



__global__ void render_expression_basis_texture_colmajor_rotated(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int Nunique_pixels, const ushort* __restrict__ tl,
        float*  __restrict__ REX, float*  __restrict__ REY, float*  __restrict__ REZ,
        const ushort* __restrict__ triangle_idx, const float* R__);





__global__ void render_expression_basis_texture_colmajor(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int N1, const ushort* __restrict__ tl,
        float* __restrict__ REX, float* __restrict__ REY, float* __restrict__ REZ,
        const ushort* __restrict__ triangle_idx);



__global__ void render_expression_basis_texture_colmajor2(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int Nunique_pixels, const ushort* __restrict__ tl,
        float* __restrict__ REX, float* __restrict__ REY, float* __restrict__ REZ,
        const ushort* __restrict__ triangle_idx);





__global__ void render_identity_basis_texture(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int N1, const ushort* __restrict__ tl,
        float* __restrict__ RIX, float* __restrict__ RIY, float* __restrict__ RIZ,
        const ushort* __restrict__ triangle_idx, const ushort Kalpha);





__global__ void render_identity_basis_texture_colmajor_rotated(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int Nunique_pixels, const ushort* __restrict__ tl,
        float* __restrict__ RIX, float* __restrict__ RIY, float* __restrict__ RIZ,
        const ushort* __restrict__ triangle_idx, const float* R__);







__global__ void render_identity_basis_texture_colmajor_rotated2(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int Nunique_pixels, const ushort* __restrict__ tl,
        float* __restrict__ RIX, float* __restrict__ RIY, float* __restrict__ RIZ,
        const ushort* __restrict__ triangle_idx, const float* R__);







__global__ void elementwise_vector_multiplication(float *vec_out, const float *vec_in, const int vec_len );












__global__ void render_texture_basis_texture(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int N1, const ushort* __restrict__ tl,
        float* __restrict__ RTEX, const ushort* __restrict__ triangle_idx, const ushort Kbeta);



__global__ void render_identity_basis(
        const float *alphas, const float *betas, const float *gammas,
        const uint *indices, const int N1, ushort *tl,
        float *RIX, float *RIY, float *RIZ,
        const ushort* triangle_idx);















#endif /* FUNCS_H_ */
