#include "cuda.h"
#include <experimental/filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "config.h"

#include <string>

#include <vector>
#include <stdio.h>
#include <numeric>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "constants.h"
#include "renderer.h"

#include "derivative_computer.h"
#include "logbarrier_initializer.h"

#include "model_fitter.h"
#include "funcs.h"
#include "newfuncs.h"
#include "rotation_header.h"
#include "Optimizer.h"
#include "camera.h"
#include "solver.h"
#include "preprocessing.h"



#include <glob.h> // glob(), globfree()
#include <string.h> // memset()
#include <stdexcept>
#include <sstream>

#include <opencv2/dnn.hpp>

#ifdef VISUALIZE_3D
#include "GLfuncs.h"

#endif


//using namespace cv;
//using namespace cv::dnn;

using std::string;
using std::vector;

// these exist on the GPU side
texture<float,2> EX_texture;
texture<float,2> EY_texture;
texture<float,2> EZ_texture;

texture<float,2> IX_texture;
texture<float,2> IY_texture;
texture<float,2> IZ_texture;

texture<float,2> TEX_texture;

bool fit_multiframe(Camera &cam, Renderer &r, const std::vector<std::vector<float> >& selected_frame_xps, const  std::vector<std::vector<float> >& selected_frame_yps,
                   const std::vector<std::vector<float> >& selected_frame_xranges, const  std::vector<std::vector<float> >& selected_frame_yranges,
                   const std::vector< cv::Mat >& selected_frames,
                   float *h_X0, float *h_Y0, float *h_Z0, float *h_tex_mu, std::vector<std::string>* result_basepaths=NULL);

using std::vector;

int create_data_for_multiframe(Renderer &r, const std::string& outdir, Camera &cam0,
                               vector<vector<float> >& xps, vector<vector<float> >& yps,
                               vector<vector<float> >& xranges, vector<vector<float> >& yranges,
                               vector<cv::Mat> &selected_frames, vector<std::string>& result_basepaths, const std::vector<string>& imfiles,
                               cv::dnn::Net &detection_net, cv::dnn::Net &landmark_net, cv::dnn::Net &leye_net, cv::dnn::Net &reye_net, cv::dnn::Net &mouth_net, cv::dnn::Net &correction_net,
                               float &mean_face_size,
                               bool set_RESIZE_COEF_via_median=true, int combination_id = -1);

//const int KERNEL_RADIUS=2;

// /media/v/SSD1TB/dataset/videos/treecam/ML/ML0001.mp4 /media/v/SSD1TB/dataset/videos/treecam/ML/ML0001.mp4.avi
// /media/v/SSD1TB/dataset/Florence/images/for_experiments/ /media/v/SSD1TB/dataset/Florence/results/3DIv2

// /media/v/SSD1TB/dataset/BU4DFE/images/ /media/v/SSD1TB/dataset/BU4DFE/results/3DIv2

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "not enough arguments! exiting." << std::endl;
    }

    Camera cam0;
    float field_of_view = 0.0f;
    float field_of_viewy = 0.0f;

    std::string calibration_path("");
    if (argc >= 4) {
        if (!is_float(argv[3]))
        {
            calibration_path = argv[3];
            cam0.init(calibration_path);
        } else {
            field_of_view = std::stof(argv[3]);
            // Camera will be initialized later -- if necessary
        }
    }

    if (argc >= 6) {
        if (!is_float(argv[5]))
        {
            calibration_path = argv[5];
            cam0.init(calibration_path);
        } else {
            field_of_viewy = std::stof(argv[5]);
            // Camera will be initialized later -- if necessary
        }
    }
    else
        field_of_viewy = field_of_view;

    std::ifstream inFile;
    inFile.open(argv[1]); //open the input file

    std::string config_filepath(argv[2]);
    config::set_params_from_YAML_file(config_filepath);

    std::stringstream strStream;
    strStream << inFile.rdbuf(); //read the file
    std::string str = strStream.str(); //str holds the content of the file
    vector<string> imfile_combinations = str_split(str, ';');

    std::string outdir_root = std::string(argv[4]);
    std::string outdir_mid =  std::string(argv[4]) + "/" + config::get_key();
    std::string outdir;

    if (field_of_view != 1.0f)
        outdir = outdir_mid + "/" + std::to_string((int)field_of_view);
    else
        outdir = outdir_mid;

    if (!config::OUTDIR_WITH_PARAMS)
        outdir = outdir_root;

    if (!std::experimental::filesystem::exists(outdir_root))
        std::experimental::filesystem::create_directory(outdir_root);

    if (config::OUTDIR_WITH_PARAMS)
    {
        if (!std::experimental::filesystem::exists(outdir_mid))
            std::experimental::filesystem::create_directory(outdir_mid);

        if (!std::experimental::filesystem::exists(outdir))
            std::experimental::filesystem::create_directory(outdir);
    }


    Renderer r(config::NFRAMES, 199, 199, config::K_EPSILON, true, true, true);
    // Bind texture memories
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc desc3 = cudaCreateChannelDesc<float>();

    // Start with expression bases
    cudaBindTexture2D(0, EX_texture, r.d_EX_row_major, desc, config::K_EPSILON, config::NPTS, r.pitch);
    cudaBindTexture2D(0, EY_texture, r.d_EY_row_major, desc, config::K_EPSILON, config::NPTS, r.pitch);
    cudaBindTexture2D(0, EZ_texture, r.d_EZ_row_major, desc, config::K_EPSILON, config::NPTS, r.pitch);

    // Now identity bases
    if (r.use_identity)
    {
        cudaBindTexture2D(0, IX_texture, r.d_IX_row_major, desc2, config::NID_COEFS, config::NPTS, r.pitch2);
        cudaBindTexture2D(0, IY_texture, r.d_IY_row_major, desc2, config::NID_COEFS, config::NPTS, r.pitch2);
        cudaBindTexture2D(0, IZ_texture, r.d_IZ_row_major, desc2, config::NID_COEFS, config::NPTS, r.pitch2);
    }

    // Finally the texture bases
    if (r.use_texture)
    {
        cudaBindTexture2D(0, TEX_texture, r.d_TEX_row_major, desc3, config::NTEX_COEFS, config::NPTS, r.pitch3);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    cv::dnn::Net detection_net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    detection_net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

    std::string tfLandmarkNet("models/landmark_models/model_FAN_frozen.pb");
    cv::dnn::Net landmark_net = cv::dnn::readNetFromTensorflow(tfLandmarkNet);
    landmark_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    landmark_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::dnn::Net leye_net = cv::dnn::readNetFromTensorflow("models/landmark_models/m-64l64g0-64-128-5121968464leye.pb");
    leye_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    leye_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::dnn::Net reye_net = cv::dnn::readNetFromTensorflow("models/landmark_models/m-64l64g0-64-128-5121968464reye.pb");
    reye_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    reye_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::dnn::Net mouth_net = cv::dnn::readNetFromTensorflow("models/landmark_models/m-64l64g0-64-128-5121968464mouth.pb");
    mouth_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    mouth_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::dnn::Net correction_net = cv::dnn::readNetFromTensorflow("models/landmark_models/model_correction.pb");

    int combination_id = 0;

    float mean_face_size = -1.0f;
    for (auto cur_combination : imfile_combinations)
    {
        std::vector<std::string> imfiles = str_split(cur_combination, ',');

        if (imfiles.size() < config::NFRAMES) {
            std::cout << "Not enough files in this combination; skipping comb" << std::endl;
            continue;
        }


        std::cout << imfiles[0] << std::endl;

        if (!cam0.initialized)
        {
            cv::Mat im = cv::imread(imfiles[0]);

            float cam_cx = im.cols/2.0;
            float cam_cy = im.rows/2.0;
            //cam_cx = 997.2f;
            //cam_cy = 706.5f;

            double angle_x = field_of_view*M_PI/180.0; // angle in radians
            //double angle_y = angle_x; //60.0f*M_PI/180.0; //(cam_cy/cam_cx)*angle_x;
            double angle_y = field_of_viewy*M_PI/180.0; //60.0f*M_PI/180.0; //(cam_cy/cam_cx)*angle_x;

            float cam_alphax = cam_cx/(tan(angle_x/2.0));
            float cam_alphay = cam_cy/(tan(angle_y/2.0));
            if (field_of_view == field_of_viewy) {
                cam_alphay = cam_alphax;
            }
            std::cout << cam_alphax << std::endl;
            std::cout << cam_alphay << std::endl;
            //float cam_alphay = cam_alphax; //cam_cy/(tan(angle_y/2.0));

            cam0.init(cam_alphax, cam_alphay, cam_cx, cam_cy, false);
        }

        combination_id++;

        std::cout << "CUR_COMBINATION:  " << cur_combination << std::endl;
        try {
            vector<vector<float> > xps, yps, xranges, yranges;
            vector<cv::Mat> selected_frames;
            vector<std::string> basepaths;

            cam0.update_camera(1.0f);

            int res_ = create_data_for_multiframe(r, outdir, cam0, xps, yps, xranges, yranges, selected_frames,
                                                  basepaths, imfiles, detection_net, landmark_net, leye_net, reye_net, mouth_net, correction_net,
                                                  mean_face_size, false, combination_id);
            if (res_ == -1) {
	    	std::cout << "failed_to_fit" << std::endl;
                continue;
	    }

            //!std::cout << "N_BASEPATHS IS: " << basepaths.size() << std::endl;

            if (basepaths.size() != config::NFRAMES) {
	    	std::cout << "not_enough_files" << std::endl;
                continue;
	    }

//            cam0.update_camera(config::REF_FACE_SIZE/mean_face_size);
//            cam0.update_camera(0.5f);
            cam0.update_camera(1.0f);

            float *h_X0, *h_Y0, *h_Z0, *h_tex_mu;

            h_X0 = (float*)malloc( config::NPTS*sizeof(float) );
            h_Y0 = (float*)malloc( config::NPTS*sizeof(float) );
            h_Z0 = (float*)malloc( config::NPTS*sizeof(float) );
            h_tex_mu = (float*)malloc( config::NPTS*sizeof(float) );


            std::string identity_path(basepaths[0]+".id.txt");
            std::string texture_path(basepaths[0]+".tex.txt");

            bool success = fit_multiframe(cam0, r, xps, yps, xranges, yranges, selected_frames, h_X0, h_Y0, h_Z0, h_tex_mu, &basepaths);
            if (success)
            {
                write_identity(identity_path, h_X0, h_Y0, h_Z0);
                write_texture(texture_path, h_tex_mu);
            }

            free(h_X0);
            free(h_Y0);
            free(h_Z0);
            free(h_tex_mu);

        }  catch (std::exception&e) {
            std::cout << "Encountered error, continuing ... :";
            std::cout << e.what() << std::endl;
        }
    }

    if (r.use_expression)
    {
        cudaUnbindTexture(EX_texture);
        cudaUnbindTexture(EY_texture);
        cudaUnbindTexture(EZ_texture);
    }

    if (r.use_identity)
    {
        cudaUnbindTexture(IX_texture);
        cudaUnbindTexture(IY_texture);
        cudaUnbindTexture(IZ_texture);
    }

    if (r.use_texture)
    {
        cudaUnbindTexture(TEX_texture);
    }
}









__global__ void render_expression_basis_texture_colmajor_rotated2(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int Nunique_pixels, const ushort* __restrict__ tl,
        float*  __restrict__ REX, float*  __restrict__ REY, float*  __restrict__ REZ,
        const ushort* __restrict__ triangle_idx, const float* R__, int N_TRIANGLES, uint Nredundant)
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

    const float tmpx = tex2D(EX_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EX_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EX_texture,colix,tl[tl_i3])*gammas[rel_index];
    const float tmpy = tex2D(EY_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EY_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EY_texture,colix,tl[tl_i3])*gammas[rel_index];
    const float tmpz = tex2D(EZ_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(EZ_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(EZ_texture,colix,tl[tl_i3])*gammas[rel_index];

    REX[idx] = tmpx*R00 + tmpy*R01 + tmpz*R02;
    REY[idx] = tmpx*R10 + tmpy*R11 + tmpz*R12;
    REZ[idx] = tmpx*R20 + tmpy*R21 + tmpz*R22;
}



__global__ void render_identity_basis_texture(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int N1, const ushort* __restrict__ tl,
        float* __restrict__ RIX, float* __restrict__ RIY, float* __restrict__ RIZ,
        const ushort* __restrict__ triangle_idx, const ushort Kalpha, const int N_TRIANGLES)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    const int rel_index = indices[rowix];

    //! Important! We fill REX, ... in a ROW-MAJOR order. This way it will be easier to extract a submatrix of REX that ignores the bottom of REX
    const int idx = threadIdx.x + Kalpha*blockIdx.x;

    const int tl_i1 = triangle_idx[rel_index];
    const int tl_i2 = tl_i1 + N_TRIANGLES;
    const int tl_i3 = tl_i2 + N_TRIANGLES;

    RIX[idx] = tex2D(IX_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(IX_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(IX_texture,colix,tl[tl_i3])*gammas[rel_index];
    RIY[idx] = tex2D(IY_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(IY_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(IY_texture,colix,tl[tl_i3])*gammas[rel_index];
    RIZ[idx] = tex2D(IZ_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(IZ_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(IZ_texture,colix,tl[tl_i3])*gammas[rel_index];
}



__global__ void render_texture_basis_texture(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int N1, const ushort* __restrict__ tl,
        float* __restrict__ RTEX, const ushort* __restrict__ triangle_idx,
        const ushort Kbeta, const int N_TRIANGLES)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    const int rel_index = indices[rowix];

    //! Important! We fill REX, ... in a ROW-MAJOR order. This way it will be easier to extract a submatrix of REX that ignores the bottom of REX
    const int idx = threadIdx.x + Kbeta*blockIdx.x;

    const int tl_i1 = triangle_idx[rel_index];
    const int tl_i2 = tl_i1 + N_TRIANGLES;
    const int tl_i3 = tl_i2 + N_TRIANGLES;

    RTEX[idx] = tex2D(TEX_texture,colix,tl[tl_i1])*alphas[rel_index] + tex2D(TEX_texture,colix,tl[tl_i2])*betas[rel_index] + tex2D(TEX_texture,colix,tl[tl_i3])*gammas[rel_index];
}








bool fit_multiframe(Camera &cam0, Renderer& r, const std::vector<std::vector<float> >& selected_frame_xps, const  std::vector<std::vector<float> >& selected_frame_yps,
                   const std::vector<std::vector<float> >& selected_frame_xranges, const  std::vector<std::vector<float> >& selected_frame_yranges,
                   const std::vector< cv::Mat >& selected_frames,
                   float *h_X0, float *h_Y0, float *h_Z0, float *h_tex_mu, std::vector<std::string>* result_basepaths)
{
    // Check python code in the end of this file to see how this kernel is generated

    float *h_Kernel; // [2*KERNEL_RADIUS+1];

    float h_Kernel_200sgm[2*KERNEL_RADIUS+1] = {4.9640312e-06, 8.9220186e-05, 1.0281866e-03, 7.5973268e-03, 3.5993993e-02, 1.0934010e-01, 2.1296541e-01, 2.6596162e-01, 2.1296541e-01, 1.0934010e-01, 3.5993993e-02, 7.5973268e-03, 1.0281866e-03, 8.9220186e-05, 4.9640312e-06};
    float h_Kernel_100sgm[2*KERNEL_RADIUS+1] = {9.1347208e-12, 6.0758834e-09, 1.4867194e-06, 1.3383022e-04, 4.4318484e-03, 5.3990968e-02, 2.4197073e-01, 3.9894229e-01, 2.4197073e-01, 5.3990968e-02, 4.4318484e-03, 1.3383022e-04, 1.4867194e-06, 6.0758834e-09, 9.1347208e-12};
    float h_Kernel_075sgm[2*KERNEL_RADIUS+1] = {6.4550189e-20, 6.7361578e-15, 1.1880850e-10, 3.5416286e-07, 1.7843490e-04, 1.5194189e-02, 2.1867350e-01, 5.3190696e-01, 2.1867350e-01, 1.5194189e-02, 1.7843490e-04, 3.5416286e-07, 1.1880850e-10, 6.7361578e-15, 6.4550189e-20};
    float h_Kernel_050sgm[2*KERNEL_RADIUS+1] = {2.15799964e-43, 4.23189662e-32, 1.51709817e-22, 9.96126162e-15, 1.19794565e-08, 2.63865077e-04, 1.06450766e-01, 7.86570668e-01, 1.06450766e-01, 2.63865077e-04, 1.19794565e-08, 9.96126162e-15, 1.51709817e-22, 4.23189662e-32, 2.15799964e-43};

    if (config::KERNEL_SIGMA == 2.0f) h_Kernel = h_Kernel_200sgm;
    if (config::KERNEL_SIGMA == 1.0f) h_Kernel = h_Kernel_100sgm;
    if (config::KERNEL_SIGMA == 0.75f) h_Kernel = h_Kernel_075sgm;
    if (config::KERNEL_SIGMA == 0.50f) h_Kernel = h_Kernel_050sgm;

    setConvolutionKernel(h_Kernel);



    cusolverDnHandle_t handleDn;
    cusolverDnCreate(&handleDn);

    cublasHandle_t handle;
    cublasCreate(&handle);

    uint T = r.T;

    OptimizationVariables ov(T, r.Kalpha, r.Kbeta, r.Kepsilon, r.use_identity, r.use_texture, r.use_expression);
    OptimizationVariables ov_linesearch(r.T, r.Kalpha, r.Kbeta, r.Kepsilon, r.use_identity, r.use_texture, r.use_expression);
    OptimizationVariables ov_lb(T, config::K_ALPHA_L, config::K_BETA_L, config::K_EPSILON_L, r.use_identity, r.use_texture, r.use_expression, true);
    OptimizationVariables ov_lb_linesearch(T, config::K_ALPHA_L, config::K_BETA_L, config::K_EPSILON_L, r.use_identity, r.use_texture, r.use_expression, true);

    Optimizer o(&ov, r.Kalpha, r.Kbeta, r.Kepsilon, r.use_identity, r.use_texture, r.use_expression);

    DerivativeComputer dc(true, r.use_identity, r.use_texture, r.use_expression);

    o.ov_ptr = &ov;

    ov.set_frame(0);
    ov_linesearch.set_frame(0);


    RotationComputer rc(ov.u);
    RotationComputer rc_linesearch(ov_linesearch.u);




    Solver s(handleDn, ov.Ktotal);
    Solver s_lambda(handleDn, 3);





    cudaEvent_t     start, stop;
    // rendered expression basis


    float *d_tmp;
    float *search_dir_Lintensity, *dg_Lintensity;

    cudaMalloc((void**)&search_dir_Lintensity, sizeof(float)*1);
    cudaMalloc((void**)&dg_Lintensity, sizeof(float)*1);

    cudaMalloc( (void**)&d_tmp, sizeof(float));

    float* d_cropped_face, *d_buffer_face;
    cudaMalloc((void**)&d_cropped_face,  sizeof(float)*T*DIMX*DIMY);
    cudaMalloc((void**)&d_buffer_face,  sizeof(float)*DIMX*DIMY);

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    std::vector<Camera> cams;

    for (size_t i=0; i<r.T; ++i) {
        cams.push_back(Camera(cam0));
        const std::vector<float> xtmp = selected_frame_xps[i];
        const std::vector<float> ytmp = selected_frame_yps[i];

        cams.back().update_camera(config::REF_FACE_SIZE/compute_face_size(&xtmp[0], &ytmp[0]));
    }

    std::cout << "BEFORE MULTI" << std::endl;
    bool success = fit_to_multi_images(cams, selected_frame_xps, selected_frame_yps,
                                selected_frame_xranges, selected_frame_yranges, selected_frames,
                                result_basepaths, r,  o, handleDn,
                                handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, ov_lb, ov_lb_linesearch, rc,
                                rc_linesearch,  dc, s, s_lambda, d_tmp, search_dir_Lintensity, dg_Lintensity,
                            h_X0, h_Y0, h_Z0, h_tex_mu);
    std::cout << "AFTER MULTI -- success: "<< success << std::endl;


    // get stop time, and display the timing results
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cusolverDnDestroy(handleDn);
    cudaFree( d_tmp );

    cudaFree( search_dir_Lintensity );
    cudaFree( dg_Lintensity );


    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    //!printf( "Time to generate:  %3.1f ms\n", elapsedTime );



    cudaFree( d_cropped_face );
    cudaFree( d_buffer_face );


    cublasDestroy(handle);

    //!std::cout << "Cleaned everything " << std::endl;

    return success;

}
















std::vector<std::string> glob(const std::string& pattern) {
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}




int create_data_for_multiframe(Renderer& r, const std::string& outdir, Camera& cam0,
                               vector<vector<float> >& xps, vector<vector<float> >& yps,
                               vector<vector<float> >& xranges, vector<vector<float> >& yranges,
                               vector<cv::Mat>& selected_frames,
                               vector<std::string>& basepaths,
                               const std::vector<string>& imfiles,
                               cv::dnn::Net &detection_net, cv::dnn::Net &landmark_net, cv::dnn::Net& leye_net, cv::dnn::Net& reye_net, cv::dnn::Net& mouth_net, cv::dnn::Net& correction_net,
                               float &mean_face_size,
                               bool set_RESIZE_COEF_via_median,
                               int combination_id)
{
//    std::cout << imfiles.size() << std::endl;

    if (imfiles.size() == 0)
        return 0;

    float *h_Kernel; // [2*KERNEL_RADIUS+1];

    float h_Kernel_200sgm[2*KERNEL_RADIUS+1] = {4.9640312e-06, 8.9220186e-05, 1.0281866e-03, 7.5973268e-03, 3.5993993e-02, 1.0934010e-01, 2.1296541e-01, 2.6596162e-01, 2.1296541e-01, 1.0934010e-01, 3.5993993e-02, 7.5973268e-03, 1.0281866e-03, 8.9220186e-05, 4.9640312e-06};
    float h_Kernel_100sgm[2*KERNEL_RADIUS+1] = {9.1347208e-12, 6.0758834e-09, 1.4867194e-06, 1.3383022e-04, 4.4318484e-03, 5.3990968e-02, 2.4197073e-01, 3.9894229e-01, 2.4197073e-01, 5.3990968e-02, 4.4318484e-03, 1.3383022e-04, 1.4867194e-06, 6.0758834e-09, 9.1347208e-12};
    float h_Kernel_075sgm[2*KERNEL_RADIUS+1] = {6.4550189e-20, 6.7361578e-15, 1.1880850e-10, 3.5416286e-07, 1.7843490e-04, 1.5194189e-02, 2.1867350e-01, 5.3190696e-01, 2.1867350e-01, 1.5194189e-02, 1.7843490e-04, 3.5416286e-07, 1.1880850e-10, 6.7361578e-15, 6.4550189e-20};
    float h_Kernel_050sgm[2*KERNEL_RADIUS+1] = {2.15799964e-43, 4.23189662e-32, 1.51709817e-22, 9.96126162e-15, 1.19794565e-08, 2.63865077e-04, 1.06450766e-01, 7.86570668e-01, 1.06450766e-01, 2.63865077e-04, 1.19794565e-08, 9.96126162e-15, 1.51709817e-22, 4.23189662e-32, 2.15799964e-43};

    if (config::KERNEL_SIGMA == 2.0f) h_Kernel = h_Kernel_200sgm;
    if (config::KERNEL_SIGMA == 1.0f) h_Kernel = h_Kernel_100sgm;
    if (config::KERNEL_SIGMA == 0.75f) h_Kernel = h_Kernel_075sgm;
    if (config::KERNEL_SIGMA == 0.50f) h_Kernel = h_Kernel_050sgm;

    setConvolutionKernel(h_Kernel);

    cusolverDnHandle_t handleDn;
    cusolverDnCreate(&handleDn);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const ushort Kalpha = 199; //199;
    const ushort Kbeta = 199; //199;
    const ushort Kepsilon = config::K_EPSILON;

    const bool use_identity = true;
    const bool use_texture = true;
    const bool use_expression = true;

    const std::string caffeConfigFile = "models/deploy.prototxt";
    const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    std::string device = "CPU";
    std::string framework = "caffe";

    OptimizationVariables ov(1, Kalpha, Kbeta, Kepsilon, use_identity, use_texture, use_expression);
    OptimizationVariables ov_lb(1, config::K_ALPHA_L, config::K_BETA_L, config::K_EPSILON_L, use_identity, use_texture, use_expression, true);
    OptimizationVariables ov_lb_linesearch(1, config::K_ALPHA_L, config::K_BETA_L, config::K_EPSILON_L, use_identity, use_texture, use_expression,  true);
    OptimizationVariables ov_linesearch(1, Kalpha, Kbeta, Kepsilon, use_identity, use_texture, use_expression);

    Optimizer o(&ov, Kalpha, Kbeta, Kepsilon, use_identity, use_texture, use_expression);

    DerivativeComputer dc(true, use_identity, use_texture, use_expression);

    o.ov_ptr = &ov;

    ov.set_frame(0);
    ov_linesearch.set_frame(0);

    RotationComputer rc(ov.u);
    RotationComputer rc_linesearch(ov_linesearch.u);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Solver s(handleDn, ov.Ktotal);
    Solver s_lambda(handleDn, 3);

    cudaEvent_t     start, stop;
    // rendered expression basis


    float *d_tmp;
    float *search_dir_Lintensity, *dg_Lintensity;

    cudaMalloc((void**)&search_dir_Lintensity, sizeof(float)*1);
    cudaMalloc((void**)&dg_Lintensity, sizeof(float)*1);

    cudaMalloc( (void**)&d_tmp, sizeof(float));

    float* d_cropped_face, *d_buffer_face;
    cudaMalloc((void**)&d_cropped_face,  sizeof(float)*DIMX*DIMY);
    cudaMalloc((void**)&d_buffer_face,  sizeof(float)*DIMX*DIMY);

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    vector<float> face_sizes;

    for (uint fi=0; fi<imfiles.size(); ++fi)
    {
        std::vector<float> xp_landmark, yp_landmark;

        std::string impath = imfiles[fi];
        cv::Mat im = cv::imread(impath);

//        Camera cam0(camera_calibration_path);
	cam0.update_camera(1.0f);
        if (cam0.cam_remap) {
            cam0.undistort(im, im);
            //cv::imshow("im", im);
            //cv::waitKey(0);
        }

        cv::copyMakeBorder(im, im, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, cv::BORDER_CONSTANT, 0);

        float face_size;

        if (im.channels() == 1) {
            cv::cvtColor(im, im, cv::COLOR_GRAY2RGB);
        }

        std::string result_basepath = outdir+std::string("/") + remove_extension(base_name(imfiles[fi]));

        if (combination_id != -1) {
            std::stringstream sstmp;
            sstmp << "_comb" << std::setw(2) << std::setfill('0') << combination_id;
            result_basepath += sstmp.str();
        }

        std::string obj_path(result_basepath+".obj");
        std::string vars_path(result_basepath+".vars");

        if (std::experimental::filesystem::exists(obj_path) || std::experimental::filesystem::exists(vars_path)) {
            std::cout << "This file exists, skipping ..." << std::endl;
            break;
            //continue;
        }
        /*
*/
        std::vector<float> xrange, yrange;
        try
        {
            cv::Rect ROI(-1, -1, -1, -1);
            double face_confidence;
            cv::Rect face_rect = detect_face_opencv(detection_net, framework, im, &ROI, &face_confidence, true);
            detect_landmarks_opencv(face_rect, face_confidence, landmark_net, leye_net, reye_net, mouth_net, correction_net, im,
                                     xp_landmark, yp_landmark, face_size,
                    xrange, yrange, config::USE_LOCAL_MODELS, false);

            cam0.update_camera(config::REF_FACE_SIZE/face_size);

            face_sizes.push_back(face_size);
        } catch (std::exception& e) {
            std::cout << "Problems at DETECTION phase -- skipping" << std::endl;
        }


        if (xp_landmark.size() == 0)
            continue;

        //!std::cout << '\t' << remove_extension(base_name(imfiles[fi])) << std::endl;

        //        try
        //        {

        cv::Mat inputImage = cv::imread(impath, cv::IMREAD_GRAYSCALE);
        cv::copyMakeBorder(inputImage, inputImage, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, config::PAD_SINGLE_IMAGE, cv::BORDER_CONSTANT, 0);
        cv::resize(inputImage, inputImage, cv::Size(), cam0.resize_coef, cam0.resize_coef);

        cv::copyMakeBorder(inputImage, inputImage, 0, DIMY, 0, DIMX, cv::BORDER_CONSTANT, 0);
        //!!!!cam0.update_camera(cam0.resize_coef);


        std::vector<Camera> cams;
        cams.push_back(Camera(cam0));

        bool fit_success = fit_to_single_image_autolandmarks(impath, xp_landmark, yp_landmark, xrange, yrange, result_basepath,
                                                             r, o, handleDn, handle, d_cropped_face, d_buffer_face, ov, ov_linesearch,
                                                             ov_lb, ov_lb_linesearch, rc, rc_linesearch, dc, s, s_lambda, d_tmp,
                                                             search_dir_Lintensity, dg_Lintensity, cams, true);
        if (fit_success) {
            xps.push_back(xp_landmark);
            yps.push_back(yp_landmark);
            xranges.push_back(xrange);
            yranges.push_back(yrange);

            selected_frames.push_back(im);
            basepaths.push_back(result_basepath);
        }
        else
        {
            std::cout << impath << std::endl;
            std::cout << "fit success: " << fit_success << std::endl;
        }

    }

    if (face_sizes.size() > 0) {
        //        return 0;
        /*
        // The lines below compute the median face size
        size_t ni = face_sizes.size() / 2;
        nth_element(face_sizes.begin(), face_sizes.begin()+ni, face_sizes.end());
        float median_face = face_sizes[ni];
        */

        mean_face_size = std::accumulate(face_sizes.begin(), face_sizes.end(), 0.0)/((float) face_sizes.size());

        /**
        if (set_RESIZE_COEF_via_median) {
            config::set_resize_coef(config::REF_FACE_SIZE/mean_face);
        }
        */
    }

    // get stop time, and display the timing results
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cusolverDnDestroy(handleDn);
    cudaFree( d_tmp );

    cudaFree( search_dir_Lintensity );
    cudaFree( dg_Lintensity );


    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    //!printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    ushort pixel_idx[config::Nredundant];
    bool rend_flag[config::N_TRIANGLES*NTMP];

    cudaMemcpy( pixel_idx, r.d_pixel_idx,  sizeof(ushort)*config::Nredundant, cudaMemcpyDeviceToHost );
    cudaMemcpy( rend_flag, r.d_rend_flag,  sizeof(bool)*config::N_TRIANGLES*NTMP, cudaMemcpyDeviceToHost );

    cudaFree( d_cropped_face );
    cudaFree( d_buffer_face );

    cublasDestroy(handle);

    //!std::cout << "Cleaned everything " << std::endl;

    return 0;

}



/*
 *
 *
import numpy


def gaussian_kernel(width = 7, sigma = 0.5):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)
    sigma = numpy.float32(sigma)
    filterx = x*x / (2 * sigma * sigma)
    filterx = numpy.exp(-1 * filterx)
    assert filterx.sum()>0,  'something very wrong if gaussian kernel sums to zero!'
    filterx /= filterx.sum()
    return filterx

f = gaussian_kernel()
 *
 */
