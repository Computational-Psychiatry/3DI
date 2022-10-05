#include "cuda.h"
#include "config.h"
#include "video_fitter.h"
#include <experimental/filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <cmath>
#include <deque>

#include <vector>
#include <stdio.h>
#include <numeric>
#include <random>
#include <algorithm>    // std::shuffle
#include <set>

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


using namespace cv;
using namespace cv::dnn;

using std::vector;

// these exist on the GPU side
texture<float,2> EX_texture;
texture<float,2> EY_texture;
texture<float,2> EZ_texture;

texture<float,2> IX_texture;
texture<float,2> IY_texture;
texture<float,2> IZ_texture;

texture<float,2> TEX_texture;

using std::vector;

int create_data_for_multiframe(const std::string& imdir, Renderer &r, const std::string& outdir, const uint subj_id, float fovx, float fovy,
                               vector<vector<float> >& xps, vector<vector<float> >& yps,
                               vector<vector<float> >& xranges, vector<vector<float> >& yranges,
                               vector<Mat> &selected_frames, vector<std::string>& result_basepaths, const std::vector<int> &angle_idx,
                               Net &detection_net, Net &landmark_net, Net &leye_net, Net &reye_net, Net &mouth_net, Net &correction_net,
                               bool set_RESIZE_COEF_via_median=true, int combination_id = -1);

//const int KERNEL_RADIUS=2;

// /media/v/SSD1TB/dataset/videos/treecam/ML/ML0001.mp4 /media/v/SSD1TB/dataset/videos/treecam/ML/ML0001.mp4.avi
// /media/v/SSD1TB/dataset/Florence/images/for_experiments/ /media/v/SSD1TB/dataset/Florence/results/3DIv2

// /media/v/SSD1TB/dataset/BU4DFE/images/ /media/v/SSD1TB/dataset/BU4DFE/results/3DIv2
///data/videos/treecam/1041a/test_RA2_NA.mkv /data/videos/treecam/ML/output/ 15


int main(int argc, char** argv)
{
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //    std::string filepath = "/data/videos/treecam/baby/bv1.mp4";
    // /media/v/Samsung_T5/dataset/videos/treecam/ML/ML0001.mp4 /media/v/Samsung_T5/dataset/videos/treecam/ML/ML0001.mp4.avi
    // /data/videos/treecam/ML/ML0001.mp4 /data/videos/treecam/ML/ML0001_TT.avi
    if (argc < 2) {
        std::cout << "You need at least one argument -- the filepath for the input video" << std::endl;
        return -1;
    }

    std::string config_filepath(argv[3]);
    config::set_params_from_YAML_file(config_filepath);

    std::string filepath(argv[1]);

    std::string outputVideoPath("output.avi");
    if (argc < 3) {
        std::cout << "we need at least 2 arguments (the 2nd needs to be output dir)" << std::endl;
    }

    std::string outputVideoDir = argv[2];

    if (!std::experimental::filesystem::exists(outputVideoDir))
        std::experimental::filesystem::create_directory(outputVideoDir);

    if (config::OUTDIR_WITH_PARAMS)
        outputVideoDir += "/" + config::get_key();
    else
        outputVideoDir += "/";

    if (!std::experimental::filesystem::exists(outputVideoDir))
        std::experimental::filesystem::create_directory(outputVideoDir);

    Camera cam0;
    float field_of_view = 40;

    //    std::string calibration_path = "./models/cameras/TreeCam_1041a.txt";
    std::string calibration_path("");
    if (argc == 5) {
        if (!is_float(argv[4]))
        {
            calibration_path = argv[4];
            cam0.init(calibration_path);
        } else {
            field_of_view = std::stof(argv[4]);

            outputVideoDir += std::string("/") + argv[4];

            if (!std::experimental::filesystem::exists(outputVideoDir))
                std::experimental::filesystem::create_directory(outputVideoDir);
        }
    }


    outputVideoPath = outputVideoDir + "/" + remove_extension(base_name(filepath)) + ".avi";


    std::string identityPath = remove_extension(outputVideoPath) + ".id.txt";
    std::string texturePath = remove_extension(outputVideoPath) + ".tex.txt";

    /*
    if (std::experimental::filesystem::exists(identityPath))
        return 0;
        */

    if (!cam0.initialized)
    {
        cv::VideoCapture tmpCap(filepath);

        int video_width = tmpCap.get(cv::CAP_PROP_FRAME_WIDTH);
        int video_height = tmpCap.get(cv::CAP_PROP_FRAME_HEIGHT);

        if (config::PRINT_DEBUG)
            std::cout << video_width << '\t' << video_height << std::endl;

        tmpCap.release();

        float cam_cx = video_width/2.0;
        float cam_cy = video_height/2.0;

//        double angle_x = 120.0f*M_PI/180.0; // angle in radians
        double angle_x = field_of_view*M_PI/180.0; // angle in radians
        double angle_y = angle_x; //60.0f*M_PI/180.0; //(cam_cy/cam_cx)*angle_x;

        float cam_alphax = cam_cx/(tan(angle_x/2.0));
        float cam_alphay = cam_alphax; //cam_cy/(tan(angle_y/2.0));

        /*
        std::cout << "cam_alphax: " << cam_alphax << std::endl;
        std::cout << "cam_alphay: " << cam_alphay << std::endl;
        */

        cam0.init(cam_alphax, cam_alphay, cam_cx, cam_cy, false);
    }





    float *h_X0, *h_Y0, *h_Z0, *h_tex_mu;
    h_X0 = (float*)malloc( NPTS*sizeof(float) );
    h_Y0 = (float*)malloc( NPTS*sizeof(float) );
    h_Z0 = (float*)malloc( NPTS*sizeof(float) );
    h_tex_mu = (float*)malloc( NPTS*sizeof(float) );

    {
        VideoFitter vf_identity(cam0,
                                NID_COEFS, NTEX_COEFS, config::K_EPSILON,
                                K_ALPHA_L, 0, config::K_EPSILON_L, config::NFRAMES,
                                false, false);



        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<float>();
        cudaChannelFormatDesc desc3 = cudaCreateChannelDesc<float>();

        // Start with expression bases
        cudaBindTexture2D(0, EX_texture, vf_identity.r.d_EX_row_major, desc, config::K_EPSILON, NPTS, vf_identity.r.pitch);
        cudaBindTexture2D(0, EY_texture, vf_identity.r.d_EY_row_major, desc, config::K_EPSILON, NPTS, vf_identity.r.pitch);
        cudaBindTexture2D(0, EZ_texture, vf_identity.r.d_EZ_row_major, desc, config::K_EPSILON, NPTS, vf_identity.r.pitch);

        // Now identity bases
        if (vf_identity.r.use_identity)
        {
            cudaBindTexture2D(0, IX_texture, vf_identity.r.d_IX_row_major, desc2, NID_COEFS, NPTS, vf_identity.r.pitch2);
            cudaBindTexture2D(0, IY_texture, vf_identity.r.d_IY_row_major, desc2, NID_COEFS, NPTS, vf_identity.r.pitch2);
            cudaBindTexture2D(0, IZ_texture, vf_identity.r.d_IZ_row_major, desc2, NID_COEFS, NPTS, vf_identity.r.pitch2);
        }

        // Finally the texture bases
        if (vf_identity.r.use_texture)
        {
            cudaBindTexture2D(0, TEX_texture, vf_identity.r.d_TEX_row_major, desc3, NTEX_COEFS, NPTS, vf_identity.r.pitch3);
        }

        ///////////////////////////////////////////
        ///////////////////////////////////////////
        std::cout << "Learning the 3D identity of subject in video ... this may take a few minutes" << std::endl;
        vf_identity.learn_identity(filepath, h_X0, h_Y0, h_Z0, h_tex_mu);
        std::cout << "\tDone" << std::endl;
        ///////////////////////////////////////////
        ///////////////////////////////////////////


        if (vf_identity.r.use_identity) {
            cudaUnbindTexture(IX_texture);
            cudaUnbindTexture(IY_texture);
            cudaUnbindTexture(IZ_texture);
        }

        if (vf_identity.r.use_texture) {
            cudaUnbindTexture(TEX_texture);
        }

        cudaUnbindTexture(EX_texture);
        cudaUnbindTexture(EY_texture);
        cudaUnbindTexture(EZ_texture);
    }




    if (config::OUTPUT_IDENTITY) {
        write_identity(identityPath, h_X0, h_Y0, h_Z0);
        write_texture(texturePath, h_tex_mu);
    }

    /*
    if (true) {

        free(h_X0);
        free(h_Y0);
        free(h_Z0);
        free(h_tex_mu);
        return 0;
    }
*/
    cam0.update_camera(1.0f);
    VideoFitter vf(cam0, 0, 0, config::K_EPSILON,
                   0, 0, config::K_EPSILON_L, config::TIME_T,
                    config::USE_TEMP_SMOOTHING, config::USE_EXP_REGULARIZATION,
                    h_X0, h_Y0, h_Z0, h_tex_mu);



    // Bind texture memories
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc desc3 = cudaCreateChannelDesc<float>();

    // Start with expression bases
    cudaBindTexture2D(0, EX_texture, vf.r.d_EX_row_major, desc, vf.r.Kepsilon, NPTS, vf.r.pitch);
    cudaBindTexture2D(0, EY_texture, vf.r.d_EY_row_major, desc, vf.r.Kepsilon, NPTS, vf.r.pitch);
    cudaBindTexture2D(0, EZ_texture, vf.r.d_EZ_row_major, desc, vf.r.Kepsilon, NPTS, vf.r.pitch);



    std::cout << "Processing video ..."  << std::endl;
    VideoOutput vid_out = vf.fit_video_frames_auto(filepath, outputVideoPath);
    std::cout << "\tDone" << std::endl;

    std::string outputVideoPath_3D(""), outputVideoPath_texture("");

    outputVideoPath_3D = outputVideoPath;
    outputVideoPath_3D.replace(outputVideoPath_3D.end()-4,outputVideoPath_3D.end(), "_3Drenders.avi");

    outputVideoPath_texture = outputVideoPath;
    outputVideoPath_texture.replace(outputVideoPath_texture.end()-4,outputVideoPath_texture.end(), "_texture.avi");

    std::string exp_path = outputVideoPath;
    exp_path.replace(exp_path.end()-4,exp_path.end(), ".expressions");

    std::string pose_path = outputVideoPath;
    pose_path.replace(pose_path.end()-4,pose_path.end(), ".poses");

    vid_out.save_expressions(exp_path);
    vid_out.save_poses(pose_path);

    if (config::OUTPUT_VISUALS)
    {
        std::cout << "Creating texture video ..." << std::endl;
        vf.visualize_texture(vid_out, filepath, outputVideoPath_texture);
        std::cout << "\tDone" << std::endl;

        std::cout << "Creating render video ..." << std::endl;
        vf.visualize_3dmesh(vid_out, filepath, outputVideoPath_3D);
        std::cout << "\tDone" << std::endl;
    }


    cudaUnbindTexture(EX_texture);
    cudaUnbindTexture(EY_texture);
    cudaUnbindTexture(EZ_texture);


    free(h_X0);
    free(h_Y0);
    free(h_Z0);
    free(h_tex_mu);
}









__global__ void render_expression_basis_texture_colmajor_rotated2(
        const float* __restrict__ alphas, const float* __restrict__ betas, const float* __restrict__ gammas,
        const uint* __restrict__ indices, const int Nunique_pixels, const ushort* __restrict__ tl,
        float*  __restrict__ REX, float*  __restrict__ REY, float*  __restrict__ REZ,
        const ushort* __restrict__ triangle_idx, const float* R__)
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
        const ushort* __restrict__ triangle_idx, const ushort Kalpha)
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
        const ushort Kbeta)
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





