#include "cuda.h"
#include "config.h"
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
#include "video_fitter.h"


#include <glob.h> // glob(), globfree()
#include <string.h> // memset()
#include <stdexcept>
#include <sstream>

#include <opencv2/dnn.hpp>

#ifdef VISUALIZE_3D
#include "GLfuncs.h"

#endif

using std::vector;


//const int KERNEL_RADIUS=2;

int main(int argc, char** argv)
{
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    // std::string filepath = "/data/videos/treecam/baby/bv1.mp4";
    if (argc < 2) {
        std::cout << "You need at least one argument -- the filepath for the input video" << std::endl;
        return -1;
    }

    std::string filepath(argv[1]);
    std::string config_filepath(argv[2]);
    config::set_params_from_YAML_file(config_filepath);
    if (!config::check_all_necessary_files())
        return 1;

    if (argc < 3) {
        std::cout << "we need at least 2 arguments (the 2nd needs to be output dir)" << std::endl;
    }

    Camera cam0;
    float field_of_view = 40;

    std::string calibration_path("");
    if (argc >= 4) {
        if (!is_float(argv[3]))
        {
            calibration_path = argv[3];
            cam0.init(calibration_path);
        } else {
            field_of_view = std::stof(argv[3]);
        }
    }

    if (!cam0.initialized)
    {
        cv::VideoCapture tmpCap(filepath);

        int video_width = tmpCap.get(cv::CAP_PROP_FRAME_WIDTH);
        int video_height = tmpCap.get(cv::CAP_PROP_FRAME_HEIGHT);

//        std::cout << video_width << '\t' << video_height << std::endl;

        tmpCap.release();

        float cam_cx = video_width/2.0;
        float cam_cy = video_height/2.0;

//        double angle_x = 120.0f*M_PI/180.0; // angle in radians
        double angle_x = field_of_view*M_PI/180.0; // angle in radians
        double angle_y = angle_x; //60.0f*M_PI/180.0; //(cam_cy/cam_cx)*angle_x;

        float cam_alphax = cam_cx/(tan(angle_x/2.0));
        float cam_alphay = cam_alphax;
        cam0.init(cam_alphax, cam_alphay, cam_cx, cam_cy, false);
    }

    std::vector<std::vector<float> > selected_frame_xps, selected_frame_yps;
    std::vector<std::vector<float> > selected_frame_xranges, selected_frame_yranges;
    std::vector<cv::Mat> selected_frames;

    float *h_X0, *h_Y0, *h_Z0, *h_tex_mu;

    h_X0 = (float*)malloc( config::NPTS*sizeof(float) );
    h_Y0 = (float*)malloc( config::NPTS*sizeof(float) );
    h_Z0 = (float*)malloc( config::NPTS*sizeof(float) );
    h_tex_mu = (float*)malloc( config::NPTS*sizeof(float) );
    int min_x(0), max_x(0), min_y(0), max_y(0);

    std::vector< std::vector<float> > id = read2DVectorFromFile<float>(argv[4],  config::NPTS, 3);
    std::vector< std::vector<float> > tex = read2DVectorFromFile<float>(argv[5],  config::NPTS, 1);

    std::string exp_path(argv[6]), pose_path(argv[7]), illum_path(argv[8]);
    std::string outputVideoPath_3D(argv[9]), outputVideoPath_texture(argv[10]);

    for (size_t pi=0; pi<config::NPTS; ++pi)
    {
        h_X0[pi] = id[pi][0];
        h_Y0[pi] = id[pi][1];
        h_Z0[pi] = id[pi][2];
        h_tex_mu[pi] = tex[pi][0];
    }
    std::cout << "read identity" << std::endl;

    cam0.update_camera(1.0f);

    int T=1;

//    int T=9;
    VideoFitter vf(cam0, 0, 0, config::K_EPSILON,0, 0, config::K_EPSILON_L, T,
                   config::USE_TEMP_SMOOTHING, config::USE_EXP_REGULARIZATION,
                   h_X0, h_Y0, h_Z0, h_tex_mu);

    // Bind texture memories
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc desc3 = cudaCreateChannelDesc<float>();

    /*
    // Start with expression bases
    cudaBindTexture2D(0, EX_texture, vf.r.d_EX_row_major, desc, vf.r.Kepsilon, config::NPTS, vf.r.pitch);
    cudaBindTexture2D(0, EY_texture, vf.r.d_EY_row_major, desc, vf.r.Kepsilon, config::NPTS, vf.r.pitch);
    cudaBindTexture2D(0, EZ_texture, vf.r.d_EZ_row_major, desc, vf.r.Kepsilon, config::NPTS, vf.r.pitch);
    */

    float h_lambdas[3] = {-7.3627f, 51.1364f, 100.1784f};
    float h_Lintensity = 0.1;

    vf.ov.set_frame(0);
    cudaMemcpy(vf.ov.lambda, h_lambdas, sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(vf.ov.Lintensity, &h_Lintensity, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(vf.o.d_TEX_ID_NREF, 0, sizeof(float)*Nrender_estimated*3850*vf.ov.T);
    cudaMemset(vf.d_cropped_face, 0, sizeof(float)*DIMX*vf.r.T*DIMY);
    cudaMemset(vf.d_buffer_face, 0, sizeof(float)*DIMX*DIMY);


    VideoOutput vid_out(config::K_EPSILON);

    using std::vector;

    std::cout << exp_path << std::endl;
    vector<vector<float>> exp_coefs = read2DVectorFromFile_unknown_size<float>(exp_path);
    vector<vector<float>> poses = read2DVectorFromFile_unknown_size<float>(pose_path);
    vector<vector<float>> illums = read2DVectorFromFile_unknown_size<float>(illum_path);

    vf.visualize_texture(vid_out, filepath, outputVideoPath_texture, &exp_coefs, &poses, &illums);
    vf.visualize_3dmesh(vid_out, filepath, outputVideoPath_3D, &exp_coefs, &poses);

    /*
    cudaUnbindTexture(EX_texture);
    cudaUnbindTexture(EY_texture);
    cudaUnbindTexture(EZ_texture);
    */

    free(h_X0);
    free(h_Y0);
    free(h_Z0);
    free(h_tex_mu);
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





