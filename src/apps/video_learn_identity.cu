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



using std::vector;

int create_data_for_multiframe(const std::string& imdir, Renderer &r, const std::string& outdir, const uint subj_id, float fovx, float fovy,
                               vector<vector<float> >& xps, vector<vector<float> >& yps,
                               vector<vector<float> >& xranges, vector<vector<float> >& yranges,
                               vector<cv::Mat> &selected_frames, vector<std::string>& result_basepaths, const std::vector<int> &angle_idx,
                               cv::dnn::Net &detection_net, cv::dnn::Net &landmark_net, cv::dnn::Net &leye_net, cv::dnn::Net &reye_net, cv::dnn::Net &mouth_net, cv::dnn::Net &correction_net,
                               bool set_RESIZE_COEF_via_median=true, int combination_id = -1);


int main(int argc, char** argv)
{
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    if (argc < 2) {
        std::cout << "You need at least one argument -- the filepath for the input video" << std::endl;
        return -1;
    }

    std::string video_path(argv[1]);
    std::string landmarks_path(argv[2]);
    std::string config_filepath(argv[3]);
    config::set_params_from_YAML_file(config_filepath);

    if (!config::check_all_necessary_files())
        return 1;

    config::OPTS_DELTA_EPS = 0.4f;

    if (argc < 3) {
        std::cout << "we need at least 2 arguments (the 2nd needs to be output dir)" << std::endl;
    }


    Camera cam0;
    float field_of_view = 40;

    //    std::string calibration_path = "./models/cameras/TreeCam_1041a.txt";
    std::string calibration_path("");
    if (argc >= 5) {
        if (!is_float(argv[4]))
        {
            calibration_path = argv[4];
            cam0.init(calibration_path);
        } else {
            field_of_view = std::stof(argv[4]);
        }
    }

    std::string shpCoeffsPath(argv[5]);
    std::string texCoeffsPath(argv[6]);

    LandmarkData ld(landmarks_path);

    if (std::experimental::filesystem::exists(shpCoeffsPath) && std::experimental::filesystem::exists(texCoeffsPath))
        return 0;

    if (!cam0.initialized)
    {
        cv::VideoCapture tmpCap(video_path);

        int video_width = tmpCap.get(cv::CAP_PROP_FRAME_WIDTH);
        int video_height = tmpCap.get(cv::CAP_PROP_FRAME_HEIGHT);

        if (config::PRINT_DEBUG)
            std::cout << video_width << '\t' << video_height << std::endl;

        tmpCap.release();

        float cam_cx = video_width/2.0;
        float cam_cy = video_height/2.0;

        double angle_x = field_of_view*M_PI/180.0; // angle in radians
        //!double angle_y = angle_x; //60.0f*M_PI/180.0; //(cam_cy/cam_cx)*angle_x;

        float cam_alphax = cam_cx/(tan(angle_x/2.0));
        float cam_alphay = cam_alphax; //cam_cy/(tan(angle_y/2.0));

        cam0.init(cam_alphax, cam_alphay, cam_cx, cam_cy, false);
    }

    VideoFitter vf_identity(cam0,
                            config::NID_COEFS, config::NTEX_COEFS, config::K_EPSILON,
                            config::K_ALPHA_L, 0, config::K_EPSILON_L, config::NFRAMES,
                            false, false);

    std::vector<float> h_alphas(vf_identity.ov.Kalpha, 0.0f), h_betas(vf_identity.ov.Kbeta, 0.0f);


    ///////////////////////////////////////////
    ///////////////////////////////////////////
    std::cout << "Learning the 3D identity of subject in video ... this may take a few minutes" << std::endl;
    vf_identity.learn_identity(video_path, ld, &h_alphas[0], &h_betas[0]);

    write_1d_vector<float>(shpCoeffsPath, h_alphas);
    write_1d_vector<float>(texCoeffsPath, h_betas);

    std::cout << "\tDone" << std::endl;


    /****
    if (config::OUTPUT_IDENTITY) {
        write_identity(identityPath, &h_X0[0], &h_Y0[0], &h_Z0[0]);
        write_texture(texturePath, &h_tex_mu[0]);
    }
    *****/

    return 0;
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





