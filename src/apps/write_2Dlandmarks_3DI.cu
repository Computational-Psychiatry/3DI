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

int fit_video_frames_landmarks_sparse(Camera &cam0,
                                      const std::string& filepath,
                                      std::vector<std::vector<float> >& selected_frame_xps,
                                      std::vector<std::vector<float> >& selected_frame_yps,
                                      std::vector<std::vector<float> >& selected_frame_xranges,
                                      std::vector<std::vector<float> >& selected_frame_yranges,
                                      std::vector< cv::Mat >& selected_frames,
                                      int* min_x, int* max_x, int* min_y, int* max_y, int frame_offset=0);

using std::vector;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "You need at least one argument -- the filepath for the input video" << std::endl;
        return -1;
    }

    std::string filepath(argv[1]);
    std::string config_filepath(argv[2]);
    config::set_params_from_YAML_file(config_filepath);

    if (argc < 3) {
        std::cout << "we need at least 2 arguments (the 2nd needs to be output dir)" << std::endl;
    }

    Camera cam0;
    float field_of_view = 40;

    //    std::string calibration_path = "./models/cameras/TreeCam_1041a.txt";
    std::string calibration_path("");
    if (argc >= 5) {
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
    std::string exp_path(argv[5]), pose_path(argv[6]);
    std::string output_landmarks_path(argv[7]);

    for (size_t pi=0; pi<config::NPTS; ++pi)
    {
        h_X0[pi] = id[pi][0];
        h_Y0[pi] = id[pi][1];
        h_Z0[pi] = id[pi][2];
        h_tex_mu[pi] = 0;
    }

    std::cout << "read identity" << std::endl;

    cam0.update_camera(1.0f);

    int T=1;
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

    // Start with expression bases
    cudaBindTexture2D(0, EX_texture, vf.r.d_EX_row_major, desc, vf.r.Kepsilon, config::NPTS, vf.r.pitch);
    cudaBindTexture2D(0, EY_texture, vf.r.d_EY_row_major, desc, vf.r.Kepsilon, config::NPTS, vf.r.pitch);
    cudaBindTexture2D(0, EZ_texture, vf.r.d_EZ_row_major, desc, vf.r.Kepsilon, config::NPTS, vf.r.pitch);


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

    vf.output_landmarks(vid_out, filepath, output_landmarks_path, &exp_coefs, &poses);

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
        const ushort* __restrict__ triangle_idx, const float* R__,
        int N_TRIANGLES, uint Nredundant)
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
        const ushort* __restrict__ triangle_idx, const ushort Kalpha,
        const int N_TRIANGLES)
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












int fit_video_frames_landmarks_sparse(Camera &cam0,
                                      const std::string& filepath,
                                      std::vector<std::vector<float> >& selected_frame_xps,
                                      std::vector<std::vector<float> >& selected_frame_yps,
                                      std::vector<std::vector<float> >& selected_frame_xranges,
                                      std::vector<std::vector<float> >& selected_frame_yranges,
                                      std::vector< cv::Mat >& selected_frames,
                                      int* min_x, int* max_x, int* min_y, int* max_y, int frame_offset)
{
    const std::string caffeConfigFile = "models/deploy.prototxt";
    const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    std::string device = "CPU";
    std::string framework = "caffe";

    Net detection_net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    // detection_net.setPreferableBackend(DNN_BACKEND_CUDA);
    // detection_net.setPreferableTarget(DNN_TARGET_CUDA);

    std::string tfLandmarkNet("models/landmark_models/model_FAN_frozen.pb");
    Net landmark_net = cv::dnn::readNetFromTensorflow(tfLandmarkNet);
    landmark_net.setPreferableBackend(DNN_BACKEND_CUDA);
    landmark_net.setPreferableTarget(DNN_TARGET_CUDA);

    Net leye_net = cv::dnn::readNetFromTensorflow("models/landmark_models/m-64l64g0-64-128-5121968464leye.pb");
    leye_net.setPreferableBackend(DNN_BACKEND_CUDA);
    leye_net.setPreferableTarget(DNN_TARGET_CUDA);

    Net reye_net = cv::dnn::readNetFromTensorflow("models/landmark_models/m-64l64g0-64-128-5121968464reye.pb");
    reye_net.setPreferableBackend(DNN_BACKEND_CUDA);
    reye_net.setPreferableTarget(DNN_TARGET_CUDA);

    Net mouth_net = cv::dnn::readNetFromTensorflow("models/landmark_models/m-64l64g0-64-128-5121968464mouth.pb");
    mouth_net.setPreferableBackend(DNN_BACKEND_CUDA);
    mouth_net.setPreferableTarget(DNN_TARGET_CUDA);

    Net correction_net = cv::dnn::readNetFromTensorflow("models/landmark_models/model_correction.pb");

    cv::VideoCapture capture(filepath);
    int FPS = capture.get(cv::CAP_PROP_FPS);

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    cv::Mat frame;

    vector< vector<float> > all_xp_vec, all_yp_vec;
    vector< vector<float> > all_xrange, all_yrange;

    vector<float> face_sizes;
    vector<float> minxs, minys, maxxs, maxys;
    int idx = 0;
    while (true) {
        idx++;

        all_xp_vec.push_back(std::vector<float>());
        all_yp_vec.push_back(std::vector<float>());

        all_xrange.push_back(std::vector<float>());
        all_yrange.push_back(std::vector<float>());

        capture >> frame;

        if (frame.empty())
            break;

        if (cam0.cam_remap) {
            cv::remap(frame, frame, cam0.map1, cam0.map2, cv::INTER_LINEAR);
        }

        if (idx == 1) {
            if (min_x != NULL)
                *min_x = frame.cols;

            if (max_x != NULL)
                *max_x = 0;

            if (min_y != NULL)
                *min_y = frame.rows;

            if (max_y != NULL)
                *max_y = 0;
        }

        if (idx < FPS*config::SKIP_FIRST_N_SECS)
            continue;
        /*
        if (idx > 2700)
            break;
        */

        // PUTBACK
        //if (idx % 10 != frame_offset)
        if ((idx % FPS*config::EVERY_N_SECS) != frame_offset)
            continue;
        //        if (idx % 120 != 0)
        //            continue;

        float face_size;


        try {
            cv::Rect ROI(-1,-1,-1,-1);
            double face_confidence;
            cv::Rect face_rect = detect_face_opencv(detection_net, framework, frame, &ROI, &face_confidence, true);
            detect_landmarks_opencv(face_rect, face_confidence, landmark_net, leye_net, reye_net, mouth_net, correction_net, frame,
                                    all_xp_vec[all_xp_vec.size()-1], all_yp_vec[all_yp_vec.size()-1], face_size,
                    all_xrange[all_xrange.size()-1], all_yrange[all_yrange.size()-1], config::USE_LOCAL_MODELS, false);


        }
        catch (cv::Exception)
        {
            std::cout << "Skipping this one" << std::endl;
            continue;
        }

        if (face_size == -1.0f)
            continue;

        //        cv::imshow("heyb", frame);
        //        cv::waitKey(20);

        std::vector<float>& xcur = all_xp_vec[all_xp_vec.size()-1];
        std::vector<float>& ycur = all_yp_vec[all_yp_vec.size()-1];

        if (xcur.size() == 0)
            continue;

        int cur_xmin = (int) *std::min_element(xcur.begin(), xcur.end());
        int cur_xmax = (int) *std::max_element(xcur.begin(), xcur.end());

        int cur_ymin = (int) *std::min_element(ycur.begin(), ycur.end());
        int cur_ymax = (int) *std::max_element(ycur.begin(), ycur.end());

        minxs.push_back(cur_xmin);
        minys.push_back(cur_ymin);

        maxxs.push_back(cur_xmax);
        maxys.push_back(cur_ymax);

        if (cur_xmin < *min_x)
            *min_x = cur_xmin;

        if (cur_xmax > *max_x)
            *max_x = cur_xmax;

        if (cur_ymin < *min_y)
            *min_y = cur_ymin;

        if (cur_ymax > *max_y)
            *max_y = cur_ymax;

//        std::cout << idx << std::endl;

        // !PUTBACK
        //        if (idx >= 6000)
        //            break;
        if (idx >= 5000)
            break;
        /*
*/

        face_sizes.push_back(face_size);
//        std::cout << "face size is " << face_size << std::endl;
    }

    // The lines below compute the median face size
    size_t ni = face_sizes.size() / 2;
    nth_element(face_sizes.begin(), face_sizes.begin()+ni, face_sizes.end());
    float median_face = face_sizes[ni];

    nth_element(minxs.begin(), minxs.begin()+ni, minxs.end());
    nth_element(minys.begin(), minys.begin()+ni, minys.end());
    nth_element(maxxs.begin(), maxxs.begin()+ni, maxxs.end());
    nth_element(maxys.begin(), maxys.begin()+ni, maxys.end());

    float median_face_width = maxxs[ni]-minxs[ni];
    float median_face_height = maxys[ni]-minys[ni];

    *min_x = minxs[ni]-median_face_width/1.5;
    *max_x = maxxs[ni]+median_face_width/1.5;

    *min_y = minys[ni]-median_face_height/1.5;
    *max_y = maxys[ni]+median_face_height/1.5;




//    std::cout << "Cleaned everything " << std::endl;
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





