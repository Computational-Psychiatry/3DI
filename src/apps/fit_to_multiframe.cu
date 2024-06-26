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

#include "input_data.h"

#include <string.h> // memset()
#include <stdexcept>
#include <sstream>

#include <opencv2/dnn.hpp>

#ifdef VISUALIZE_3D
#include "GLfuncs.h"

#endif

using std::string;
using std::vector;


bool fit_multiframe(Camera &cam, Renderer &r, const std::vector<std::vector<float> >& selected_frame_xps, const  std::vector<std::vector<float> >& selected_frame_yps,
                   const std::vector<std::vector<float> >& selected_frame_xranges, const  std::vector<std::vector<float> >& selected_frame_yranges,
                   const std::vector< cv::Mat >& selected_frames,
                   float *h_X0, float *h_Y0, float *h_Z0, float *h_tex_mu, std::vector<std::string>* result_basepaths=NULL);

int create_data_for_multiframe(Renderer &r, const std::string& outdir, Camera &cam0,
                               vector<vector<float> >& xps, vector<vector<float> >& yps,
                               vector<vector<float> >& xranges, vector<vector<float> >& yranges,
                               vector<cv::Mat> &selected_frames, vector<std::string>& result_basepaths, const std::vector<string>& imfiles,
                               cv::dnn::Net &detection_net, cv::dnn::Net &landmark_net, cv::dnn::Net &leye_net, cv::dnn::Net &reye_net, cv::dnn::Net &mouth_net, cv::dnn::Net &correction_net,
                               float &mean_face_size,
                               bool set_RESIZE_COEF_via_median=true, int combination_id = -1);


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

    int NFRAMES=-1;
    if (argc >= 6)
        NFRAMES=std::stoi(argv[5]);
        

    if (argc >= 7) {
        if (!is_float(argv[6]))
        {
            calibration_path = argv[6];
            cam0.init(calibration_path);
        } else {
            field_of_viewy = std::stof(argv[6]);
            // Camera will be initialized later -- if necessary
        }
    }
    else
        field_of_viewy = field_of_view;

    std::ifstream inFile;
    inFile.open(argv[1]); //open the input file

    std::string config_filepath(argv[2]);
    config::set_params_from_YAML_file(config_filepath);
    if (NFRAMES != -1)
        config::NFRAMES = NFRAMES;
    
    if (!config::check_all_necessary_files())
        return 1;


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


    cv::dnn::Backend LANDMARK_DETECTOR_BACKEND(cv::dnn::DNN_BACKEND_CUDA);
    cv::dnn::Target LANDMARK_DETECTOR_TARGET(cv::dnn::DNN_TARGET_CUDA);

    LandmarkData::check_CUDA(LANDMARK_DETECTOR_BACKEND, LANDMARK_DETECTOR_TARGET);

    Renderer r(config::NFRAMES, config::NID_COEFS, config::NTEX_COEFS, config::K_EPSILON, true, true, true);

    cv::dnn::Net detection_net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    detection_net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

    cv::dnn::Net landmark_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_MPATH);
    landmark_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    landmark_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::dnn::Net leye_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_LEYE_MPATH);
    leye_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    leye_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::dnn::Net reye_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_REYE_MPATH);
    reye_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    reye_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::dnn::Net mouth_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_MOUTH_MPATH);
    mouth_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    mouth_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::dnn::Net correction_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_CORRECTION_MPATH);

    int combination_id = 0;

    float mean_face_size = -1.0f;
    for (auto cur_combination : imfile_combinations)
    {
        std::vector<std::string> imfiles = str_split(cur_combination, ',');

        if (imfiles.size() < config::NFRAMES) {
            std::cout << "Not enough files in this combination; skipping comb" << std::endl;
            continue;
        }

        if (!cam0.initialized)
        {
            cv::Mat im = cv::imread(imfiles[0]);

            float cam_cx = im.cols/2.0;
            float cam_cy = im.rows/2.0;

            double angle_x = field_of_view*M_PI/180.0; // angle in radians
            //double angle_y = angle_x; //60.0f*M_PI/180.0; //(cam_cy/cam_cx)*angle_x;
            double angle_y = field_of_viewy*M_PI/180.0; //60.0f*M_PI/180.0; //(cam_cy/cam_cx)*angle_x;

            float cam_alphax = cam_cx/(tan(angle_x/2.0));
            float cam_alphay = cam_cy/(tan(angle_y/2.0));
            if (field_of_view == field_of_viewy) {
                cam_alphay = cam_alphax;
            }
            cam0.init(cam_alphax, cam_alphay, cam_cx, cam_cy, false);
        }

        combination_id++;

        std::cout << "COMBINATION:  " << cur_combination << std::endl;
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

    return 0;
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

    bool success = fit_to_multi_images(cams, selected_frame_xps, selected_frame_yps,
                                selected_frame_xranges, selected_frame_yranges, selected_frames,
                                result_basepaths, r,  o, handleDn,
                                handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, ov_lb, ov_lb_linesearch, rc,
                                rc_linesearch,  dc, s, s_lambda, d_tmp, search_dir_Lintensity, dg_Lintensity,
                            h_X0, h_Y0, h_Z0, h_tex_mu);

    std::cout << "Success: "<< success << std::endl;

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cusolverDnDestroy(handleDn);
    cudaFree( d_tmp );

    cudaFree( search_dir_Lintensity );
    cudaFree( dg_Lintensity );


    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    cudaFree( d_cropped_face );
    cudaFree( d_buffer_face );

    cublasDestroy(handle);

    return success;

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
