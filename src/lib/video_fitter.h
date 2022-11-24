#ifndef VIDEO_FITTER_H
#define VIDEO_FITTER_H

#include "cuda.h"
#include "config.h"
#include "constants.h"
#include <opencv2/dnn.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

#include "input_data.h"
#include "camera.h"
#include "newfuncs.h"
#include "renderer.h"
#include "logbarrier_initializer.h"
#include "derivative_computer.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <map>
#include <set>
#include <vector>

using namespace cv::dnn;

using std::set;
using std::map;
using std::vector;

struct VideoOutput
{
    map<size_t, vector<vector<float>> > exp_coefs;
    map<size_t, vector<vector<float>> > poses;
    set<int> abs_frame_ids;

    void add_exp_coeffs(const size_t frame_id, const vector<float> &cur_exp_coefs);
    void add_pose(const size_t frame_id, const vector<float> &cur_pose);


    std::vector<float> compute_avg_exp_coeffs(const size_t frame_id);
    std::vector<float> combine_exp_coeffs(const size_t frame_id);
    std::vector<float> compute_avg_pose(const size_t frame_id);

    int Kepsilon;

    VideoOutput(int _Kepsilon) : Kepsilon(_Kepsilon) {}

    void save_expressions(const std::string& exp_filename);
    void save_expressions_all(const std::string& exp_filename);
    void save_poses(const std::string& filepath, OptimizationVariables *ov=NULL, RotationComputer *rc=NULL);
};



struct VideoFitter
{
    std::vector<Camera> cams;
    size_t T;
    bool use_temp_smoothing;
    bool use_exp_regularization;

    cusolverDnHandle_t handleDn;
    cublasHandle_t handle;

//    float *h_X0, *h_Y0, *h_Z0;
//    float *h_tex_mu;

    float* d_cropped_face, *d_buffer_face;

    Net detection_net;
    Net landmark_net;
    Net leye_net, reye_net, mouth_net;
    Net correction_net;

    VideoFitter(Camera &cam0,
                const ushort Kalpha = 0, const ushort Kbeta = 0, const ushort Kepsilon = config::K_EPSILON,
                const ushort Kalpha_L= 0, const ushort Kbeta_L = 0, const ushort Kepsilon_L = config::K_EPSILON_L, size_t _Nframes=1,
                bool _use_temp_smoothing=false, bool _use_exp_regularization=false, float *_h_X0=NULL, float *_h_Y0=NULL, float *_h_Z0=NULL, float *_h_tex_mu=NULL);
    ~VideoFitter();
    Renderer r;

    OptimizationVariables ov;
    OptimizationVariables ov_lb;
    OptimizationVariables ov_lb_linesearch;
    OptimizationVariables ov_linesearch;

    Logbarrier_Initializer li_init;
    Logbarrier_Initializer li;

    Optimizer o;
    DerivativeComputer dc;

    RotationComputer rc;
    RotationComputer rc_linesearch;


    Solver s;
    Solver s_lb;
    Solver s_lambda;


    float *d_tmp;
    float *search_dir_Lintensity, *dg_Lintensity;

    int min_x, max_x, min_y, max_y;
    int FPS;

    int fit_video_frames_landmarks_sparse(const std::string& filepath,
                                              std::vector<std::vector<float> >& selected_frame_xps,
                                              std::vector<std::vector<float> >& selected_frame_yps,
                                              std::vector<std::vector<float> >& selected_frame_xranges,
                                              std::vector<std::vector<float> >& selected_frame_yranges,
                                              std::vector< cv::Mat >& selected_frames);

    bool fit_multiframe(const std::vector<std::vector<float> >& selected_frame_xps, const  std::vector<std::vector<float> >& selected_frame_yps,
          const std::vector<std::vector<float> >& selected_frame_xranges, const  std::vector<std::vector<float> >& selected_frame_yranges,
          const std::vector< cv::Mat >& selected_frames,
          float *h_X0, float *h_Y0, float *h_Z0, float *h_tex_mu, std::vector<std::string>* result_basepaths=NULL);

    VideoOutput fit_video_frames_auto(const std::string& filepath,
                              const std::string& outputVideoPath,
                              int *_min_x=NULL, int *_max_x=NULL, int *_min_y=NULL, int *_max_y=NULL);

    bool learn_identity(const std::string &filepath, float* h_X0, float* h_Y0, float *h_Z0, float *h_tex_mu);


    bool update_shape_single_resize_coef(float *xp_orig, float *yp_orig,
                            const cv::Mat& frame,  const uint frame_id, cv::VideoWriter *outputVideo, const std::string& outputVideoPath,
                            float& ref_face_size, float cur_resize_coef, std::vector<float>& exp_coeffs_combined);

    bool update_shape_single_ref_size(InputData& id, float ref_face_size, cv::VideoWriter *outputVideo,
                                                      const std::string& outputVideoPath,
                                                      VideoOutput &out);



    bool output_landmarks_expression_variation(VideoOutput& out, std::string& input_path, std::string& output_landmarks_txt,
                                      vector<vector<float> >* all_exps=NULL, vector<vector<float> >* all_poses=NULL);

    bool output_facial_parts(VideoOutput& out,
                                       std::string& input_path, std::string &output_path_sleye, std::string &output_path_sreye, std::string& output_path_smouth,
                                       vector<vector<float> >* all_exps=NULL, vector<vector<float> >* all_poses=NULL);

    bool fit_to_video_frame(InputData& id);

    bool visualize_3dmesh(VideoOutput& out, std::string &input_path, std::string &output_path, vector<vector<float> > *all_exps=NULL, vector<vector<float> > *all_poses=NULL);
    //bool visualize_3dmesh_novidin(VideoOutput& out, std::string &output_path, vector<vector<float> > *all_exps=NULL, vector<vector<float> > *all_poses=NULL);
    bool visualize_texture(VideoOutput& out, std::string &input_path, std::string &output_path, vector<vector<float> > *all_exps=NULL, vector<vector<float> > *all_poses=NULL);

    bool generate_texture(int subj_id, int imwidth, const std::string& out_dir_root, const float fov_data, const float tz);

};

#endif // VIDEO_FITTER_H
