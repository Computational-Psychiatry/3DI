#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

using std::vector;
using std::deque;


struct InputData
{
    size_t T;
    deque<cv::Mat> frames;
    deque<float> face_sizes;
    deque<size_t> abs_frame_ids;
    deque<vector<float>> xp_origs; // "Original" location of landmarks
    deque<vector<float>> yp_origs; // (i.e., no resize is done, but located on undistorted image)

    InputData(size_t _T) : T(_T) {}
    void add_data(const cv::Mat& frame, const vector<float>& xp, const vector<float> &yp, size_t fi, float face_size);
    void get_resized_landmarks(size_t rel_frame_id, const float resize_coef, float *xp, float *yp);
    void get_resized_frame(size_t rel_frame_id, const float resize_coef, cv::Mat& frame_dst);
    void clear();
};




struct LandmarkData
{
    vector<vector<float> > xp_vecs;
    vector<vector<float> > yp_vecs;

    cv::dnn::Backend LANDMARK_DETECTOR_BACKEND = cv::dnn::DNN_BACKEND_CUDA;
    cv::dnn::Target LANDMARK_DETECTOR_TARGET = cv::dnn::DNN_TARGET_CUDA;

    LandmarkData() {}
    LandmarkData(const std::string& landmarks_path);
    LandmarkData(const std::string& video_path, const std::string& faces_path, const std::string& landmarks_path);

    void fill_xpypvec(vector<vector<float> > &all_lmks);
    void init_from_txtfile(const std::string& landmarks_path);
    void init_from_videofile(const std::string &video_path);
    size_t get_num_frames() { return xp_vecs.size(); };

    vector<float> get_xpvec(size_t t) { return xp_vecs[t]; }
    vector<float> get_ypvec(size_t t) { return yp_vecs[t]; }

    int get_face_size(size_t t);

    static bool check_CUDA(cv::dnn::dnn4_v20230620::Backend &LANDMARK_DETECTOR_BACKEND, cv::dnn::dnn4_v20230620::Target &LANDMARK_DETECTOR_TARGET);

    vector<vector<float> > detect_faces(const std::string& filepath, const std::string& rects_filepath);
    vector<vector<float> > detect_landmarks(const std::string &video_filepath, const vector<vector<float> > &face_rects, const std::string &landmarks_filepath);

};

#endif // INPUT_DATA_H
