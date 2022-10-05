#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <iostream>
#include <opencv2/opencv.hpp>

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

#endif // INPUT_DATA_H
