/*
 * preprocessing.h
 *
 *  Created on: Aug 1, 2020
 *      Author: sariyanide
 */

#ifndef PREPREPCESSING_H_
#define PREPREPCESSING_H_




#include <deque>
#include <iostream>
#include <memory>
#include <iomanip>
#include <string>
#include <vector>

#include <ostream>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "constants.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;


const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.3; // was 0.1 during experiments, but 0.2 makes more sense // 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

enum PART { LEYE, REYE, MOUTH };


const std::string caffeConfigFile = "models/deploy.prototxt";
const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "models/opencv_face_detector_uint8.pb";

cv::Rect detectFaceOpenCVDNN_multiscale(Net net, Mat &frameOpenCVDNN, string framework, double *confidence=NULL);
cv::Rect detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN, string framework, double *confidence=NULL, Rect *ROI=NULL);
cv::Point transform_pt(float px, float py, const cv::Point2f& center, float scale, float resolution, bool invert = false);

double st_dev(vector<float> v, float ave);


//void detect_landmarks_opencv(Net& detection_net, Net &landmark_net,
//                      const std::string& framework, cv::Mat &frame,
//                      std::vector<float>& xp_vec, std::vector<float>& yp_vec, float &bbox_size);

void detect_landmarks_opencv(const cv::Rect& d, double face_confidence, Net& landmark_net, Net& leye_net, Net& reye_net, Net& mouth_net, Net &correction_net, cv::Mat &frame,
                             std::vector<float>& xp_vec, std::vector<float>& yp_vec, float &bbox_size,
                             std::vector<float> &xrange, std::vector<float> &yrange, bool use_local_models, bool plot=false,
                             vector<vector<double> > *xs = NULL, vector<vector<double> > *ys = NULL);

void detect_landmarks_opencv_single(const cv::Rect& d, double face_confidence, Net& landmark_net, Net& leye_net, Net& reye_net, Net& mouth_net, Net &correction_net, cv::Mat &frame,
                             std::vector<float>& xp_vec, std::vector<float>& yp_vec, float &bbox_size,
                             std::vector<float> &xrange, std::vector<float> &yrange, bool use_local_models, bool plot=false);


cv::Rect detect_face_opencv(Net& detection_net, const std::string& framework, cv::Mat &frame, Rect *prev_d=NULL, double* face_confidence=NULL, bool multiscale_start=false);


std::pair<std::vector<float>, std::vector<float>> compute_landmarks_wlocal_models(cv::Mat& frame, int face_size,
                                                                                  std::vector<float>& xp_vec0, std::vector<float>& yp_vec0, cv::Mat& part,
                                                                                  Net& leye_net, Net& reye_net, Net& mouth_net, double tx_rate, double ty_rate);

void paint_innermouth_black(cv::Mat& frame, const std::vector<float>& xp_vec, const std::vector<float>& yp_vec);

Point2f transform_pt(float px, float py, const cv::Point2f* center, float* scale, float* resolution, bool invert = false);

void heatmaps_to_landmarks(cv::Mat* netOut, cv::Mat* netOut_flipped,
                           std::vector<float>& xp_vec, std::vector<float>& yp_vec,
                           cv::Point2f* ptCenter, uint num_landmarks,
                           float* scale, float* resolution, bool do_flip=false, bool do_transformation=true);


void crop_part(int part_id, cv::Mat& im, double face_size, std::vector<float>& xp_vec, std::vector<float>& yp_vec,
               double tx_rate, double ty_rate, double& rescale_rate, int& top, int& left, cv::Mat& part);



#endif /* FUNCS_H_ */
