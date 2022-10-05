#ifndef GLFUNCS_H
#define GLFUNCS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void initGLwindow();

cv::Mat drawFace(const std::vector<std::vector<int> > &tl_vector, const std::vector<float>& X0, const std::vector<float>& Y0,const std::vector<float>& Z0);
cv::Mat drawFace_fromptr(const std::vector<std::vector<int> > &tl_vector, float* X0, float* Y0, float* Z0);

cv::Mat opengl2opencv();

#endif // GLFUNCS_H
