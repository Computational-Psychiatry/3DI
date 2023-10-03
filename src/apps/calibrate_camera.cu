#include "cuda.h"
#include "config.h"

#include <experimental/filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <cmath>
#include <deque>


#include <random>
#include <algorithm>

#include <vector>

#include "camera.h"

#include <glob.h> // glob(), globfree()
#include <string.h> // memset()
#include <stdexcept>
#include <sstream>

using namespace cv;
using namespace cv::dnn;

using std::vector;

std::vector<std::string> glob(const std::string& pattern);

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "You need at least one argument -- the filepath for the input video" << std::endl;
        return -1;
    }

    std::string filepattern(argv[1]);
    std::string outpath(argv[2]);

    std::vector<std::string> impaths = glob(filepattern);

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(impaths), std::end(impaths), rng);

    std::vector<cv::Mat> ims;
    for (auto& impath: impaths) {
//        ims.push_back(cv::imread(impath, cv::IMREAD_GRAYSCALE));
        ims.push_back(cv::imread(impath));
        std::cout << impath << std::endl;

        if (ims.size()>=100)
            break;
    }

//    return 0;
//    std::cout << ims.size() << std::endl;

    Camera cam;
    cam.calibrateFromFrames(ims, outpath, cv::Size(6,9)); // the pattern.png that we print from OpenCV has a grid of 7x10
//    cam.calibrateFromFrames(ims, outpath, cv::Size(10,7)); // the pattern.png that we print from OpenCV has a grid of 7x10

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





