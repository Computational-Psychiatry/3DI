#include <string>
#include <vector>
#include "camera.h"

using std::vector;

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "You need at least three arguments as follows:" << std::endl;
        std::cout << "\t ./video_undistort #input_video_path# #camera_calibration_path# #output_video_path#" << std::endl;
        return -1;
    }

    std::string input_video_path(argv[1]);
    std::string camera_calibration_path(argv[2]);
    std::string output_video_path(argv[3]);

    MassUndistorter mu(camera_calibration_path);

    mu.undistort(input_video_path, output_video_path);

    return 1;
}

