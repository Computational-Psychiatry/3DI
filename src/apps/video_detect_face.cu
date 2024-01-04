#include <string>
#include <vector>
#include "config.h"
#include "constants.h"
#include "input_data.h"

using std::vector;

int main(int argc, char** argv)
{
    std::string video_filepath(argv[1]);
    std::string rects_filepath(argv[2]);

    if (argc < 3) {
        std::cout << "You need two arguments to run this file:" << std::endl;
        std::cout << "\t ./video_detect_face #video_filepath# #rectangles_filepath#" << std::endl;
        return -1;
    }

    if (!config::check_detector_models())
        return 1;

    LandmarkData ld;
    ld.detect_faces(video_filepath, rects_filepath);
}
