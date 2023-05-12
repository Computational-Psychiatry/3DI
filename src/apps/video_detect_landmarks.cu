#include <string>
#include <vector>

#include "config.h"
#include "funcs.h"
#include "constants.h"
#include "input_data.h"

using std::vector;

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "You need at least one arguments" << std::endl;
        std::cout << "usage is as follows (arguments within [brackets] are optional):" << std::endl;
        std::cout << "\t ./video_detect_landmarks #video_filepath# #rectangles_filepath# #output_landmarks_filepath# [#config_filepath#]" << std::endl;
        return -1;
    }

    std::string video_filepath(argv[1]);
    std::string rects_filepath(argv[2]);
    std::string landmarks_filepath(argv[3]);

    std::string config_filepath("./config/L/G1.txt");

    if (argc >= 5)
        config_filepath = std::string(argv[4]);

    config::set_params_from_YAML_file(config_filepath);
    LandmarkData ld(video_filepath, rects_filepath, landmarks_filepath);
    return 1;
}

