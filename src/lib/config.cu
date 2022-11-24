// In config.cpp

#include <sstream>
#include <iomanip>
#include <iostream>
#include <set>
#include "config.h"

namespace config
{
    double OPTS_DELTA = 0.6;
    double OPTS_DELTA_BETA = 2.0;

    double REF_FACE_SIZE = 86; //120; // 84;
//    double RESIZE_COEF = 0.50;

    double CONFIDENCE_RANGE = 0.42;
//    double CONFIDENCE_RANGE = 0.15;

    bool IGNORE_NOSE = false;

    int NPERMS = 10; // number of permutations of random frames used to estimate identity
    int NMAX_FRAMES = 3500;
    int NTOT_RECONSTRS = 6; // 6; // 10; //10;
    int OUTPUT_IDENTITY = 0;
    int OUTPUT_VISUALS = 0;
    int OUTPUT_FACIAL_PARTS = 0;
    int OUTPUT_LANDMARKS_EXP_VARIATION = 0;
    int OUTPUT_EXPRESSIONS = 1;
    int OUTPUT_EXPRESSIONS_ALL = 0;
    int OUTPUT_POSES = 1;

    int NMULTICOMBS = 4; //5; //15; // Landmark
    int NSMOOTH_FRAMES = 1;
    int NRES_COEFS = 4; // 8; // spatial random resizing ;
    int SINGLE_LANDMARKS = 0;
    int PAD_RENDERING = 1;

    int NFRAMES = 7;
    int SIGMAS_FILE = 0;

    int IGNORE_SOME_LANDMARKS = 0;

    int PAD_SINGLE_IMAGE = 0;

    int USE_LOCAL_MODELS = 1;
    int USE_CONSTANT_BOUNDS = 0;

    float SKIP_FIRST_N_SECS = 0.0f;
    float KERNEL_SIGMA = 0.5f;
    int SAVE_RECONSTRUCTIONS = 0;

    int MAX_VID_FRAMES_TO_PROCESS = 50000;

    // Which expression component to use.
    // if -1, then all components are used,
    // otherwise the stated component is used
    int USE_EXPR_COMPONENT = -1;

    bool L2_TRAIN_MODE = false;
    bool USE_TEMP_SMOOTHING = false;
    bool USE_EXP_REGULARIZATION = false;

    bool EXPR_UNIFORM_REG = false;

    float EXPR_L2_WEIGHT = 0.5f;
    float DEXPR_L2_WEIGHT = 0.00025f;
    float DPOSE_L2_WEIGHT = 0.0001f;

    float EVERY_N_SECS = 1.5f;
    int TIME_T = 1;

    int NFRAMES_PER_ANGLE_BIN = 14;


    int PRINT_EVERY_N_FRAMES = 100;
    int PRINT_WARNINGS = 0;
    int PRINT_DEBUG = 0;
    int OUTDIR_WITH_PARAMS = 0;
    int PREPEND_BLANK_FRAMES = 1;

    int FILENAME_WITH_TIMES = 0;
    int PAINT_INNERMOUTH_BLACK = 1;

    int FINETUNE_EXPRESSIONS = 0;
    float FINETUNE_COEF = 1.5f;

    /**
     * @brief Expression basis parameters
     */
    std::string EXP_BASIS = "GLOBAL79";

    double OPTS_DELTA_EPS = 1.1;

    size_t K_EPSILON = 79;
    size_t K_EPSILON_L = 79;

    std::string EX_PATH = "models/dat_files/E/EX.dat";
    std::string EY_PATH = "models/dat_files/E/EY.dat";
    std::string EZ_PATH = "models/dat_files/E/EZ.dat";

    std::string EL_PATH = "models/dat_files/E/EL.dat";
    std::string EL_FULL_PATH = "models/dat_files/E/EL_full.dat"; // expression basis (landmarks)
    std::string EXP_LOWER_BOUND_PATH = "models/dat_files/E/sigma_epsilons_lower.dat";
    std::string EXP_UPPER_BOUND_PATH = "models/dat_files/E/sigma_epsilons_upper.dat";

    /*
    void set_resize_coef(double _resize_coef)
    {
        RESIZE_COEF = _resize_coef;
    }
    */

    void set_Ntot_recs(int _ntot_recs)
    {
        NTOT_RECONSTRS = _ntot_recs;
    }

    void set_ref_face_size(double _ref_face_size)
    {
        REF_FACE_SIZE = _ref_face_size;
    }

    void set_ignore_nose(bool _ignore_nose)
    {
        IGNORE_NOSE = _ignore_nose;
    }

    void set_Nframes(int _Nframes)
    {
        NFRAMES = _Nframes;
    }

    void set_sigmas_file(int _sigmas_file)
    {
        SIGMAS_FILE = _sigmas_file;
    }

    void set_confidence_range(double _confidence_range)
    {
        CONFIDENCE_RANGE = _confidence_range;
    }

    void set_skip_first_nsecs(float _set_skip_first_nsecs)
    {
        SKIP_FIRST_N_SECS = _set_skip_first_nsecs;
    }

    void set_max_vid_frames_to_process(int _max_vid_frames_to_process)
    {
        MAX_VID_FRAMES_TO_PROCESS = _max_vid_frames_to_process;
    }

    void set_3DMM_coeffs(double opts_delta, double opts_delta_beta, double opts_delta_eps)
    {
        OPTS_DELTA = opts_delta;
        OPTS_DELTA_BETA = opts_delta_beta;
        OPTS_DELTA_EPS = opts_delta_eps;
    }

    void set_exp_basis(const std::string& basis_name)
    {
        if (basis_name == "GLOBAL29")
        {
//            OPTS_DELTA_EPS = 1.0;
            EXP_BASIS = basis_name;
            K_EPSILON = 29;
            K_EPSILON_L = 29;
            EX_PATH = "models/dat_files/E/EX.dat";
            EY_PATH = "models/dat_files/E/EY.dat";
            EZ_PATH = "models/dat_files/E/EZ.dat";

            EL_PATH = "models/dat_files/E/EL.dat";
            EL_FULL_PATH = "models/dat_files/E/EL_full.dat"; // expression basis (landmarks)
            EXP_LOWER_BOUND_PATH = "models/dat_files/E/sigma_epsilons_lower.dat";
            EXP_UPPER_BOUND_PATH = "models/dat_files/E/sigma_epsilons_upper.dat";
        }
        else if (basis_name == "GLOBAL79")
        {
//            OPTS_DELTA_EPS = 1.0;
            EXP_BASIS = basis_name;
            K_EPSILON = 79;
            K_EPSILON_L = 79;
            EX_PATH = "models/dat_files/E/EX_79.dat";
            EY_PATH = "models/dat_files/E/EY_79.dat";
            EZ_PATH = "models/dat_files/E/EZ_79.dat";

            EL_PATH = "models/dat_files/E/EL_79.dat";
            EL_FULL_PATH = "models/dat_files/E/EL_79.dat"; // expression basis (landmarks)
            EXP_LOWER_BOUND_PATH = "models/dat_files/E/sigma_epsilons_79_lowerv2.dat";
            EXP_UPPER_BOUND_PATH = "models/dat_files/E/sigma_epsilons_79_upperv2.dat";
        }
        else if (basis_name == "LOCAL60")
        {
//            OPTS_DELTA_EPS = 1.25*(1.0/OPTS_DELTA);
            OPTS_DELTA_EPS /= OPTS_DELTA;
            EXP_BASIS = basis_name;
            K_EPSILON = 60;
            K_EPSILON_L = 60;
            EX_PATH = "models/dat_files/E/LEX_L60.dat";
            EY_PATH = "models/dat_files/E/LEY_L60.dat";
            EZ_PATH = "models/dat_files/E/LEZ_L60.dat";

            EL_PATH = "models/dat_files/E/LEL_L60.dat";
            EL_FULL_PATH = "models/dat_files/E/LEL_full_L60.dat"; // expression basis (landmarks)
            EXP_LOWER_BOUND_PATH = "models/dat_files/E/lower_bounds_L60.dat";
            EXP_UPPER_BOUND_PATH = "models/dat_files/E/upper_bounds_L60.dat";
        }
    }

    void set_params_from_YAML_file(const std::string& filepath)
    {
        cv::FileStorage file(filepath, cv::FileStorage::READ);
        cv::FileNode fn = file.root();

        std::set<std::string> keys;

        for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit)
        {
            cv::FileNode item = *fit;
            std::string keyname= item.name();
            keys.insert(keyname);
        }


        if (keys.find("EXP_LOWER_BOUND_PATH") != keys.end())
            file["EXP_LOWER_BOUND_PATH"] >> EXP_LOWER_BOUND_PATH;

        if (keys.find("EXP_BASIS") != keys.end())
            file["EXP_BASIS"] >> EXP_BASIS;

        if (keys.find("OPTS_DELTA_EPS") != keys.end())
            file["OPTS_DELTA_EPS"] >> OPTS_DELTA_EPS;

        set_exp_basis(EXP_BASIS);


        if (keys.find("OPTS_DELTA") != keys.end())
            file["OPTS_DELTA"] >> OPTS_DELTA;

        if (keys.find("OPTS_DELTA_BETA") != keys.end())
            file["OPTS_DELTA_BETA"] >> OPTS_DELTA_BETA;

        if (keys.find("REF_FACE_SIZE") != keys.end())
            file["REF_FACE_SIZE"] >> REF_FACE_SIZE;

        if (keys.find("CONFIDENCE_RANGE") != keys.end())
            file["CONFIDENCE_RANGE"] >> CONFIDENCE_RANGE;

        if (keys.find("IGNORE_NOSE") != keys.end())
            file["IGNORE_NOSE"] >> IGNORE_NOSE;

        if (keys.find("SIGMAS_FILE") != keys.end())
            file["SIGMAS_FILE"] >> SIGMAS_FILE;

        if (keys.find("PAD_SINGLE_IMAGE") != keys.end())
            file["PAD_SINGLE_IMAGE"] >> PAD_SINGLE_IMAGE;

        if (keys.find("IGNORE_SOME_LANDMARKS") != keys.end())
            file["IGNORE_SOME_LANDMARKS"] >> IGNORE_SOME_LANDMARKS;

        if (keys.find("NTOT_RECONSTRS") != keys.end())
            file["NTOT_RECONSTRS"] >> NTOT_RECONSTRS;

        if (keys.find("SINGLE_LANDMARKS") != keys.end())
            file["SINGLE_LANDMARKS"] >> SINGLE_LANDMARKS;

        if (keys.find("NPERMS") != keys.end())
            file["NPERMS"] >> NPERMS;

        if (keys.find("NMAX_FRAMES") != keys.end())
            file["NMAX_FRAMES"] >> NMAX_FRAMES;

        if (keys.find("NFRAMES") != keys.end())
            file["NFRAMES"] >> NFRAMES;

        if (keys.find("NMULTICOMBS") != keys.end())
            file["NMULTICOMBS"] >> NMULTICOMBS;

        if (keys.find("NRES_COEFS") != keys.end())
            file["NRES_COEFS"] >> NRES_COEFS;

        if (keys.find("NSMOOTH_FRAMES") != keys.end())
            file["NSMOOTH_FRAMES"] >> NSMOOTH_FRAMES;

        if (keys.find("USE_LOCAL_MODELS") != keys.end())
            file["USE_LOCAL_MODELS"] >> USE_LOCAL_MODELS;

        if (keys.find("USE_CONSTANT_BOUNDS") != keys.end())
            file["USE_CONSTANT_BOUNDS"] >> USE_CONSTANT_BOUNDS;

        if (keys.find("PAD_RENDERING") != keys.end())
            file["PAD_RENDERING"] >> PAD_RENDERING;

        if (keys.find("SKIP_FIRST_N_SECS") != keys.end())
            file["SKIP_FIRST_N_SECS"] >> SKIP_FIRST_N_SECS;

        if (keys.find("EVERY_N_SECS") != keys.end())
            file["EVERY_N_SECS"] >> EVERY_N_SECS;

        if (keys.find("SAVE_RECONSTRUCTIONS") != keys.end())
            file["SAVE_RECONSTRUCTIONS"] >> SAVE_RECONSTRUCTIONS;

        if (keys.find("OUTPUT_IDENTITY") != keys.end())
            file["OUTPUT_IDENTITY"] >> OUTPUT_IDENTITY;

        if (keys.find("OUTPUT_VISUALS") != keys.end())
            file["OUTPUT_VISUALS"] >> OUTPUT_VISUALS;

        if (keys.find("OUTPUT_FACIAL_PARTS") != keys.end())
            file["OUTPUT_FACIAL_PARTS"] >> OUTPUT_FACIAL_PARTS;

        if (keys.find("OUTPUT_LANDMARKS_EXP_VARIATION") != keys.end())
            file["OUTPUT_LANDMARKS_EXP_VARIATION"] >> OUTPUT_LANDMARKS_EXP_VARIATION;

        if (keys.find("OUTPUT_EXPRESSIONS") != keys.end())
            file["OUTPUT_EXPRESSIONS"] >> OUTPUT_EXPRESSIONS;

        if (keys.find("OUTPUT_EXPRESSIONS_ALL") != keys.end())
            file["OUTPUT_EXPRESSIONS_ALL"] >> OUTPUT_EXPRESSIONS_ALL;

        if (keys.find("OUTPUT_POSES") != keys.end())
            file["OUTPUT_POSES"] >> OUTPUT_POSES;

        if (keys.find("OUTDIR_WITH_PARAMS") != keys.end())
            file["OUTDIR_WITH_PARAMS"] >> OUTDIR_WITH_PARAMS;

        if (keys.find("PRINT_EVERY_N_FRAMES") != keys.end())
            file["PRINT_EVERY_N_FRAMES"] >> PRINT_EVERY_N_FRAMES;

        if (keys.find("PRINT_WARNINGS") != keys.end())
            file["PRINT_WARNINGS"] >> PRINT_WARNINGS;

        if (keys.find("PRINT_DEBUG") != keys.end())
            file["PRINT_DEBUG"] >> PRINT_DEBUG;

        if (keys.find("MAX_VID_FRAMES_TO_PROCESS") != keys.end())
            file["MAX_VID_FRAMES_TO_PROCESS"] >> MAX_VID_FRAMES_TO_PROCESS;

        if (keys.find("EXPR_L2_WEIGHT") != keys.end())
            file["EXPR_L2_WEIGHT"] >> EXPR_L2_WEIGHT;

        if (keys.find("DEXPR_L2_WEIGHT") != keys.end())
            file["DEXPR_L2_WEIGHT"] >> DEXPR_L2_WEIGHT;

        if (keys.find("DPOSE_L2_WEIGHT") != keys.end())
            file["DPOSE_L2_WEIGHT"] >> DPOSE_L2_WEIGHT;

        if (keys.find("USE_TEMP_SMOOTHING") != keys.end())
            file["USE_TEMP_SMOOTHING"] >> USE_TEMP_SMOOTHING;

        if (keys.find("USE_EXP_REGULARIZATION") != keys.end())
            file["USE_EXP_REGULARIZATION"] >> USE_EXP_REGULARIZATION;

        if (keys.find("PREPEND_BLANK_FRAMES") != keys.end())
            file["PREPEND_BLANK_FRAMES"] >> PREPEND_BLANK_FRAMES;

        if (keys.find("TIME_T") != keys.end())
            file["TIME_T"] >> TIME_T;

        if (keys.find("EXPR_UNIFORM_REG") != keys.end())
            file["EXPR_UNIFORM_REG"] >> EXPR_UNIFORM_REG;

        if (keys.find("FILENAME_WITH_TIMES") != keys.end())
            file["FILENAME_WITH_TIMES"] >> FILENAME_WITH_TIMES;

        if (keys.find("PAINT_INNERMOUTH_BLACK") != keys.end())
            file["PAINT_INNERMOUTH_BLACK"] >> PAINT_INNERMOUTH_BLACK;

        if (keys.find("FINETUNE_EXPRESSIONS") != keys.end())
            file["FINETUNE_EXPRESSIONS"] >> FINETUNE_EXPRESSIONS;

        if (keys.find("FINETUNE_COEF") != keys.end())
            file["FINETUNE_COEF"] >> FINETUNE_COEF;

        if (keys.find("USE_EXPR_COMPONENT") != keys.end())
            file["USE_EXPR_COMPONENT"] >> USE_EXPR_COMPONENT;

        int tmp;
        file["NFRAMES_PER_ANGLE_BIN"] >> tmp;
        if (tmp > 0)
            NFRAMES_PER_ANGLE_BIN = tmp;

        float kernel_sigma;
        file["KERNEL_SIGMA"] >> kernel_sigma;
        if (kernel_sigma > 0)
            KERNEL_SIGMA = kernel_sigma;
    }

    std::string get_key()
    {
        std::stringstream ss;

        ss << "OD" << OPTS_DELTA << "_OE" << OPTS_DELTA_EPS  << "_OB" << OPTS_DELTA_BETA << "_RS" << REF_FACE_SIZE
                 << "_IN" << (int) IGNORE_NOSE << "_SF" << SIGMAS_FILE << "_CF" << std::setw(3) << std::setfill('0') << CONFIDENCE_RANGE << "NF"<< NFRAMES
                 << "_NTC" << NTOT_RECONSTRS << "_UL" << USE_LOCAL_MODELS << "_CB" << USE_CONSTANT_BOUNDS << EXP_BASIS
                 << "_IS" << IGNORE_SOME_LANDMARKS << "K" << KERNEL_RADIUS;

        if (USE_TEMP_SMOOTHING) {
            ss << TIME_T;
            ss << DEXPR_L2_WEIGHT;
            ss << DPOSE_L2_WEIGHT;
        }

        if (USE_EXP_REGULARIZATION) {
            ss << EXPR_L2_WEIGHT;
            ss << EXPR_UNIFORM_REG;
            ss << "_";
        }

        if (FINETUNE_EXPRESSIONS)
            ss << "F" << FINETUNE_COEF << "_";

        ss << NRES_COEFS << NMULTICOMBS << KERNEL_SIGMA << "P" << PAINT_INNERMOUTH_BLACK;

        return ss.str();
    }
}
