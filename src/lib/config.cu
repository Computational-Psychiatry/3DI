// In config.cpp

#include <sstream>
#include <iomanip>
#include <iostream>
#include "config.h"

namespace config
{
    double OPTS_DELTA = 0.5;
    double OPTS_DELTA_BETA = 1.5;

    double REF_FACE_SIZE = 92; //120; // 84;
//    double RESIZE_COEF = 0.50;

    double CONFIDENCE_RANGE = 0.42;
//    double CONFIDENCE_RANGE = 0.15;

    bool IGNORE_NOSE = false;

    int NPERMS = 10; // number of permutations of random frames used to estimate identity
    int NMAX_FRAMES = 700;
    int NTOT_RECONSTRS = 12; // 6; // 10; //10;
    int OUTPUT_IDENTITY = 1;
    int OUTPUT_VISUALS = 1;

    int NMULTICOMBS = 12; //5; //15; // Landmark
    int NSMOOTH_FRAMES = 4;
    int NRES_COEFS = 12; // 8; // spatial random resizing ;
    int SINGLE_LANDMARKS = 0;
    int PAD_RENDERING = 1;

    int NFRAMES = 9;
    int SIGMAS_FILE = 0;

    int IGNORE_SOME_LANDMARKS = 1;

    int PAD_SINGLE_IMAGE = 0;

    int USE_LOCAL_MODELS = 1;
    int USE_CONSTANT_BOUNDS = 0;

    float SKIP_FIRST_N_SECS = 15.0;
    float KERNEL_SIGMA = 1.0f;
    int SAVE_RECONSTRUCTIONS = 0;

    int MAX_VID_FRAMES_TO_PROCESS = 3000;

    // Which expression component to use.
    // if -1, then all components are used,
    // otherwise the stated component is used
    int USE_EXPR_COMPONENT = -1;

    bool L2_TRAIN_MODE = false;
    bool USE_TEMP_SMOOTHING = true;
    bool USE_EXP_REGULARIZATION = true;

    bool EXPR_UNIFORM_REG = false;

    float EXPR_L2_WEIGHT = 0.75f;
    float DEXPR_L2_WEIGHT = 7.5f;
    float DPOSE_L2_WEIGHT = 5.0f;

    float EVERY_N_SECS = 1.0f;
    int TIME_T = 4;

    int NFRAMES_PER_ANGLE_BIN = 14;


    int PRINT_EVERY_N_FRAMES = 30;
    int PRINT_WARNINGS = 0;
    int PRINT_DEBUG = 0;
    int OUTDIR_WITH_PARAMS = 0;



    /**
     * @brief Expression basis parameters
     */
    std::string EXP_BASIS = "GLOBAL29";

    double OPTS_DELTA_EPS = 0.4;

    size_t K_EPSILON = 29;
    size_t K_EPSILON_L = 29;

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
        file["EXP_LOWER_BOUND_PATH"] >> EXP_LOWER_BOUND_PATH;
        file["EXP_BASIS"] >> EXP_BASIS;

        file["OPTS_DELTA_EPS"] >> OPTS_DELTA_EPS;
        set_exp_basis(EXP_BASIS);


        file["OPTS_DELTA"] >> OPTS_DELTA;
        file["OPTS_DELTA_BETA"] >> OPTS_DELTA_BETA;
        file["REF_FACE_SIZE"] >> REF_FACE_SIZE;
//        file["RESIZE_COEF"] >> RESIZE_COEF;
        file["CONFIDENCE_RANGE"] >> CONFIDENCE_RANGE;
        file["IGNORE_NOSE"] >> IGNORE_NOSE;
        file["SIGMAS_FILE"] >> SIGMAS_FILE;
        file["PAD_SINGLE_IMAGE"] >> PAD_SINGLE_IMAGE;
        file["IGNORE_SOME_LANDMARKS"] >> IGNORE_SOME_LANDMARKS;
        file["NTOT_RECONSTRS"] >> NTOT_RECONSTRS;

        file["SINGLE_LANDMARKS"] >> SINGLE_LANDMARKS;
        file["NPERMS"] >> NPERMS;
        file["NMAX_FRAMES"] >> NMAX_FRAMES;

        file["NFRAMES"] >> NFRAMES;
        file["NMULTICOMBS"] >> NMULTICOMBS;
        file["NRES_COEFS"] >> NRES_COEFS;
        file["NSMOOTH_FRAMES"] >> NSMOOTH_FRAMES;
        file["USE_LOCAL_MODELS"] >> USE_LOCAL_MODELS;
        file["USE_CONSTANT_BOUNDS"] >> USE_CONSTANT_BOUNDS;
        file["PAD_RENDERING"] >> PAD_RENDERING;
        file["SKIP_FIRST_N_SECS"] >> SKIP_FIRST_N_SECS;
        file["EVERY_N_SECS"] >> EVERY_N_SECS;
        file["SAVE_RECONSTRUCTIONS"] >> SAVE_RECONSTRUCTIONS;
        file["OUTPUT_IDENTITY"] >> OUTPUT_IDENTITY;
        file["OUTPUT_VISUALS"] >> OUTPUT_VISUALS;
        file["OUTDIR_WITH_PARAMS"] >> OUTDIR_WITH_PARAMS;
        file["PRINT_EVERY_N_FRAMES"] >> PRINT_EVERY_N_FRAMES;
        file["PRINT_WARNINGS"] >> PRINT_WARNINGS;
        file["PRINT_DEBUG"] >> PRINT_DEBUG;
        file["MAX_VID_FRAMES_TO_PROCESS"] >> MAX_VID_FRAMES_TO_PROCESS;
        file["EXPR_L2_WEIGHT"] >> EXPR_L2_WEIGHT;
        file["DEXPR_L2_WEIGHT"] >> DEXPR_L2_WEIGHT;
        file["DPOSE_L2_WEIGHT"] >> DPOSE_L2_WEIGHT;
        file["USE_TEMP_SMOOTHING"] >> USE_TEMP_SMOOTHING;
        file["USE_EXP_REGULARIZATION"] >> USE_EXP_REGULARIZATION;
        file["TIME_T"] >> TIME_T;
        file["EXPR_UNIFORM_REG"] >> EXPR_UNIFORM_REG;

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

        ss << NRES_COEFS << NMULTICOMBS << KERNEL_SIGMA;

        return ss.str();
    }
}
