// In config.cpp

#include <sstream>
#include <iomanip>
#include <iostream>
#include <set>
#include "config.h"
#include "constants.h"

#include <experimental/filesystem>


namespace config
{
    double OPTS_DELTA = 0.66;
    double OPTS_DELTA_BETA = 2.0;

    double REF_FACE_SIZE = 86; //120; // 84;
//    double RESIZE_COEF = 0.50;

    double CONFIDENCE_RANGE = 0.42;
//    double CONFIDENCE_RANGE = 0.15;

    bool IGNORE_NOSE = false;
    bool TWOSTAGE_ILLUM = false;

    int NPERMS = 10; // number of permutations of random frames used to estimate identity
    int NMAX_FRAMES = 432000; // this corresponds to 2 hours for a 60fps video
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
    int NTEX_COEFS;
    int NID_COEFS;
    int K_ALPHA_L;
    int K_BETA_L;

    int PRINT_EVERY_N_FRAMES = 100;
    int PRINT_WARNINGS = 0;
    int PRINT_DEBUG = 0;
    int OUTDIR_WITH_PARAMS = 0;
    int PREPEND_BLANK_FRAMES = 1;

    int FILENAME_WITH_TIMES = 0;
    int PAINT_INNERMOUTH_BLACK = 1;

    bool FINETUNE_ONLY = false;
    int FINETUNE_EXPRESSIONS = 0;
    int NPTS;
    int N_TRIANGLES;
    uint Nredundant;
    float FINETUNE_COEF = 1.5f;

    /**
     * @brief Expression basis parameters
     */
    std::string EXP_BASIS = "GLOBAL79";
    std::string MM = "BFM-23660";


    double OPTS_DELTA_EPS = 1.1;

    size_t K_EPSILON = 79;
    size_t K_EPSILON_L = 79;
    std::string EX_PATH, EY_PATH, EZ_PATH;
    std::string IX_PATH, IY_PATH, IZ_PATH; // identity basis (dense 3DMM)
    std::string X0_PATH, Y0_PATH, Z0_PATH; // mean face
    std::string TEX_PATH, TEXMU_PATH; // texture basis (dense 3DMM)

    std::string TL_PATH; // triangulation path

    std::string SIGMA_ALPHAS_PATH, SIGMA_BETAS_PATH;
    std::string AL60_PATH, AL_FULL_PATH, EL_PATH, EL_FULL_PATH; // expression basis (landmarks)

    std::string EXP_LOWER_BOUND_PATH;
    std::string EXP_UPPER_BOUND_PATH;
    std::string P0L_PATH;
    std::vector<uint> LIS;



    std::string FACE_DETECTOR_DPATH = "models/deploy.prototxt";
    std::string FACE_DETECTOR_MPATH = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
    std::string LANDMARK_MPATH = "models/landmark_models/model_FAN_frozen.pb";
    std::string LANDMARK_MODELS_DIR = "models/landmark_models";
    std::string LANDMARK_LEYE_MPATH = "models/landmark_models/m-64l64g0-64-128-5121968464leye.pb";
    std::string LANDMARK_REYE_MPATH = "models/landmark_models/m-64l64g0-64-128-5121968464reye.pb";
    std::string LANDMARK_MOUTH_MPATH = "models/landmark_models/m-64l64g0-64-128-5121968464mouth.pb";
    std::string LANDMARK_CORRECTION_MPATH = "models/landmark_models/model_correction.pb";

    int lmk_lec = 19;
    int lmk_rec = 28;

    /*
    */

    /*
    void set_resize_coef(double _resize_coef)
    {
        RESIZE_COEF = _resize_coef;
    }
    */

    void set_MM(const std::string& mm)
    {
        /**
          TODO
          this neads to be read separately for each MM
          */
        NID_COEFS = 199;
        NTEX_COEFS = 199;
        K_ALPHA_L = 60;
        K_BETA_L = 0;


        MM = mm;
/*
        EX_PATH = "models/MMs/" + MM + "/E/EX.dat";
        EY_PATH = "models/MMs/" + MM + "/E/EY.dat";
        EZ_PATH = "models/MMs/" + MM + "/E/EZ.dat";

        EL_PATH = "models/MMs/" + MM + "/E/EL.dat";
        EL_FULL_PATH = "models/MMs/" + MM + "/E/EL_full.dat"; // expression basis (landmarks)
        EXP_LOWER_BOUND_PATH = "models/MMs/" + MM + "/E/sigma_epsilons_lower.dat";
        EXP_UPPER_BOUND_PATH = "models/MMs/" + MM + "/E/sigma_epsilons_upper.dat";
*/
        IX_PATH = "models/MMs/" + MM + "/IX.dat";
        IY_PATH = "models/MMs/" + MM + "/IY.dat";
        IZ_PATH = "models/MMs/" + MM + "/IZ.dat";

        X0_PATH = "models/MMs/" + MM + "/X0_mean.dat";
        Y0_PATH = "models/MMs/" + MM + "/Y0_mean.dat";
        Z0_PATH = "models/MMs/" + MM + "/Z0_mean.dat";

        TEX_PATH = "models/MMs/" + MM + "/TEX.dat";
        TEXMU_PATH = "models/MMs/" + MM + "/tex_mu.dat";

        TL_PATH = "models/MMs/" + MM + "/tl.dat";
        AL_FULL_PATH = "models/MMs/" + MM + "/AL_full.dat";
        AL60_PATH = "models/MMs/" + MM + "/AL_60.dat";

        SIGMA_ALPHAS_PATH = "models/MMs/" + MM + "/sigma_alphas.dat";
        SIGMA_BETAS_PATH = "models/MMs/" + MM + "/sigma_betas.dat";
        P0L_PATH = "models/MMs/" + MM + "/p0L_mat.dat";

        if (MM == "BFM-23660" || MM == "BFMmm-23660") {
            N_TRIANGLES = 46703;
            LIS = std::vector<uint>({19106,19413,19656,19814,19981,20671,20837,20995,21256,
                                     21516,8161, 8175, 8184, 8190, 6758, 7602, 8201, 8802,
                                     9641, 1831, 3759, 5049, 6086, 4545, 3515, 10455,11482,
                                     12643,14583,12915,11881,5522, 6154, 7375, 8215, 9295,
                                     10523,10923, 9917, 9075, 8235, 7395, 6548, 5908,7264,
                                     8224,9184,10665,8948,8228,7508});
        }
        else if (MM == "BFM-14643" || MM == "BFMmm-14643")
        {
            N_TRIANGLES = 28694;
            LIS = std::vector<uint>({12900, 13203, 13402, 13508, 13604, 13892, 13984, 14088, 14298, 14556, 6020, 6034, 6043, 6049, 4620, 5461, 6060, 6661, 7500, 660,
                                     1735, 2932, 3953, 2450, 1532, 8310, 9315, 10383, 11470, 10612, 9696, 3396, 4021, 5234, 6074, 7154, 8378, 8773, 7776, 6934, 6094, 5254, 4412, 3777, 5123, 6083, 7043, 8519, 6807, 6087, 5367});
        }
        else if (MM == "BFM-17572" || MM == "BFMmm-17572")
        {
            N_TRIANGLES = 34537;
            LIS = std::vector<uint>({15132, 15439, 15681, 15826, 15958, 16406, 16536, 16679, 16939, 17199, 6951, 6965, 6974, 6980, 5548, 6392, 6991, 7592, 8431, 897, 2552, 3839, 4876,
                                     3335, 2315, 9245, 10272, 11429, 13092, 11694, 10671, 4312, 4944, 6165, 7005, 8085, 9313, 9713, 8707, 7865, 7025, 6185, 5338, 4698, 6054, 7014, 7974, 9455, 7738, 7018, 6298});
        }
        else if (MM == "BFM-17572" || MM == "BFMmm-17572")
        {
            N_TRIANGLES = 34537;
            LIS = std::vector<uint>({15132, 15439, 15681, 15826, 15958, 16406, 16536, 16679, 16939, 17199, 6951, 6965, 6974, 6980, 5548, 6392, 6991, 7592, 8431, 897, 2552, 3839, 4876,
                                     3335, 2315, 9245, 10272, 11429, 13092, 11694, 10671, 4312, 4944, 6165, 7005, 8085, 9313, 9713, 8707, 7865, 7025, 6185, 5338, 4698, 6054, 7014, 7974, 9455, 7738, 7018, 6298});
        }
        else if (MM == "BFMmm-18934")
        {
            N_TRIANGLES = 37254;
            LIS = std::vector<uint>({16126,16433,16676,16834,16984,17497,17645,17802,18063,18323,7370,
                                     7384,7393,7399,5967,6811,7410,8011,8850,1123,2968,4258,5295,3754,
                                     2724,9664,10691,11852,13700,12124,11090,4731,5363,6584,7424,8504,
                                     9732,10132,9126,8284,7444,6604,5757,5117,6473,7433,8393,9874,8157,
                                     7437,6717});
        }
        else if (MM == "BFMmm-19830")
        {
            N_TRIANGLES = 39053;
            LIS = std::vector<uint>({17286,17577,17765,17885,18012,18542,18668,18788,18987,19236,7882,7896,7905,7911,6479,
                                   7323,7922,8523,9362,1586,3480,4770,5807,4266,3236, 10176,11203,12364,14269,12636,11602,
                                   5243,5875,7096,7936,9016,10244,10644,9638,8796,7956,7116,6269,5629,6985,7945,8905,10386,8669,7949,7229});
        }

        Nredundant = N_TRIANGLES*NTMP;

        std::vector<std::string> result;
        std::stringstream ss(MM);
        std::string item;

        while (getline (ss, item, '-')) {
            result.push_back (item);
        }

        NPTS = std::stoi(result[1]);
        set_exp_basis(EXP_BASIS);
    }


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
            EX_PATH = "models/MMs/" + MM +"/E/EX.dat";
            EY_PATH = "models/MMs/" + MM +"/E/EY.dat";
            EZ_PATH = "models/MMs/" + MM +"/E/EZ.dat";

            EL_PATH = "models/MMs/" + MM +"/E/EL.dat";
            EL_FULL_PATH = "models/MMs/" + MM +"/E/EL_full.dat"; // expression basis (landmarks)
            EXP_LOWER_BOUND_PATH = "models/MMs/" + MM +"/E/sigma_epsilons_lower.dat";
            EXP_UPPER_BOUND_PATH = "models/MMs/" + MM +"/E/sigma_epsilons_upper.dat";
        }
        else if (basis_name == "GLOBAL79")
        {
//            OPTS_DELTA_EPS = 1.0;
            EXP_BASIS = basis_name;
            K_EPSILON = 79;
            K_EPSILON_L = 79;
            EX_PATH = "models/MMs/" + MM +"/E/EX_79.dat";
            EY_PATH = "models/MMs/" + MM +"/E/EY_79.dat";
            EZ_PATH = "models/MMs/" + MM +"/E/EZ_79.dat";

            EL_PATH = "models/MMs/" + MM +"/E/EL_79.dat";
            EL_FULL_PATH = "models/MMs/" + MM +"/E/EL_79.dat"; // expression basis (landmarks)
            EXP_LOWER_BOUND_PATH = "models/MMs/" + MM +"/E/sigma_epsilons_79_lowerv2.dat";
            EXP_UPPER_BOUND_PATH = "models/MMs/" + MM +"/E/sigma_epsilons_79_upperv2.dat";
        }
        else if (basis_name == "LOCAL60")
        {
//            OPTS_DELTA_EPS = 1.25*(1.0/OPTS_DELTA);
            OPTS_DELTA_EPS /= OPTS_DELTA;
            EXP_BASIS = basis_name;
            K_EPSILON = 60;
            K_EPSILON_L = 60;
            EX_PATH = "models/MMs/" + MM +"/E/LEX_L60.dat";
            EY_PATH = "models/MMs/" + MM +"/E/LEY_L60.dat";
            EZ_PATH = "models/MMs/" + MM +"/E/LEZ_L60.dat";

            EL_PATH = "models/MMs/" + MM +"/E/LEL_L60.dat";
            EL_FULL_PATH = "models/MMs/" + MM +"/E/LEL_full_L60.dat"; // expression basis (landmarks)
            EXP_LOWER_BOUND_PATH = "models/MMs/" + MM +"/E/lower_bounds_L60.dat";
            EXP_UPPER_BOUND_PATH = "models/MMs/" + MM +"/E/upper_bounds_L60.dat";
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

        if (keys.find("TWOSTAGE_ILLUM") != keys.end())
            file["TWOSTAGE_ILLUM"] >> TWOSTAGE_ILLUM;

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

        if (keys.find("FINETUNE_ONLY") != keys.end())
            file["FINETUNE_ONLY"] >> FINETUNE_ONLY;

        if (keys.find("FINETUNE_COEF") != keys.end())
            file["FINETUNE_COEF"] >> FINETUNE_COEF;

        if (keys.find("USE_EXPR_COMPONENT") != keys.end())
            file["USE_EXPR_COMPONENT"] >> USE_EXPR_COMPONENT;

        if (keys.find("MM") != keys.end())
            file["MM"] >> MM;

        set_MM(MM);

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
        ss << MM;

        return ss.str();
    }

    /**
     * @brief Check that all files needed by 3DI (model files etc.) are where they need to be.
     * @return
     */
    bool check_all_necessary_files()
    {
        /**
         * @brief (1) First check 3DMM model files
         */
        std::vector<std::string> model_files;
        model_files.push_back(EX_PATH);
        model_files.push_back(EY_PATH);
        model_files.push_back(EZ_PATH);

        model_files.push_back(IX_PATH);
        model_files.push_back(IY_PATH);
        model_files.push_back(IZ_PATH);

        model_files.push_back(X0_PATH);
        model_files.push_back(Y0_PATH);
        model_files.push_back(Z0_PATH);

        model_files.push_back(TL_PATH);
        model_files.push_back(TEX_PATH);
        model_files.push_back(TEXMU_PATH);

        model_files.push_back(SIGMA_ALPHAS_PATH);
        model_files.push_back(SIGMA_BETAS_PATH);

        model_files.push_back(AL60_PATH);
        model_files.push_back(AL_FULL_PATH);
        model_files.push_back(EL_PATH);
        model_files.push_back(EL_FULL_PATH);

        model_files.push_back(EXP_LOWER_BOUND_PATH);
        model_files.push_back(EXP_UPPER_BOUND_PATH);
        model_files.push_back(P0L_PATH);

        bool success = true;
        for (std::string& model_file : model_files)
        {
            if (!std::experimental::filesystem::exists(model_file)) {
                std::cerr << "Model file not found at expected location: " << model_file << std::endl;
                std::cerr << "Make sure that you downloaded the BFM and preprocessed it by following the README instructions." << std::endl;
                success = false;
            }
        }


        /**
         * @brief (2) Now check face/landmark detection models
         */
        std::vector<std::string> landmark_files;

        landmark_files.push_back(FACE_DETECTOR_DPATH);
        landmark_files.push_back(FACE_DETECTOR_MPATH);
        landmark_files.push_back(LANDMARK_MPATH);
        landmark_files.push_back(LANDMARK_LEYE_MPATH);
        landmark_files.push_back(LANDMARK_REYE_MPATH);
        landmark_files.push_back(LANDMARK_MOUTH_MPATH);
        landmark_files.push_back(LANDMARK_CORRECTION_MPATH);


        for (std::string& landmark_file : landmark_files)
        {
            if (!std::experimental::filesystem::exists(landmark_file)) {
                std::cerr << "Model file not found at expected location: " << landmark_file << std::endl;
                std::cerr << "Make sure that you downloaded the models for face and landmark detection by following the README instructions." << std::endl;
                success = false;
            }
        }

        return success && check_detector_models();
    }

    bool check_detector_models()
    {
        bool success = true;

        /**
         * @brief Check face/landmark detection models
         */
        std::vector<std::string> landmark_files;

        landmark_files.push_back(FACE_DETECTOR_DPATH);
        landmark_files.push_back(FACE_DETECTOR_MPATH);
        landmark_files.push_back(LANDMARK_LEYE_MPATH);
        landmark_files.push_back(LANDMARK_REYE_MPATH);
        landmark_files.push_back(LANDMARK_MOUTH_MPATH);
        landmark_files.push_back(LANDMARK_CORRECTION_MPATH);


        for (std::string& landmark_file : landmark_files)
        {
            if (!std::experimental::filesystem::exists(landmark_file)) {
                std::cerr << "Model file not found at expected location: " << landmark_file << std::endl;
                std::cerr << "Make sure that you downloaded the models for face and landmark detection by following the README instructions." << std::endl;
                success = false;
            }
        }

        /**
         * @brief Check landmark bound related files
         */

        std::vector<std::string> landmark_bound_files;

        for (size_t i=0; i<NLANDMARKS_51; ++i)
        {
            std::string extension_partbased("dangle15_pctile_2.00");
            std::string extension_global;

            if (config::USE_CONSTANT_BOUNDS == 1) {
                extension_global = "dangle120_pctile_1.50";
            } else {
                extension_global = "dangle12_pctile_1.50";
            }

            std::stringstream ss1, ss2, ss3;
            if (config::USE_LOCAL_MODELS)
            {
                ss1 << config::LANDMARK_MODELS_DIR << "/bounds/ymins_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_partbased << ".txt";
                ss2 << config::LANDMARK_MODELS_DIR << "/bounds/ymaxs_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_partbased << ".txt";
            }
            else
            {
                ss1 << config::LANDMARK_MODELS_DIR << "/bounds/ymins_nopart_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_global << ".txt";
                ss2 << config::LANDMARK_MODELS_DIR << "/bounds/ymaxs_nopart_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_global << ".txt";
            }

            landmark_bound_files.push_back(ss1.str());
            landmark_bound_files.push_back(ss2.str());

            if (config::USE_LOCAL_MODELS) {
                ss3 << config::LANDMARK_MODELS_DIR << "/bounds/bound_estimator_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_partbased << ".pb";
            } else {
                ss3 << config::LANDMARK_MODELS_DIR << "/bounds/bound_estimator_nopart_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_global << ".pb";
            }
            landmark_bound_files.push_back(ss3.str());
        }

        landmark_bound_files.push_back("models/dat_files/bounds_fao_lx.dat_36_1.50_partbased");
        landmark_bound_files.push_back("models/dat_files/bounds_fao_ly.dat_36_1.50_partbased");
        landmark_bound_files.push_back("models/dat_files/bounds_fao_ux.dat_36_1.50_partbased");
        landmark_bound_files.push_back("models/dat_files/bounds_fao_uy.dat_36_1.50_partbased");

        for (std::string& lb_file : landmark_bound_files)
        {
            if (!std::experimental::filesystem::exists(lb_file)) {
                std::cerr << "Model file not found at expected location: " << lb_file << std::endl;
                std::cerr << "Make sure that you downloaded the models for face and landmark detection by following the README instructions." << std::endl;
                success = false;
            }
        }

        return success;
    }

}
