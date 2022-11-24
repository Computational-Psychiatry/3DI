#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <opencv2/core.hpp>

namespace config
{
    extern double OPTS_DELTA, OPTS_DELTA_BETA, OPTS_DELTA_EPS;
    extern double CONFIDENCE_RANGE, RESIZE_COEF;
    extern double REF_FACE_SIZE;
    extern bool IGNORE_NOSE;

    extern double REF_FACE_SIZE;

    extern int NFRAMES;
    extern int SIGMAS_FILE;
    extern int SINGLE_LANDMARKS;

    extern int PAD_RENDERING;

    extern int NTOT_RECONSTRS;

    extern int NMULTICOMBS;
    extern int NRES_COEFS;

    extern int NPERMS; // number of permutations of random frames used to estimate identity
    extern int NMAX_FRAMES;

    extern int NSMOOTH_FRAMES;

    extern int IGNORE_SOME_LANDMARKS;
    extern int PAD_SINGLE_IMAGE;

    extern float SKIP_FIRST_N_SECS;
    extern float KERNEL_SIGMA;
    extern int SAVE_RECONSTRUCTIONS;

    extern int USE_LOCAL_MODELS;
    extern int USE_CONSTANT_BOUNDS;

    extern int OUTPUT_IDENTITY;
    extern int OUTPUT_VISUALS;
    extern int OUTPUT_FACIAL_PARTS;
    extern int OUTPUT_LANDMARKS_EXP_VARIATION;

    extern int OUTPUT_EXPRESSIONS;
    extern int OUTPUT_EXPRESSIONS_ALL;
    extern int OUTPUT_POSES;

    extern int MAX_VID_FRAMES_TO_PROCESS;

    extern bool L2_TRAIN_MODE;
    extern bool USE_TEMP_SMOOTHING;
    extern bool USE_EXP_REGULARIZATION;
    extern bool EXPR_UNIFORM_REG;

    // Which expression component to use.
    // if -1, then all components are used,
    // otherwise the stated component is used
    extern int USE_EXPR_COMPONENT;

    extern float EXPR_L2_WEIGHT;
    extern float DEXPR_L2_WEIGHT;
    extern float DPOSE_L2_WEIGHT;

    extern float EVERY_N_SECS;
    extern int TIME_T;

    extern int NFRAMES_PER_ANGLE_BIN;

    extern int PRINT_EVERY_N_FRAMES;
    extern int PRINT_WARNINGS;
    extern int PRINT_DEBUG;

    extern int OUTDIR_WITH_PARAMS;
    extern int PREPEND_BLANK_FRAMES;

    extern int FILENAME_WITH_TIMES;

    extern int PAINT_INNERMOUTH_BLACK;

    extern int FINETUNE_EXPRESSIONS;
    extern float FINETUNE_COEF;

    /**
     * @brief Expression basis parameters
     */
    extern std::string EXP_BASIS; // name of expression basis
    extern std::string EX_PATH, EY_PATH, EZ_PATH; // expression basis (dense 3DMM)
    extern std::string EL_PATH, EL_FULL_PATH; // expression basis (landmarks)
    extern std::string EXP_LOWER_BOUND_PATH, EXP_UPPER_BOUND_PATH; // paths to the files that contain lower and upper bounds
    extern size_t K_EPSILON, K_EPSILON_L;


    void set_resize_coef(double _resize_coef);
    void set_ref_face_size(double _ref_face_size);
    void set_ignore_nose(bool _ignore_nose);
    void set_sigmas_file(int _sigmas_file);
    void set_Nframes(int _Nframes);
    void set_Ntot_recs(int _Nframes);
    void set_confidence_range(double _confidence_range);
    void set_3DMM_coeffs(double opts_delta, double opts_delta_beta, double opts_delta_eps);
    void set_exp_basis(const std::string& basis_name);

    void set_skip_first_nsecs(float _set_skip_first_nsecs);
    void set_max_vid_frames_to_process(int _max_vid_frames_to_process);

    void set_params_from_YAML_file(const std::string& filepath);

    std::string get_key();
}

#endif // CONFIG_H
