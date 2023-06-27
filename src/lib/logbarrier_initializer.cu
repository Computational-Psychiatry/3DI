/*
 * logbarrier_initializer.cu
 *
 *  Created on: Sep 21, 2020
 *      Author: root
 */

#include "logbarrier_initializer.h"

#include "config.h"



__global__ void set_vec_ones(float *vec, uint N)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N)
        return;

    vec[n] = 1.0f;
}





__global__ void initialize_kernel(const float *meanx, const float *meany,
                                  const float *phix, const float *phiy, const float *cx, const float *cy,
                                  const float *sigmax_orth, const float *sigmay_orth, const float *u_orth,
                                  float *taux, float *tauy, float *tauz, float *u)
{
    tauz[0] = phix[0]/((fabs(sigmax_orth[0])+fabs(sigmay_orth[0]))*0.5);
//    tauz[0] = phix[0]/sigmax_orth[0];


    taux[0] = tauz[0]*(meanx[0]-cx[0])/phix[0];
    tauy[0] = tauz[0]*(meany[0]-cy[0])/phiy[0];

    u[0] = u_orth[0];
    u[1] = u_orth[1];
    u[2] = u_orth[2];
}






Logbarrier_Initializer::Logbarrier_Initializer(std::vector<Camera> *_cams_ptr,  OptimizationVariables *ov, cusolverDnHandle_t& handleDn, float bdelta,
                                               bool use_identity_, bool use_texture_, bool use_expression_, Renderer& r, bool _use_slack, float _CONFIDENCE, bool _use_temp_smoothing, bool _use_exp_regularization)
    : T(ov->T), cams_ptr(_cams_ptr), use_slack(_use_slack),
      Ktotal_base(ov->Kalpha + ov->Kepsilon*ov->T + 6*ov->T),
      Ktotal(ov->Kalpha + ov->Kepsilon*ov->T + 6*ov->T+int(_use_slack)),
      s(handleDn, ov->Kalpha + ov->Kepsilon*ov->T + 6*ov->T + int(_use_slack)),
      s_qr(handleDn, ov->Kalpha + ov->Kepsilon*ov->T + 6*ov->T + int(_use_slack)), oi(handleDn),
      f_beta_lb(NULL), f_beta_ub(NULL), beta_lb(NULL), beta_ub(NULL),
      use_identity(use_identity_), use_texture(use_texture_), use_expression(use_expression_),
      fit_success(false), CONFIDENCE(_CONFIDENCE),
      use_temp_smoothing(_use_temp_smoothing), use_exp_regularization(_use_exp_regularization)
{
    face_sizes =  (float*)malloc( T*sizeof(float) );
    angle_idx =  (uint*)malloc( T*sizeof(uint) );

    HANDLE_ERROR( cudaMalloc( (void**)&xmeans, sizeof(float)*T ) );
    HANDLE_ERROR( cudaMalloc( (void**)&ymeans, sizeof(float)*T ) );

    HANDLE_ERROR( cudaMalloc( (void**)&nabla2F, sizeof(float)*(Ktotal*Ktotal + Ktotal*NLANDMARKS_51) ) );

    nablaF = nabla2F + Ktotal*Ktotal;


    HANDLE_ERROR( cudaMalloc( (void**)&gradient, sizeof(float)*Ktotal) );



    HANDLE_ERROR( cudaMalloc( (void**)&f_alpha_lb, sizeof(float)*(2*ov->Kalpha+2*ov->Kepsilon)) );
    f_alpha_ub = f_alpha_lb + ov->Kalpha;
    f_epsilon_lb = f_alpha_ub + ov->Kalpha;
    f_epsilon_ub = f_epsilon_lb + ov->Kepsilon;


    HANDLE_ERROR( cudaMalloc( (void**)&p, sizeof(float)*NLANDMARKS_51*3 ) );
    HANDLE_ERROR( cudaMalloc( (void**)&p0L_mat, sizeof(float)*NLANDMARKS_51*3 ) );
    HANDLE_ERROR( cudaMalloc( (void**)&Gxs_ALL, sizeof(float)*NLANDMARKS_51*12 ) );

    Gx_minus_tmp = Gxs_ALL;
    Gy_minus_tmp = Gx_minus_tmp+NLANDMARKS_51;
    Gx_plus_tmp = Gy_minus_tmp+NLANDMARKS_51;
    Gy_plus_tmp = Gx_plus_tmp+NLANDMARKS_51;

    inv_Gx_minus_tmp = Gy_plus_tmp+NLANDMARKS_51;
    inv_Gy_minus_tmp = inv_Gx_minus_tmp+NLANDMARKS_51;
    inv_Gx_plus_tmp = inv_Gy_minus_tmp+NLANDMARKS_51;
    inv_Gy_plus_tmp = inv_Gx_plus_tmp+NLANDMARKS_51;

    nlog_Gx_minus = inv_Gy_plus_tmp + NLANDMARKS_51;
    nlog_Gy_minus = nlog_Gx_minus + NLANDMARKS_51;
    nlog_Gx_plus = nlog_Gy_minus + NLANDMARKS_51;
    nlog_Gy_plus = nlog_Gx_plus + NLANDMARKS_51;

    HANDLE_ERROR( cudaMalloc( (void**)&xl, sizeof(float)*NLANDMARKS_51*T ) );
    HANDLE_ERROR( cudaMalloc( (void**)&yl, sizeof(float)*NLANDMARKS_51*T ) );

    HANDLE_ERROR( cudaMalloc( (void**)&bounds_ux, sizeof(float)*NLANDMARKS_51*N_ANGLE_COMBINATIONS ) );
    HANDLE_ERROR( cudaMalloc( (void**)&bounds_uy, sizeof(float)*NLANDMARKS_51*N_ANGLE_COMBINATIONS ) );
    HANDLE_ERROR( cudaMalloc( (void**)&bounds_lx, sizeof(float)*NLANDMARKS_51*N_ANGLE_COMBINATIONS ) );
    HANDLE_ERROR( cudaMalloc( (void**)&bounds_ly, sizeof(float)*NLANDMARKS_51*N_ANGLE_COMBINATIONS ) );


    HANDLE_ERROR( cudaMalloc( (void**)&bounds_ux_cur, sizeof(float)*NLANDMARKS_51*T ) );
    HANDLE_ERROR( cudaMalloc( (void**)&bounds_uy_cur, sizeof(float)*NLANDMARKS_51*T ) );
    HANDLE_ERROR( cudaMalloc( (void**)&bounds_lx_cur, sizeof(float)*NLANDMARKS_51*T ) );
    HANDLE_ERROR( cudaMalloc( (void**)&bounds_ly_cur, sizeof(float)*NLANDMARKS_51*T ) );



    HANDLE_ERROR( cudaMalloc( (void**)&AL, sizeof(float)*NLANDMARKS_51*3*ov->Kalpha ) );
    HANDLE_ERROR( cudaMalloc( (void**)&EL, sizeof(float)*NLANDMARKS_51*3*ov->Kepsilon ) );

    HANDLE_ERROR( cudaMalloc( (void**)&Rp, sizeof(float)*NLANDMARKS_51*3 ) );



    HANDLE_ERROR( cudaMalloc( (void**)&alpha_lb, sizeof(float)*ov->Kalpha ) );
    HANDLE_ERROR( cudaMalloc( (void**)&alpha_ub, sizeof(float)*ov->Kalpha ) );
    HANDLE_ERROR( cudaMalloc( (void**)&epsilon_lb, sizeof(float)*ov->Kepsilon ) );
    HANDLE_ERROR( cudaMalloc( (void**)&epsilon_ub, sizeof(float)*ov->Kepsilon ) );
    HANDLE_ERROR( cudaMalloc( (void**)&epsilon_lb_regular, sizeof(float)*ov->Kepsilon ) );
    HANDLE_ERROR( cudaMalloc( (void**)&epsilon_ub_regular, sizeof(float)*ov->Kepsilon ) );
    HANDLE_ERROR( cudaMalloc( (void**)&epsilon_lb_finetune, sizeof(float)*ov->Kepsilon ) );
    HANDLE_ERROR( cudaMalloc( (void**)&epsilon_ub_finetune, sizeof(float)*ov->Kepsilon ) );

    if (ov->Kbeta > 0) {
        HANDLE_ERROR( cudaMalloc( (void**)&beta_lb, sizeof(float)*ov->Kbeta) );
        HANDLE_ERROR( cudaMalloc( (void**)&beta_ub, sizeof(float)*ov->Kbeta) );

        HANDLE_ERROR( cudaMalloc( (void**)&f_beta_lb, sizeof(float)*(2*ov->Kbeta) ) );
        f_beta_ub = f_beta_lb + ov->Kbeta;
    }

    HANDLE_ERROR( cudaMalloc( (void**)&eps_l2weights, sizeof(float)*ov->Kepsilon ) );
    HANDLE_ERROR( cudaMalloc( (void**)&eps_l2weights_x2, sizeof(float)*ov->Kepsilon ) );

    HANDLE_ERROR( cudaMalloc( (void**)&deps_l2weights, sizeof(float)*ov->Kepsilon ) );
    HANDLE_ERROR( cudaMalloc( (void**)&deps_l2weights_x2, sizeof(float)*ov->Kepsilon ) );

    HANDLE_ERROR( cudaMalloc( (void**)&drigid_l2weights, sizeof(float)*6 ) );
    HANDLE_ERROR( cudaMalloc( (void**)&drigid_l2weights_x2, sizeof(float)*6 ) );

    vector< vector<float> > sigma_betas_vec =  read2DVectorFromFile<float>(config::SIGMA_BETAS_PATH, ov->Kbeta, 1);

    vector< vector<float> > sigma_alphas_vec;
    if (config::SIGMAS_FILE==0) {
        sigma_alphas_vec = read2DVectorFromFile<float>(config::SIGMA_ALPHAS_PATH, ov->Kalpha, 1);
    }



    vector< vector<float> > sigma_epsilons_vec_upper, sigma_epsilons_vec_lower;
    sigma_epsilons_vec_lower = read2DVectorFromFile<float>(config::EXP_LOWER_BOUND_PATH, ov->Kepsilon, 1);
    sigma_epsilons_vec_upper= read2DVectorFromFile<float>(config::EXP_UPPER_BOUND_PATH, ov->Kepsilon, 1);

#ifdef N_K_ALPHAS
    if (N_K_ALPHAS < 199 && ov->Kalpha == 199)
    {
        for (uint i=N_K_ALPHAS; i<199; ++i)
            sigma_alphas_vec[i][0] *= 0.05;
    }
#endif



    //    vector< vector<float> > bounds_ymin_vec = read2DVectorFromFile<float>("/home/v/car-vision/python/adaptive_confidence_interval/models_frozen/ymins.txt", NLANDMARKS_51*4, 1);
    //    vector< vector<float> > bounds_ymax_vec = read2DVectorFromFile<float>("/home/v/car-vision/python/adaptive_confidence_interval/models_frozen/ymaxs.txt", NLANDMARKS_51*4, 1);

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
            ss1 << "models/landmark_models/bounds/ymins_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_partbased << ".txt";
            ss2 << "models/landmark_models/bounds/ymaxs_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_partbased << ".txt";
        }
        else
        {
            ss1 << "models/landmark_models/bounds/ymins_nopart_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_global << ".txt";
            ss2 << "models/landmark_models/bounds/ymaxs_nopart_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_global << ".txt";
        }
        vector< vector<float> > bounds_ymin_vec_ = read2DVectorFromFile<float>(ss1.str(), 4, 1);
        vector< vector<float> > bounds_ymax_vec_ = read2DVectorFromFile<float>(ss2.str(), 4, 1);
        vec_bounds_ymin.push_back(vec2arr(bounds_ymin_vec_, 4, 1, false));
        vec_bounds_ymax.push_back(vec2arr(bounds_ymax_vec_, 4, 1, false));

        if (config::USE_LOCAL_MODELS) {
            ss3 << "models/landmark_models/bounds/bound_estimator_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_partbased << ".pb";
        } else {
            ss3 << "models/landmark_models/bounds/bound_estimator_nopart_" << std::setfill('0') << std::setw(2) << i+1 << "_" << extension_global << ".pb";
        }
        vec_bound_estimator.push_back(cv::dnn::readNetFromTensorflow(ss3.str()));
    }

    vector< vector<float> > bounds_lx_vec = read2DVectorFromFile<float>("models/dat_files/bounds_fao_lx.dat_36_1.50_partbased", NLANDMARKS_51*N_ANGLE_COMBINATIONS, 1);
    vector< vector<float> > bounds_ly_vec = read2DVectorFromFile<float>("models/dat_files/bounds_fao_ly.dat_36_1.50_partbased", NLANDMARKS_51*N_ANGLE_COMBINATIONS, 1);
    vector< vector<float> > bounds_ux_vec = read2DVectorFromFile<float>("models/dat_files/bounds_fao_ux.dat_36_1.50_partbased", NLANDMARKS_51*N_ANGLE_COMBINATIONS, 1);
    vector< vector<float> > bounds_uy_vec = read2DVectorFromFile<float>("models/dat_files/bounds_fao_uy.dat_36_1.50_partbased", NLANDMARKS_51*N_ANGLE_COMBINATIONS, 1);

    std::string AL_path, EL_path;

    if (ov->for_landmarks)
    {
        AL_path = config::AL60_PATH;
        EL_path = config::EL_PATH;
    }
    else
    {
        AL_path = config::AL_FULL_PATH;
        EL_path = config::EL_FULL_PATH;
    }

    vector< vector<float> > AL_vec = read2DVectorFromFile<float>(AL_path, NLANDMARKS_51*3, ov->Kalpha);
    vector< vector<float> > EL_vec = read2DVectorFromFile<float>(EL_path, NLANDMARKS_51*3, ov->Kepsilon);

    float *h_AL, *h_EL;
    h_AL = vec2arr(AL_vec, NLANDMARKS_51*3, ov->Kalpha, false);
    h_EL = vec2arr(EL_vec, NLANDMARKS_51*3, ov->Kepsilon, false);
    set_dictionaries_from_host(h_AL, h_EL, ov);

    float *h_bounds_lx, *h_bounds_ly, *h_bounds_ux, *h_bounds_uy;

    //    bdelta *= 0.8;
    //    bdelta *= 1.25;

    //    if (T == 1) {
    //        bdelta *= 1.25f;
    //    }

    h_bounds_lx = vec2arr(bounds_lx_vec, NLANDMARKS_51*N_ANGLE_COMBINATIONS, 1, false, bdelta);
    h_bounds_ly = vec2arr(bounds_ly_vec, NLANDMARKS_51*N_ANGLE_COMBINATIONS, 1, false, bdelta);
    h_bounds_ux = vec2arr(bounds_ux_vec, NLANDMARKS_51*N_ANGLE_COMBINATIONS, 1, false, bdelta);
    h_bounds_uy = vec2arr(bounds_uy_vec, NLANDMARKS_51*N_ANGLE_COMBINATIONS, 1, false, bdelta);

    set_bounds_from_host( h_bounds_lx, h_bounds_ly, h_bounds_ux, h_bounds_uy);

    /**
     * std::min_element(v.begin(), v.end()) - v.begin()
     */

    // This is what we've mostly been using
    // float opts_delta = 0.65f;
    // float opts_delta_eps = 1.35f;


    //    float opts_delta = 0.6f;
    //    float opts_delta_eps = 1.1f;
    //    float opts_beta = 2.0f*opts_delta;

    float opts_delta = config::OPTS_DELTA; //0.85f;
    float opts_delta_eps = config::OPTS_DELTA_EPS; //1.3f;
    float opts_beta = config::OPTS_DELTA_BETA;// 2.5f*opts_delta;

    float *h_alpha_lb, *h_alpha_ub, *h_epsilon_lb, *h_epsilon_ub, *h_epsilon_lb_finetune, *h_epsilon_ub_finetune, *h_beta_lb = NULL, *h_beta_ub = NULL;
    if (ov->Kbeta > 0) {
        h_beta_lb = vec2arr(sigma_betas_vec, ov->Kbeta, 1, false, -opts_beta);
        h_beta_ub = vec2arr(sigma_betas_vec, ov->Kbeta, 1, false, opts_beta);
    }

    h_alpha_lb = vec2arr(sigma_alphas_vec, ov->Kalpha, 1, false, -opts_delta);
    h_alpha_ub = vec2arr(sigma_alphas_vec, ov->Kalpha, 1, false, opts_delta);

    h_epsilon_lb = vec2arr(sigma_epsilons_vec_lower, ov->Kepsilon, 1, false, opts_delta_eps);
    h_epsilon_ub = vec2arr(sigma_epsilons_vec_upper, ov->Kepsilon, 1, false, opts_delta_eps);
    h_epsilon_lb_finetune = vec2arr(sigma_epsilons_vec_lower, ov->Kepsilon, 1, false, config::FINETUNE_COEF*opts_delta_eps);
    h_epsilon_ub_finetune = vec2arr(sigma_epsilons_vec_upper, ov->Kepsilon, 1, false, config::FINETUNE_COEF*opts_delta_eps);

    if (config::USE_EXPR_COMPONENT != -1)
    {
        for (int ei=0; ei<(int)ov->Kepsilon; ++ei) {
            if (config::USE_EXPR_COMPONENT == ei)
                continue;

            h_epsilon_lb[ei] *= 0.00001;
            h_epsilon_ub[ei] *= 0.00001;
        }
    }

    if (ov->Kbeta > 0) {
        HANDLE_ERROR( cudaMemcpy( beta_lb, h_beta_lb, sizeof(float)*ov->Kbeta, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( beta_ub, h_beta_ub, sizeof(float)*ov->Kbeta, cudaMemcpyHostToDevice ) );
    }

    HANDLE_ERROR( cudaMemcpy( alpha_lb, h_alpha_lb, sizeof(float)*ov->Kalpha, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( alpha_ub, h_alpha_ub, sizeof(float)*ov->Kalpha, cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( epsilon_lb, h_epsilon_lb, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( epsilon_ub, h_epsilon_ub, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( epsilon_lb_regular, h_epsilon_lb, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( epsilon_ub_regular, h_epsilon_ub, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( epsilon_lb_finetune, h_epsilon_lb_finetune, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( epsilon_ub_finetune, h_epsilon_ub_finetune, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );

//    vector< vector<float> > h_eps_l2weights_vec = read2DVectorFromFile<float>(***, ov->Kepsilon, 1);
//    h_eps_l2weights = vec2arr(h_eps_l2weights_vec, ov->Kepsilon, 1, false, 1.0f);

    if (config::L2_TRAIN_MODE)
    {
        float *h_eps_l2weights, *h_eps_l2weights_x2;
        h_eps_l2weights =  (float*)malloc( config::K_EPSILON*sizeof(float) );
        h_eps_l2weights_x2 = (float*)malloc( config::K_EPSILON*sizeof(float) );

        for (int ei=0; ei<(int)config::K_EPSILON; ++ei) {
            if (ei == config::USE_EXPR_COMPONENT || config::USE_EXPR_COMPONENT == -1) {
                h_eps_l2weights[ei] = config::EXPR_L2_WEIGHT;
                h_eps_l2weights_x2[ei] = 2*h_eps_l2weights[ei];
            }
            else {
                h_eps_l2weights[ei] = 0.0f;
                h_eps_l2weights_x2[ei] = 0.0f;
            }
        }

        HANDLE_ERROR( cudaMemcpy( eps_l2weights, h_eps_l2weights, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( eps_l2weights_x2, h_eps_l2weights_x2, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );

        free( h_eps_l2weights );
        free( h_eps_l2weights_x2 );
    }
    else
    {
        float *h_eps_l2weights = (float*)malloc( config::K_EPSILON*sizeof(float) );
        float *h_eps_l2weights_x2 = (float*)malloc( config::K_EPSILON*sizeof(float) );

        float fw = config::EXPR_L2_WEIGHT;

        vector<vector<float> > initws = read2DVectorFromFile<float>(config::EXP_UPPER_BOUND_PATH, ov->Kepsilon, 1);

        for (size_t ei=0; ei<config::K_EPSILON; ++ei) {
            if (config::EXPR_UNIFORM_REG)
            {
                h_eps_l2weights[ei] = fw;
                h_eps_l2weights_x2[ei] = 2*fw;
            }
            else
            {
                h_eps_l2weights[ei] = fw/initws[ei][0];
                h_eps_l2weights_x2[ei] = 2*fw/initws[ei][0];
            }
        }

        HANDLE_ERROR( cudaMemcpy( eps_l2weights, h_eps_l2weights, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( eps_l2weights_x2, h_eps_l2weights_x2, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );

        free( h_eps_l2weights );
        free( h_eps_l2weights_x2 );
    }

    if (use_temp_smoothing)
    {
        //! (1) Expressions smoothing
        float *h_deps_l2weights = (float*)malloc( config::K_EPSILON*sizeof(float) );
        float *h_deps_l2weights_x2 = (float*)malloc( config::K_EPSILON*sizeof(float) );

        const float fw = config::DEXPR_L2_WEIGHT;

        for (int ei=0; ei<(int)config::K_EPSILON; ++ei) {
            h_deps_l2weights[ei] = fw;
            h_deps_l2weights_x2[ei] = 2*fw;
        }

        HANDLE_ERROR( cudaMemcpy( deps_l2weights, h_deps_l2weights, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( deps_l2weights_x2, h_deps_l2weights_x2, sizeof(float)*ov->Kepsilon, cudaMemcpyHostToDevice ) );

        free( h_deps_l2weights );
        free( h_deps_l2weights_x2 );


        //! (2) Rigid smoothing
        float *h_drigid_l2weights_x2;
        h_drigid_l2weights_x2 = (float*)malloc( 6*sizeof(float) );
        const float fwr = config::DPOSE_L2_WEIGHT;

        float h_drigid_l2weights[6] = {fwr,fwr,fwr,fwr,fwr,fwr};
        for (int ei=0; ei<(int)6; ++ei)
            h_drigid_l2weights_x2[ei] = 2*h_drigid_l2weights[ei];

        HANDLE_ERROR( cudaMemcpy( drigid_l2weights, h_drigid_l2weights, sizeof(float)*6, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( drigid_l2weights_x2, h_drigid_l2weights_x2, sizeof(float)*6, cudaMemcpyHostToDevice ) );

        free( h_drigid_l2weights_x2 );
    }

    free(h_bounds_lx);
    free(h_bounds_ly);
    free(h_bounds_ux);
    free(h_bounds_uy);

    free(h_alpha_lb);
    free(h_alpha_ub);

    if (ov->Kbeta > 0)
    {
        free(h_beta_lb);
        free(h_beta_ub);
    }

    free(h_epsilon_lb);
    free(h_epsilon_ub);
    free(h_epsilon_lb_finetune);
    free(h_epsilon_ub_finetune);

    free(h_AL);
    free(h_EL);

    for (uint landmark_i=0; landmark_i<config::LIS.size(); ++landmark_i) {
        HANDLE_ERROR( cudaMemcpy( p0L_mat + landmark_i*3, r.X0_mean + config::LIS[landmark_i], sizeof(float), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( p0L_mat + landmark_i*3 + 1, r.Y0_mean + config::LIS[landmark_i], sizeof(float), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( p0L_mat + landmark_i*3 + 2, r.Z0_mean + config::LIS[landmark_i], sizeof(float), cudaMemcpyHostToDevice ) );
    }


    HANDLE_ERROR( cudaMalloc( (void**)&for_nabla2F_dsdc, sizeof(float)*NLANDMARKS_51*Ktotal_base) );

    HANDLE_ERROR( cudaMalloc( (void**)&nablaWx, sizeof(float)*NLANDMARKS_51*Ktotal ) );
    HANDLE_ERROR( cudaMalloc( (void**)&nablaWy, sizeof(float)*NLANDMARKS_51*Ktotal ) );

    HANDLE_ERROR( cudaMalloc( (void**)&for_nablaPhi_Gx_minus, sizeof(float)*NLANDMARKS_51*Ktotal ) );
    HANDLE_ERROR( cudaMalloc( (void**)&for_nablaPhi_Gy_minus, sizeof(float)*NLANDMARKS_51*Ktotal ) );
    HANDLE_ERROR( cudaMalloc( (void**)&for_nablaPhi_Gx_plus, sizeof(float)*NLANDMARKS_51*Ktotal ) );
    HANDLE_ERROR( cudaMalloc( (void**)&for_nablaPhi_Gy_plus, sizeof(float)*NLANDMARKS_51*Ktotal ) );
    HANDLE_ERROR( cudaMalloc( (void**)&nabla2F_dsdc, sizeof(float)*Ktotal_base ) );





    uint length_vecOnes = max( ov->Kalpha, max(ov->Kepsilon, NLANDMARKS_51*4));


    HANDLE_ERROR( cudaMalloc( (void**)&vecOnes, sizeof(float)*length_vecOnes ) );
    set_vec_ones<<<(NLANDMARKS_51*4+NTHREADS-1)/NTHREADS, NTHREADS>>>(vecOnes, NLANDMARKS_51*4);
}


__global__ void compute_Gs(
        const float *varx, const float phix, const float phiy, const float cx, const float cy, const float face_size, const float *Rp,
        const float *bounds_lx, const float *bounds_ly, const float *bounds_ux, const float *bounds_uy, const float *xl, const float *yl,
        float *Gx_minus, float *Gy_minus, float *Gx_plus, float *Gy_plus,
        float *inv_Gx_minus, float *inv_Gy_minus, float *inv_Gx_plus, float *inv_Gy_plus,
        float *nlog_Gx_minus, float *nlog_Gy_minus, float *nlog_Gx_plus, float *nlog_Gy_plus, const float *slack__)
{
    const ushort rowix = blockIdx.x;
    const ushort colix = threadIdx.x;

    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////

    ushort i = colix + rowix*blockDim.x;

    if (i >= NLANDMARKS_51)
        return;

    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float tauz[1];
    __shared__ float slack[1];

    if (colix == 0)
    {
        taux[0] = varx[0];
        tauy[0] = varx[1];
        tauz[0] = varx[2];
        slack[0] = slack__[0];
    }


    __syncthreads();

    const float r1p = Rp[i*3];
    const float r2p = Rp[i*3+1];
    const float r3p = Rp[i*3+2];

    const float mineps = 0.000001f;

    const float xproj = phix*((r1p+taux[0])/(r3p+tauz[0]))+cx;
    const float yproj = phiy*((r2p+tauy[0])/(r3p+tauz[0]))+cy;

    Gx_minus[i] = slack[0]-(xl[i] + bounds_lx[i]*face_size - xproj);
    Gy_minus[i] = slack[0]-(yl[i] + bounds_ly[i]*face_size - yproj);
    Gx_plus[i] = slack[0]-(xproj - bounds_ux[i]*face_size - xl[i]);
    Gy_plus[i] = slack[0]-(yproj - bounds_uy[i]*face_size - yl[i]);

    inv_Gx_minus[i] = 1.0f/(mineps+Gx_minus[i]);
    inv_Gy_minus[i] = 1.0f/(mineps+Gy_minus[i]);
    inv_Gx_plus[i] = 1.0f/(mineps+Gx_plus[i]);
    inv_Gy_plus[i] = 1.0f/(mineps+Gy_plus[i]);

    nlog_Gx_minus[i] = -logf(Gx_minus[i]);
    nlog_Gy_minus[i] = -logf(Gy_minus[i]);
    nlog_Gx_plus[i] = -logf(Gx_plus[i]);
    nlog_Gy_plus[i] = -logf(Gy_plus[i]);
}




__global__ void compute_xyproj(
        const float *varx, const float phix, const float phiy, const float cx, const float cy,  const float *Rp,
        const float *xl, const float *yl, float *xproj, float *yproj)
{
    const ushort rowix = blockIdx.x;
    const ushort colix = threadIdx.x;

    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////

    ushort i = colix + rowix*blockDim.x;

    if (i >= NLANDMARKS_51)
        return;

    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float tauz[1];
    __shared__ float slack[1];

    if (colix == 0)
    {
        taux[0] = varx[0];
        tauy[0] = varx[1];
        tauz[0] = varx[2];
    }


    __syncthreads();

    const float r1p = Rp[i*3];
    const float r2p = Rp[i*3+1];
    const float r3p = Rp[i*3+2];

    const float mineps = 0.000001f;

    xproj[i] = phix*((r1p+taux[0])/(r3p+tauz[0]))+cx;
    yproj[i] = phiy*((r2p+tauy[0])/(r3p+tauz[0]))+cy;
}





__global__ void neglogify(float *vec, const uint N)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= N)
        return;

    vec[n] = -logf(-vec[n]);
}





__global__ void fill_optimization_dc_landmark(const float phix, const float phiy,
                                              const float *varx,
                                              const float *Rp, const float *p,
                                              const float *dR_du1__, const float *dR_du2__, const float *dR_du3__,
                                              const float *Gx_minus, const float *Gy_minus, const float *Gx_plus, const float *Gy_plus,
                                              float *nablaWx_c, float *nablaWy_c,
                                              float *for_nablaPhi_Gx_minus, float *for_nablaPhi_Gy_minus, float *for_nablaPhi_Gx_plus, float *for_nablaPhi_Gy_plus)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= NLANDMARKS_51)
        return;

    // This is a very critical step -- those commonly accessed
    // variables need to be made shared (and copied as below)
    // otherwise code runs super slowly
    __shared__ float dR_du1[9];
    __shared__ float dR_du2[9];
    __shared__ float dR_du3[9];

    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float tauz[1];

    if (colix < 9) {
        dR_du1[colix] = dR_du1__[colix];
        dR_du2[colix] = dR_du2__[colix];
        dR_du3[colix] = dR_du3__[colix];
    }

    if (colix == 0) {
        taux[0] = varx[0];
        tauy[0] = varx[1];
        tauz[0] = varx[2];
    }

    __syncthreads();

    const float px = p[n*3];
    const float py = p[n*3+1];
    const float pz = p[n*3+2];

    const float vx = Rp[n*3]+taux[0];
    const float vy = Rp[n*3+1]+tauy[0];
    const float vz = Rp[n*3+2]+tauz[0];

    const float inv_vz = 1.0f/vz;
    const float inv_vz2 = inv_vz*inv_vz;


    const float dvx_du1 = dR_du1[0]*px + dR_du1[3]*py + dR_du1[6]*pz;
    const float dvy_du1 = dR_du1[1]*px + dR_du1[4]*py + dR_du1[7]*pz;
    const float dvz_du1 = dR_du1[2]*px + dR_du1[5]*py + dR_du1[8]*pz;

    const float dvx_du2 = dR_du2[0]*px + dR_du2[3]*py + dR_du2[6]*pz;
    const float dvy_du2 = dR_du2[1]*px + dR_du2[4]*py + dR_du2[7]*pz;
    const float dvz_du2 = dR_du2[2]*px + dR_du2[5]*py + dR_du2[8]*pz;

    const float dvx_du3 = dR_du3[0]*px + dR_du3[3]*py + dR_du3[6]*pz;
    const float dvy_du3 = dR_du3[1]*px + dR_du3[4]*py + dR_du3[7]*pz;
    const float dvz_du3 = dR_du3[2]*px + dR_du3[5]*py + dR_du3[8]*pz;

    nablaWx_c[n] = phix * inv_vz;
    nablaWx_c[n+NLANDMARKS_51] = 0;
    nablaWx_c[n+NLANDMARKS_51*2] = -(phix) * vx * inv_vz2;

    nablaWx_c[n+NLANDMARKS_51*3] = (phix)*inv_vz2*(vz*dvx_du1-vx*dvz_du1); // (*phix)*inv_vz2[rowix]*(vz[rowix]*_dvx_du1-vx[rowix]*_dvz_du1);
    nablaWx_c[n+NLANDMARKS_51*4] = (phix)*inv_vz2*(vz*dvx_du2-vx*dvz_du2); //(*phix)*inv_vz2[rowix]*(vz[rowix]*_dvx_du2-vx[rowix]*_dvz_du2);
    nablaWx_c[n+NLANDMARKS_51*5] = (phix)*inv_vz2*(vz*dvx_du3-vx*dvz_du3); //(*phix)*inv_vz2[rowix]*(vz[rowix]*_dvx_du3-vx[rowix]*_dvz_du3);



    nablaWy_c[n] = 0;
    nablaWy_c[n+NLANDMARKS_51] = (phiy) * inv_vz;
    nablaWy_c[n+NLANDMARKS_51*2] = -(phiy) * vy* inv_vz2;

    nablaWy_c[n+NLANDMARKS_51*3] = (phiy)*inv_vz2*(vz*dvy_du1-vy*dvz_du1); //(*phiy)*inv_vz2[rowix]*(vz[rowix]*_dvy_du1-vy[rowix]*_dvz_du1);
    nablaWy_c[n+NLANDMARKS_51*4] = (phiy)*inv_vz2*(vz*dvy_du2-vy*dvz_du2); //(*phiy)*inv_vz2[rowix]*(vz[rowix]*_dvy_du2-vy[rowix]*_dvz_du2);
    nablaWy_c[n+NLANDMARKS_51*5] = (phiy)*inv_vz2*(vz*dvy_du3-vy*dvz_du3); //(*phiy)*inv_vz2[rowix]*(vz[rowix]*_dvy_du3-vy[rowix]*_dvz_du3);

    for_nablaPhi_Gx_minus[n+NLANDMARKS_51*0] = -nablaWx_c[n+NLANDMARKS_51*0]/Gx_minus[n];
    for_nablaPhi_Gx_minus[n+NLANDMARKS_51*1] = -nablaWx_c[n+NLANDMARKS_51*1]/Gx_minus[n];
    for_nablaPhi_Gx_minus[n+NLANDMARKS_51*2] = -nablaWx_c[n+NLANDMARKS_51*2]/Gx_minus[n];
    for_nablaPhi_Gx_minus[n+NLANDMARKS_51*3] = -nablaWx_c[n+NLANDMARKS_51*3]/Gx_minus[n];
    for_nablaPhi_Gx_minus[n+NLANDMARKS_51*4] = -nablaWx_c[n+NLANDMARKS_51*4]/Gx_minus[n];
    for_nablaPhi_Gx_minus[n+NLANDMARKS_51*5] = -nablaWx_c[n+NLANDMARKS_51*5]/Gx_minus[n];

    for_nablaPhi_Gy_minus[n+NLANDMARKS_51*0] = -nablaWy_c[n+NLANDMARKS_51*0]/Gy_minus[n];
    for_nablaPhi_Gy_minus[n+NLANDMARKS_51*1] = -nablaWy_c[n+NLANDMARKS_51*1]/Gy_minus[n];
    for_nablaPhi_Gy_minus[n+NLANDMARKS_51*2] = -nablaWy_c[n+NLANDMARKS_51*2]/Gy_minus[n];
    for_nablaPhi_Gy_minus[n+NLANDMARKS_51*3] = -nablaWy_c[n+NLANDMARKS_51*3]/Gy_minus[n];
    for_nablaPhi_Gy_minus[n+NLANDMARKS_51*4] = -nablaWy_c[n+NLANDMARKS_51*4]/Gy_minus[n];
    for_nablaPhi_Gy_minus[n+NLANDMARKS_51*5] = -nablaWy_c[n+NLANDMARKS_51*5]/Gy_minus[n];

    for_nablaPhi_Gx_plus[n+NLANDMARKS_51*0] = nablaWx_c[n+NLANDMARKS_51*0]/Gx_plus[n];
    for_nablaPhi_Gx_plus[n+NLANDMARKS_51*1] = nablaWx_c[n+NLANDMARKS_51*1]/Gx_plus[n];
    for_nablaPhi_Gx_plus[n+NLANDMARKS_51*2] = nablaWx_c[n+NLANDMARKS_51*2]/Gx_plus[n];
    for_nablaPhi_Gx_plus[n+NLANDMARKS_51*3] = nablaWx_c[n+NLANDMARKS_51*3]/Gx_plus[n];
    for_nablaPhi_Gx_plus[n+NLANDMARKS_51*4] = nablaWx_c[n+NLANDMARKS_51*4]/Gx_plus[n];
    for_nablaPhi_Gx_plus[n+NLANDMARKS_51*5] = nablaWx_c[n+NLANDMARKS_51*5]/Gx_plus[n];

    for_nablaPhi_Gy_plus[n+NLANDMARKS_51*0] = nablaWy_c[n+NLANDMARKS_51*0]/Gy_plus[n];
    for_nablaPhi_Gy_plus[n+NLANDMARKS_51*1] = nablaWy_c[n+NLANDMARKS_51*1]/Gy_plus[n];
    for_nablaPhi_Gy_plus[n+NLANDMARKS_51*2] = nablaWy_c[n+NLANDMARKS_51*2]/Gy_plus[n];
    for_nablaPhi_Gy_plus[n+NLANDMARKS_51*3] = nablaWy_c[n+NLANDMARKS_51*3]/Gy_plus[n];
    for_nablaPhi_Gy_plus[n+NLANDMARKS_51*4] = nablaWy_c[n+NLANDMARKS_51*4]/Gy_plus[n];
    for_nablaPhi_Gy_plus[n+NLANDMARKS_51*5] = nablaWy_c[n+NLANDMARKS_51*5]/Gy_plus[n];
}


__global__ void fill_optimization_dI_dalpha_landmark(const float phix, const float phiy,
                                                     const float *varx, const float *R__, const float *Rp, const float *AL,
                                                     const float *Gx_minus, const float *Gy_minus, const float *Gx_plus, const float *Gy_plus,
                                                     float *nablaWx_alpha, float *nablaWy_alpha,
                                                     float *for_nablaPhi_Gx_minus, float *for_nablaPhi_Gy_minus, float *for_nablaPhi_Gx_plus, float *for_nablaPhi_Gy_plus,  const uint Kalpha)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= NLANDMARKS_51*Kalpha)
        return;

    const int i = n % NLANDMARKS_51;
    const int j = n / NLANDMARKS_51;

    __shared__ float R[9];

    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float tauz[1];

    if (colix < 9) {
        R[colix] = R__[colix];
    }

    if (colix == 0) {
        taux[0] = varx[0];
        tauy[0] = varx[1];
        tauz[0] = varx[2];
    }

    __syncthreads();

    const float R00 = R[0];
    const float R10 = R[1];
    const float R20 = R[2];
    const float R01 = R[3];
    const float R11 = R[4];
    const float R21 = R[5];
    const float R02 = R[6];
    const float R12 = R[7];
    const float R22 = R[8];

    const float r1p = Rp[i*3];
    const float r2p = Rp[i*3+1];
    const float r3p = Rp[i*3+2];

    float vx = r1p+taux[0];
    float vy = r2p+tauy[0];
    float vz = r3p+tauz[0];

    float inv_vz = 1.0f/vz;
    float inv_vz2 = inv_vz*inv_vz;

    float RIXij = AL[i*3 + j*NLANDMARKS_51*3];
    float RIYij = AL[i*3+1 + j*NLANDMARKS_51*3];
    float RIZij = AL[i*3+2 + j*NLANDMARKS_51*3];

    float dvxi_dalphaj = RIXij*R00 + RIYij*R01 + RIZij*R02;
    float dvyi_dalphaj = RIXij*R10 + RIYij*R11 + RIZij*R12;
    float dvzi_dalphaj = RIXij*R20 + RIYij*R21 + RIZij*R22;

    const float nablaWx_alpha_it = phix*inv_vz2*(dvxi_dalphaj*vz - dvzi_dalphaj*vx);
    const float nablaWy_alpha_it = phiy*inv_vz2*(dvyi_dalphaj*vz - dvzi_dalphaj*vy);

    //	nablaWx_alpha = phix*inv_vz2.*(dvx_dalpha.*vz - dvz_dalpha.*vx);


    nablaWx_alpha[i+j*NLANDMARKS_51] = nablaWx_alpha_it;
    nablaWy_alpha[i+j*NLANDMARKS_51] = nablaWy_alpha_it;

    for_nablaPhi_Gx_minus[i+j*NLANDMARKS_51] = -nablaWx_alpha_it/Gx_minus[i];
    for_nablaPhi_Gy_minus[i+j*NLANDMARKS_51] = -nablaWy_alpha_it/Gy_minus[i];
    for_nablaPhi_Gx_plus[i+j*NLANDMARKS_51] = nablaWx_alpha_it/Gx_plus[i];
    for_nablaPhi_Gy_plus[i+j*NLANDMARKS_51] = nablaWy_alpha_it/Gy_plus[i];

}

__global__ void fill_for_nabla2F_dsdc(const float *nablaWx, const float *nablaWy,
                                      const float *Gx_minus__, const float *Gy_minus__, const float *Gx_plus__, const float *Gy_plus__,
                                      const uint Ktotal, float *for_nabla2F_dsdc)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;


    if (n >= NLANDMARKS_51*Ktotal)
        return;

    __shared__ float Gx_minus[NLANDMARKS_51];
    __shared__ float Gy_minus[NLANDMARKS_51];
    __shared__ float Gx_plus[NLANDMARKS_51];
    __shared__ float Gy_plus[NLANDMARKS_51];

    if (colix < NLANDMARKS_51)
    {
        Gx_minus[colix] = Gx_minus__[colix];
        Gy_minus[colix] = Gy_minus__[colix];
        Gx_plus[colix] = Gx_plus__[colix];
        Gy_plus[colix] = Gy_plus__[colix];
    }

    const int i = n % NLANDMARKS_51;
    const int j = n / NLANDMARKS_51;

    __syncthreads();

    for_nabla2F_dsdc[ i+j*NLANDMARKS_51 ] =
            - nablaWx[i + j*NLANDMARKS_51]/(Gx_plus[i]*Gx_plus[i])
            - nablaWy[i + j*NLANDMARKS_51]/(Gy_plus[i]*Gy_plus[i])
            + nablaWx[i + j*NLANDMARKS_51]/(Gx_minus[i]*Gx_minus[i])
            + nablaWy[i + j*NLANDMARKS_51]/(Gy_minus[i]*Gy_minus[i]);
}



__global__ void fill_bottom_and_right_of_nabla2F(const float *for_nabla2F_dsdc, float *nabla2F, const uint Ktotal, const uint Ktotal_base)
{

    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int i = colix + rowix*blockDim.x;

    if (i >= Ktotal_base)
        return;

    nabla2F[Ktotal-1+i*Ktotal] = for_nabla2F_dsdc[i];
    nabla2F[i+Ktotal*(Ktotal-1)] = for_nabla2F_dsdc[i];
}


__global__ void fill_optimization_dI_depsilon_landmark(const float phix, const float phiy,
                                                       const float *varx, const float *R__, const float *Rp, const float *EL,
                                                       const float *Gx_minus, const float *Gy_minus, const float *Gx_plus, const float *Gy_plus,
                                                       float *nablaWx_epsilon, float *nablaWy_epsilon,
                                                       float *for_nablaPhi_Gx_minus, float *for_nablaPhi_Gy_minus, float *for_nablaPhi_Gx_plus, float *for_nablaPhi_Gy_plus, const uint Kepsilon)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= NLANDMARKS_51*Kepsilon)
        return;

    const int i = n % NLANDMARKS_51;
    const int j = n / NLANDMARKS_51;

    __shared__ float R[9];

    __shared__ float taux[1];
    __shared__ float tauy[1];
    __shared__ float tauz[1];

    if (colix < 9) {
        R[colix] = R__[colix];
    }

    if (colix == 0) {
        taux[0] = varx[0];
        tauy[0] = varx[1];
        tauz[0] = varx[2];
    }

    __syncthreads();

    const float R00 = R[0];
    const float R10 = R[1];
    const float R20 = R[2];
    const float R01 = R[3];
    const float R11 = R[4];
    const float R21 = R[5];
    const float R02 = R[6];
    const float R12 = R[7];
    const float R22 = R[8];

    const float r1p = Rp[i*3];
    const float r2p = Rp[i*3+1];
    const float r3p = Rp[i*3+2];

    float vx = r1p+taux[0];
    float vy = r2p+tauy[0];
    float vz = r3p+tauz[0];

    float inv_vz = 1.0f/vz;
    float inv_vz2 = inv_vz*inv_vz;

    float RIXij = EL[i*3 + j*NLANDMARKS_51*3];
    float RIYij = EL[i*3+1 + j*NLANDMARKS_51*3];
    float RIZij = EL[i*3+2 + j*NLANDMARKS_51*3];

    float dvxi_depsilonj = RIXij*R00 + RIYij*R01 + RIZij*R02;
    float dvyi_depsilonj = RIXij*R10 + RIYij*R11 + RIZij*R12;
    float dvzi_depsilonj = RIXij*R20 + RIYij*R21 + RIZij*R22;

    const float nablaWx_epsilon_it = phix*inv_vz2*(dvxi_depsilonj*vz - dvzi_depsilonj*vx);
    const float nablaWy_epsilon_it = phiy*inv_vz2*(dvyi_depsilonj*vz - dvzi_depsilonj*vy);

    nablaWx_epsilon[i+j*NLANDMARKS_51] = nablaWx_epsilon_it;
    nablaWy_epsilon[i+j*NLANDMARKS_51] = nablaWy_epsilon_it;

    for_nablaPhi_Gx_minus[i+j*NLANDMARKS_51] = -nablaWx_epsilon_it/Gx_minus[i];
    for_nablaPhi_Gy_minus[i+j*NLANDMARKS_51] = -nablaWy_epsilon_it/Gy_minus[i];
    for_nablaPhi_Gx_plus[i+j*NLANDMARKS_51] = nablaWx_epsilon_it/Gx_plus[i];
    for_nablaPhi_Gy_plus[i+j*NLANDMARKS_51] = nablaWy_epsilon_it/Gy_plus[i];
}




__global__ void update_gradient(const float *vec_lb, const float *vec_ub, float *gradient, uint Kvec, uint vec_offset)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= Kvec)
        return;

    const float nablaf_ub = -1.0f/vec_ub[n];
    const float nablaf_lb = 1.0f/vec_lb[n];

    gradient[n+vec_offset] += nablaf_ub + nablaf_lb;
}



__global__ void update_diagonal_of_hessian_wbounds(const float *vec_lb, const float *vec_ub, float *A, uint Kvec, uint Asize, uint vec_offset)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= Kvec)
        return;

    const float nablaf_ub = -1.0f/vec_ub[n];
    const float nablaf_lb = 1.0f/vec_lb[n];

    A[Asize*(n+vec_offset)+n+vec_offset] += nablaf_ub*nablaf_ub + nablaf_lb*nablaf_lb;
}



__global__ void update_diagonal_of_hessian_wvector(const float *vec, float *A, uint Kvec, uint Asize, uint vec_offset)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= Kvec)
        return;

    A[Asize*(n+vec_offset)+n+vec_offset] += vec[n];
}

__global__ void update_diags_and_offdiags_for_expdiffs(float *A, const int T, const int Keps, const int Ktotal, const float* weight)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int i = colix + rowix*blockDim.x;

    if (i >= Keps*T)
        return;

    int ei = i % Keps;

    // First do the diagonals
    if (i < Keps || i >= Keps*(T-1))
        A[i*Ktotal+i] = 2.0*weight[ei];
    else
        A[i*Ktotal+i] = 4.0*weight[ei];

    int row1 = i+Keps;
    int col1 = i;
    int row2 = i;
    int col2 = i+Keps;

    if (row1 < T*Keps && col1 < T*Keps)
        A[col1*Ktotal+row1] = -2.0*weight[ei];

    if (row2 < T*Keps && col2 < T*Keps)
        A[col2*Ktotal+row2] = -2.0*weight[ei];
}


__global__ void update_bottom_right_of_matrix(const float *A, float *B, const uint Asize, const uint offset)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= Asize*Asize)
        return;

    const int i = n % Asize;
    const int j = n / Asize;

    B[(Asize+offset)*(j + offset) + i + offset] += A[Asize*j + i];
}


__global__ void multiply_matrix_scalar(float *A, const float alpha, const uint Asize)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= Asize*Asize)
        return;

    A[n] *= alpha;
}


__global__ void multiply_vector_scalar(float *v, const float alpha, const uint vsize)
{
    const int rowix = blockIdx.x;
    const int colix = threadIdx.x;

    int n = colix + rowix*blockDim.x;

    if (n >= vsize)
        return;

    v[n] *= alpha;
}


void Logbarrier_Initializer::set_minimal_slack(cublasHandle_t &handle, OptimizationVariables* ov)
{
    float *min_slacks, *slack0;

    HANDLE_ERROR( cudaMalloc( (void**)&min_slacks, sizeof(float)*T) );

    for (uint t=0; t<T; t++)
    {
        get_minimal_slack_t(handle, ov, t);

        float h_tmp;
        cudaMemcpy(&h_tmp,
                   thrust::min_element(thrust::device, Gx_minus_tmp, Gx_minus_tmp+4*NLANDMARKS_51),
                   sizeof(float), cudaMemcpyDeviceToHost);

        h_tmp = 5.0f-h_tmp;
        cudaMemcpy(min_slacks+t, &h_tmp, sizeof(float), cudaMemcpyHostToDevice);
    }

    slack0 = thrust::max_element(thrust::device, min_slacks, min_slacks+T);

    HANDLE_ERROR( cudaMemcpy(ov->slack, slack0, sizeof(float), cudaMemcpyDeviceToDevice) );
    HANDLE_ERROR( cudaFree( min_slacks ) );
}



void Logbarrier_Initializer::get_minimal_slack_t(cublasHandle_t &handle,  OptimizationVariables* ov, uint t)
{
    ov->set_frame(t);
    rc.set_u_ptr(ov->u);
    rc.process();
    compute_nonrigid_shape(handle, ov);

    float alpha_ = 1;
    float beta_ = 0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, NLANDMARKS_51, 3, &alpha_, rc.R, 3, p, 3, &beta_, Rp, 3);

    //    int bound_offset = NLANDMARKS_51*angle_idx[t];
    int bound_offset = NLANDMARKS_51*t;


    compute_Gs<<<1, NLANDMARKS_51>>>(ov->taux, cams_ptr->at(t).h_phix, cams_ptr->at(t).h_phiy, cams_ptr->at(t).h_cx, cams_ptr->at(t).h_cy,
                                     face_sizes[t], Rp,
                                     bounds_lx_cur + bound_offset, bounds_ly_cur + bound_offset, bounds_ux_cur + bound_offset, bounds_uy_cur + bound_offset,
                                     xl+t*NLANDMARKS_51, yl+t*NLANDMARKS_51,
                                     Gx_minus_tmp, Gy_minus_tmp, Gx_plus_tmp, Gy_plus_tmp,
                                     inv_Gx_minus_tmp, inv_Gy_minus_tmp, inv_Gx_plus_tmp, inv_Gy_plus_tmp,
                                     nlog_Gx_minus, nlog_Gy_minus, nlog_Gx_plus, nlog_Gy_plus,
                                     ov->slack);
}


bool Logbarrier_Initializer::fit_model(cusolverDnHandle_t& handleDn,
                                       cublasHandle_t &handle,
                                       OptimizationVariables* ov,
                                       OptimizationVariables* ov_linesearch)
{
    float yummymulti[] = {10.0f}; //  {10.0f};
    float yummy[1] = {10.0f}; // {10.0f};
    float alpha_ = 1.0f;
    float beta_ = 0.0f;

    cudaMemcpy(ov->tau_logbarrier, yummymulti, sizeof(float), cudaMemcpyHostToDevice);

    float *logbarrier_multi_coef;
    HANDLE_ERROR( cudaMalloc( (void**)&logbarrier_multi_coef, sizeof(float)) );
    cudaMemcpy(logbarrier_multi_coef, yummy, sizeof(float), cudaMemcpyHostToDevice);


    float h_obj, h_obj_tmp, h_slack;
    float *d_obj, *d_obj_tmp;
    HANDLE_ERROR( cudaMalloc( (void**)&d_obj, sizeof(float)) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_obj_tmp, sizeof(float)) );



    fit_success = false;

    const uint MAXITER = 100;
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    for (uint tau_iter = 0; tau_iter<5; ++ tau_iter)
    {
        bool terminate = false;
        if (fit_success) 
            break;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha_, ov->tau_logbarrier, 1, logbarrier_multi_coef, 1, &beta_, ov->tau_logbarrier, 1);
        cudaMemcpy(ov_linesearch->tau_logbarrier, ov->tau_logbarrier, sizeof(float), cudaMemcpyDeviceToDevice);

        for (uint iter=0; iter<MAXITER; ++iter)
        {
            if (terminate)
                break;

            cudaMemcpy(&h_slack, ov->slack, sizeof(float), cudaMemcpyDeviceToHost);

            //////////////////////////////
            //////////////////////////////
            //////////////////////////////
            //////////////////////////////
            //////////////////////////////
            //////////////////////////////
//            if (!use_slack)
//            {
            /*
                for (uint timet=0; timet<ov->T; ++timet)
                {
                    cv::Mat emptyFrame(cams_ptr->at(timet).h_cy*2, cams_ptr->at(timet).h_cx*2, CV_32FC3, cv::Scalar::all(255));

                    float h_xl[NLANDMARKS_51], h_yl[NLANDMARKS_51];
                    float h_xproj[NLANDMARKS_51], h_yproj[NLANDMARKS_51];

                    float *d_xproj, *d_yproj;
                    cudaMalloc((void**)&d_xproj, sizeof(float)*NLANDMARKS_51);
                    cudaMalloc((void**)&d_yproj, sizeof(float)*NLANDMARKS_51);

                    HANDLE_ERROR( cudaMemcpy( h_xl, xl+timet*NLANDMARKS_51, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost ) );
                    HANDLE_ERROR( cudaMemcpy( h_yl, yl+timet*NLANDMARKS_51, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost ) );



                    compute_xyproj<<<1, NLANDMARKS_51>>>(ov->taux, cams_ptr->at(timet).h_phix, cams_ptr->at(timet).h_phiy, cams_ptr->at(timet).h_cx, cams_ptr->at(timet).h_cy,
                            Rp, xl+timet*NLANDMARKS_51, yl+timet*NLANDMARKS_51,
                            d_xproj, d_yproj);

                    cudaMemcpy(h_xproj, d_xproj, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_yproj, d_yproj, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost);

                    for (uint ui=0; ui<NLANDMARKS_51; ++ui) {
                        cv::Point2f ptd(h_xl[ui], h_yl[ui]);
                        cv::Point2f ptp(h_xproj[ui], h_yproj[ui]);
                        cv::circle(emptyFrame, ptd, 3, cv::Scalar(0,0,255), cv::FILLED, 8, 0);
                        cv::circle(emptyFrame, ptp, 3, cv::Scalar(255,128,0), cv::FILLED, 8, 0);
                    }

                    cudaFree(d_xproj);
                    cudaFree(d_yproj);

                    cv::imshow("emptyFrame", emptyFrame);
                    cv::waitKey(0);
                }

//                std::cout << "h_slack IS ==>" << h_slack << std::endl;
//            }
            */
            //////////////////////////////
            //////////////////////////////
            //////////////////////////////
            //////////////////////////////
            //////////////////////////////
            //////////////////////////////

            if (h_slack < -0.01f) {
                terminate = true;

//                 std::cout << "GOAL REACHED; h_slack is  " << h_slack << " for " << ov->T << " frames " << std::endl;
                fit_success = true;
                break;
            }

            compute_gradient_and_hessian(handle, ov, d_obj);

            copyMatsFloatToDouble<<<(s.n*s.n+NTHREADS-1)/NTHREADS, NTHREADS>>>(nabla2F, gradient, s.JTJ, s.dG_dtheta, s.n);
            if (!use_slack) {
//                std::cout << "Printin solver .. " << std::endl;
//                print_matrix(nabla2F, s.n, s.n);

//                print_matrix_double(s.JTJ, s.n, s.n);
//                print_matrix_double(s.dG_dtheta, 1, s.n);
            }
            ///////////////s_qr.solve(handleDn, handle);


            bool solve_success = s.solve(handleDn);
            if (!solve_success) {
    	        terminate = true;
                if (config::PRINT_WARNINGS)
                    std::cout << "failed to solve" << std::endl;
	        	break;
            }

            cudaMemcpy(&h_obj, d_obj, sizeof(float), cudaMemcpyDeviceToHost);


            float BETA = 0.5f;
            float t_coef = 1.0f;

            const int MAX_INNER_ITERS = 100;
            int inner_iter = 0;

            while (inner_iter < MAX_INNER_ITERS)
            {
                if (t_coef < 0.01f) {
                    terminate = true;
                    break;
                }

                set_xtmp<<<(s.n+NTHREADS-1)/NTHREADS, NTHREADS>>>(s.search_dir, ov->alphas, t_coef, ov_linesearch->alphas, s.n);
                evaluate_objective_function(handle, ov_linesearch, d_obj_tmp);
                cudaMemcpy(&h_obj_tmp, d_obj_tmp, sizeof(float), cudaMemcpyDeviceToHost);
                //std::cout << " \t == OBJ TMP == "  << "\t " << h_obj_tmp << " (t_coef is " << t_coef << ")" <<  " vs " << h_obj << std::endl;

                float tmp = 0.0f;

                if (h_obj_tmp < h_obj+tmp) {
                    cudaMemcpy(ov->alphas, ov_linesearch->alphas, sizeof(float)*s.n, cudaMemcpyDeviceToDevice);
                    break;
                }

                t_coef = t_coef * BETA;
                inner_iter++;
            }
        }
    }

    HANDLE_ERROR( cudaFree(d_obj) );
    HANDLE_ERROR( cudaFree(d_obj_tmp) );
    HANDLE_ERROR( cudaFree(logbarrier_multi_coef) );

    return fit_success;
}




void Logbarrier_Initializer::compute_gradient_and_hessian(cublasHandle_t &handle, OptimizationVariables* ov, float *obj)
{
    float plus_one = 1.0f;
    float beta_ = 0.0f;
    float minus_one = -1.0f;


    cudaMemset(obj, 0, sizeof(float));
    if (!use_slack) {
        cudaMemset(ov->slack, 0, sizeof(float));
    } else {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &plus_one, ov->tau_logbarrier, 1, ov->slack, 1, &beta_, obj, 1);
    }


    //	float *for_nabla2F_dsdc;
    cudaMemset(for_nabla2F_dsdc, 0, NLANDMARKS_51*Ktotal_base*sizeof(float));

    // the gradient
    cudaMemset(gradient, 0, Ktotal*sizeof(float));

    // the hessian
    cudaMemset(nabla2F, 0, Ktotal*Ktotal*sizeof(float));

    // variables needed for computing the gradient and the hessian
    cudaMemset(nablaWx, 0, Ktotal*NLANDMARKS_51*sizeof(float));
    cudaMemset(nablaWy, 0, Ktotal*NLANDMARKS_51*sizeof(float));

    /*
     * the 2nd deriv of objective function wrt theta *and* slack variable, namely \frac{\partial^2 f_0 }{\partial \theta \partial s}
     * see step IV in Figure page #45 of large notebook
     *
     * In MATLAB language, this is:
     * 		nabla2F_dsdc = sum(- nablaWx./((Gx_plus).^2) - nablaWy./((Gy_plus).^2) + nablaWx./((Gx_minus).^2) + nablaWy./((Gy_minus).^2),1);
     */
    //
    cudaMemset(nabla2F_dsdc, 0, Ktotal_base*sizeof(float));

    if (use_identity)
    {
        // This is the difference between the current alpha variable and the upper bound for alpha
        //     f_alpha_ub = -data.alpha_ub(1:K)+alpha';
        cudaMemcpy(f_alpha_ub, ov->alphas, sizeof(float)*ov->Kalpha, cudaMemcpyDeviceToDevice);
        cublasSaxpy(handle, ov->Kalpha, &minus_one, alpha_ub, 1, f_alpha_ub, 1);

        // This is the difference between the current alpha variable and the lower bound for alpha
        //     f_alpha_lb = -alpha'+data.alpha_lb(1:K);
        cudaMemcpy(f_alpha_lb, alpha_lb,  sizeof(float)*ov->Kalpha, cudaMemcpyDeviceToDevice);
        cublasSaxpy(handle, ov->Kalpha, &minus_one, ov->alphas, 1, f_alpha_lb, 1);
    }


    if (ov->Kbeta > 0) {
        // This is the difference between the current beta variable and the upper bound for beta
        //     f_beta_ub = -data.beta_ub(1:K)+beta';
        cudaMemcpy(f_beta_ub, ov->betas, sizeof(float)*ov->Kbeta, cudaMemcpyDeviceToDevice);
        cublasSaxpy(handle, ov->Kbeta, &minus_one, beta_ub, 1, f_beta_ub, 1);


        // This is the difference between the current beta variable and the lower bound for beta
        //     f_beta_lb = -beta'+data.beta_lb(1:K);
        cudaMemcpy(f_beta_lb, beta_lb,  sizeof(float)*ov->Kbeta, cudaMemcpyDeviceToDevice);
        cublasSaxpy(handle, ov->Kbeta, &minus_one, ov->betas, 1, f_beta_lb, 1);
    }

    float *nablaWx_alpha = nablaWx;
    float *nablaWy_alpha = nablaWy;
    uint offset = ov->Kalpha*use_identity;

    for (uint t=0; t<T; ++t)
    {
        cudaMemset(nablaWx, 0, NLANDMARKS_51*Ktotal*sizeof(float));
        cudaMemset(nablaWy, 0, NLANDMARKS_51*Ktotal*sizeof(float));

        /**
         * These four variables are needed for computing both the gradient and the hessian.
         * In MATLAB language, this is
         * 	 for_nablaPhix_Gx_minus   = nablaWx./(-Gx_minus);
         * 	 for_nablaPhix_Gy_minus   = nablaWy./(-Gy_minus);
         * 	 for_nablaPhix_Gx_plus    = nablaWx./(Gx_plus);
         * 	 for_nablaPhix_Gy_plus    = nablaWy./(Gy_plus);
         *
         * 	 Note that we are resetting these variables for each frame t
         */
        cudaMemset(for_nablaPhi_Gx_minus, 0, NLANDMARKS_51*Ktotal*sizeof(float));
        cudaMemset(for_nablaPhi_Gy_minus, 0, NLANDMARKS_51*Ktotal*sizeof(float));
        cudaMemset(for_nablaPhi_Gx_plus, 0, NLANDMARKS_51*Ktotal*sizeof(float));
        cudaMemset(for_nablaPhi_Gy_plus, 0, NLANDMARKS_51*Ktotal*sizeof(float));

        /**
         * This variable is needed for gradient only
         * In MATLAB language, what we'll do to this variable is
         *
         * 	nablaPhix =  for_nablaPhix_Gx_plus+for_nablaPhix_Gy_plus+for_nablaPhix_Gx_minus+for_nablaPhix_Gy_minus;
         * 	nablaPhis = -(1./Gx_plus + 1./Gy_plus + 1./Gx_minus + 1./Gy_minus);
         * 	nablaF = cat(2,nablaPhix,nablaPhis);
         */
        cudaMemset(nablaF, 0, Ktotal*NLANDMARKS_51*sizeof(float));

        float *nablaWx_epsilon, *nablaWy_epsilon;
        float *nablaWx_c, *nablaWy_c;


        uint epsilon_offset = (offset + t*ov->Kepsilon*use_expression)*NLANDMARKS_51;
        uint c_offset = (offset + T*ov->Kepsilon*use_expression + 6*t)*NLANDMARKS_51;

        nablaWx_epsilon = nablaWx + epsilon_offset;
        nablaWy_epsilon = nablaWy + epsilon_offset;
        nablaWx_c = nablaWx + c_offset;
        nablaWy_c = nablaWy + c_offset;


        ov->set_frame(t);
        rc.set_u_ptr(ov->u);
        rc.process();

        compute_nonrigid_shape(handle, ov);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, NLANDMARKS_51, 3, &plus_one, rc.R, 3, p, 3, &beta_, Rp, 3);

        //        int bound_offset = NLANDMARKS_51*angle_idx[t];
        int bound_offset = NLANDMARKS_51*t;


        compute_Gs<<<1, NLANDMARKS_51>>>(ov->taux, cams_ptr->at(t).h_phix, cams_ptr->at(t).h_phiy, cams_ptr->at(t).h_cx, cams_ptr->at(t).h_cy,
                                         face_sizes[t], Rp,
                                         bounds_lx_cur + bound_offset, bounds_ly_cur+ bound_offset, bounds_ux_cur+ bound_offset, bounds_uy_cur + bound_offset,
                                         xl+t*NLANDMARKS_51, yl+t*NLANDMARKS_51,
                                         Gx_minus_tmp, Gy_minus_tmp, Gx_plus_tmp, Gy_plus_tmp,
                                         inv_Gx_minus_tmp, inv_Gy_minus_tmp, inv_Gx_plus_tmp, inv_Gy_plus_tmp,
                                         nlog_Gx_minus, nlog_Gy_minus, nlog_Gx_plus, nlog_Gy_plus,
                                         ov->slack);

        if (use_identity)
        {
            /**
             * Here we are filling the parts of for_nablaPhi_Gx_minus, for_nablaPhi_Gy_minus, for_nablaPhi_Gx_plus, for_nablaPhi_Gy_plus,
             * that correspond to identity shape basis variable (i.e., alpha)
             */
            fill_optimization_dI_dalpha_landmark<<<(ov->Kalpha*NLANDMARKS_51+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                                                                 cams_ptr->at(t).h_phix, cams_ptr->at(t).h_phiy, ov->taux, rc.R, Rp, AL,
                                                                                                                 Gx_minus_tmp, Gy_minus_tmp, Gx_plus_tmp, Gy_plus_tmp,
                                                                                                                 nablaWx_alpha, nablaWy_alpha,
                                                                                                                 for_nablaPhi_Gx_minus, for_nablaPhi_Gy_minus, for_nablaPhi_Gx_plus, for_nablaPhi_Gy_plus,
                                                                                                                 ov->Kalpha);
        }

        if (use_expression)
        {
            /**
             * Here we are filling the parts of for_nablaPhi_Gx_minus, for_nablaPhi_Gy_minus, for_nablaPhi_Gx_plus, for_nablaPhi_Gy_plus,
             * that correspond to expression variable (i.e., epsilon)
             */
            fill_optimization_dI_depsilon_landmark<<<(ov->Kepsilon*NLANDMARKS_51+NTHREADS-1)/NTHREADS, NTHREADS>>>(
                                                                                                                     cams_ptr->at(t).h_phix, cams_ptr->at(t).h_phiy, ov->taux, rc.R, Rp, EL,
                                                                                                                     Gx_minus_tmp, Gy_minus_tmp, Gx_plus_tmp, Gy_plus_tmp,
                                                                                                                     nablaWx_epsilon, nablaWy_epsilon,
                                                                                                                     for_nablaPhi_Gx_minus+epsilon_offset, 	for_nablaPhi_Gy_minus+epsilon_offset,
                                                                                                                     for_nablaPhi_Gx_plus+epsilon_offset, 	for_nablaPhi_Gy_plus+epsilon_offset,
                                                                                                                     ov->Kepsilon);
        }

        /**
         * Here we are filling the parts of for_nablaPhi_Gx_minus, for_nablaPhi_Gy_minus, for_nablaPhi_Gx_plus, for_nablaPhi_Gy_plus,
         * that correspond to rigid transformation variables
         */
        fill_optimization_dc_landmark<<<1, NLANDMARKS_51>>>(cams_ptr->at(t).h_phix, cams_ptr->at(t).h_phiy,
                                                            ov->taux,
                                                            Rp, p,
                                                            rc.dR_du1, rc.dR_du2, rc.dR_du3,
                                                            Gx_minus_tmp, Gy_minus_tmp, Gx_plus_tmp, Gy_plus_tmp,
                                                            nablaWx_c, nablaWy_c,
                                                            for_nablaPhi_Gx_minus+c_offset, for_nablaPhi_Gy_minus+c_offset,
                                                            for_nablaPhi_Gx_plus+c_offset, 	for_nablaPhi_Gy_plus+c_offset);

        if (use_expression)
        {
            //     f_epsilon_ub = epsilon'-boundcoef*data.epsilon_ub(1:length(epsilon));
            cudaMemcpy(f_epsilon_ub, ov->epsilons, sizeof(float)*ov->Kepsilon, cudaMemcpyDeviceToDevice);
            cublasSaxpy(handle, ov->Kepsilon, &minus_one, epsilon_ub, 1, f_epsilon_ub, 1);

            //     f_epsilon_lb = boundcoef*data.epsilon_lb(1:length(epsilon))-epsilon';
            cudaMemcpy(f_epsilon_lb, epsilon_lb,  sizeof(float)*ov->Kepsilon, cudaMemcpyDeviceToDevice);
            cublasSaxpy(handle, ov->Kepsilon, &minus_one, ov->epsilons, 1, f_epsilon_lb, 1);
        }


        /**
         * The following lines update part of the Hessian (note that the Hessian, nabla2F, keeps being updated for t=1:T):
         * nabla2F    = nabla2F + for_nabla2Phi_Gx_plus'*for_nabla2Phi_Gx_plus;
         * nabla2F    = nabla2F + for_nabla2Phi_Gy_plus'*for_nabla2Phi_Gy_plus;
         * nabla2F    = nabla2F + for_nabla2Phi_Gx_minus'*for_nabla2Phi_Gx_minus;
         * nabla2F    = nabla2F + for_nabla2Phi_Gy_minus'*for_nabla2Phi_Gy_minus;
         *
         * The part of the hessian that we update can be seen in Figure#45 step I of the larbe notebook
         *
         */
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Ktotal, Ktotal, NLANDMARKS_51, &plus_one, for_nablaPhi_Gx_minus, NLANDMARKS_51, for_nablaPhi_Gx_minus, NLANDMARKS_51, &plus_one, nabla2F, Ktotal);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Ktotal, Ktotal, NLANDMARKS_51, &plus_one, for_nablaPhi_Gy_minus, NLANDMARKS_51, for_nablaPhi_Gy_minus, NLANDMARKS_51, &plus_one, nabla2F, Ktotal);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Ktotal, Ktotal, NLANDMARKS_51, &plus_one, for_nablaPhi_Gx_plus, NLANDMARKS_51, for_nablaPhi_Gx_plus, NLANDMARKS_51, &plus_one, nabla2F, Ktotal);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Ktotal, Ktotal, NLANDMARKS_51, &plus_one, for_nablaPhi_Gy_plus, NLANDMARKS_51, for_nablaPhi_Gy_plus, NLANDMARKS_51, &plus_one, nabla2F, Ktotal);


        /**
         * Here we update part of the gradient.
         *
         * In MATLAB language, the following 5 lines do the following:
         *
         * 	nablaF =  for_nablaPhix_Gx_plus+for_nablaPhix_Gy_plus+for_nablaPhix_Gx_minus+for_nablaPhix_Gy_minus;
         * 	gradient = gradient + sum(nablaF,1);
         *
         * 	(Note that, unlike nabla2F, the variable nablaF is reset... Because, unlike nabla2F, nablaF is a temporary variable)
         *
         *  These operations can be seen in Figure #43 (step I) of large notebook#1
         */
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NLANDMARKS_51, Ktotal, &plus_one, nablaF, NLANDMARKS_51, &plus_one, for_nablaPhi_Gx_minus, NLANDMARKS_51, nablaF, NLANDMARKS_51);
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NLANDMARKS_51, Ktotal, &plus_one, nablaF, NLANDMARKS_51, &plus_one, for_nablaPhi_Gy_minus, NLANDMARKS_51, nablaF, NLANDMARKS_51);
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NLANDMARKS_51, Ktotal, &plus_one, nablaF, NLANDMARKS_51, &plus_one, for_nablaPhi_Gx_plus, NLANDMARKS_51, nablaF, NLANDMARKS_51);
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, NLANDMARKS_51, Ktotal, &plus_one, nablaF, NLANDMARKS_51, &plus_one, for_nablaPhi_Gy_plus, NLANDMARKS_51, nablaF, NLANDMARKS_51);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, Ktotal, NLANDMARKS_51, &plus_one, vecOnes, 1, nablaF, NLANDMARKS_51, &plus_one, gradient, 1); // this is for nablaPhis
//        print_vector(gradient, Ktotal, "grad_S10");



        if (t == 0) {
            if (use_identity) {
                /**
                 * Here we update the gradient to incorporate identity (shape) constraints
                 * See Step II of Figure #43 of large notebook
                 *
                 * Since identity constraints are common for all frames, this is not a frame-dependent operation
                 * and is done only once (hence the t==0)
                 */
                update_gradient<<<(Ktotal_base+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_alpha_lb, 	f_alpha_ub, 	gradient, ov->Kalpha, 	0);
            }
        }

        if (use_expression) {
            // <!-- focus here -- update gradient -->
            /**
             * Here we update the gradient to incorporate identity (shape) constraints
             * See Step III of Figure #43 of large notebook
             */
            update_gradient<<<(Ktotal_base+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_epsilon_lb, 	f_epsilon_ub, 	gradient, ov->Kepsilon, 	offset + t*ov->Kepsilon*use_expression);
//            print_vector(gradient, Ktotal, "grad_S11");

            // <!-- focus here -- update gradient -->
            if (use_exp_regularization && !use_slack)
            {
                float *tmp_exp;
                HANDLE_ERROR( cudaMalloc( (void**)&tmp_exp, sizeof(float)*ov->Kepsilon ) );
                cudaMemcpy(tmp_exp, eps_l2weights_x2, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);
                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, ov->epsilons, ov->Kepsilon);
                cublasSaxpy(handle, ov->Kepsilon, &plus_one, tmp_exp, 1, gradient+offset+t*ov->Kepsilon, 1);
                cudaFree( tmp_exp );
            }

            if (T>1 && use_temp_smoothing && !use_slack)
            {
                //! (1) Expression smoothing
                float *exp_diff1, *exp_diff2;

                HANDLE_ERROR( cudaMalloc( (void**)&exp_diff1, sizeof(float)*ov->Kepsilon ) );
                HANDLE_ERROR( cudaMalloc( (void**)&exp_diff2, sizeof(float)*ov->Kepsilon ) );

                cudaMemcpy(exp_diff1, ov->epsilons, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(exp_diff2, ov->epsilons, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);

                if (t > 0) {
                    cublasSaxpy(handle, ov->Kepsilon, &minus_one, ov->epsilons-ov->Kepsilon, 1, exp_diff1, 1);
                    elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(exp_diff1, deps_l2weights_x2, ov->Kepsilon);
                    cublasSaxpy(handle, ov->Kepsilon, &plus_one, exp_diff1, 1, gradient+offset+t*ov->Kepsilon, 1);
                }

                if (t != T-1) {
                    cublasSaxpy(handle, ov->Kepsilon, &minus_one, ov->epsilons+ov->Kepsilon, 1, exp_diff2, 1);
                    elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(exp_diff2, deps_l2weights_x2, ov->Kepsilon);
                    cublasSaxpy(handle, ov->Kepsilon, &plus_one, exp_diff2, 1, gradient+offset+t*ov->Kepsilon, 1);
                }
//                print_vector(tmp_exp, 2*ov->Kepsilon, "tmp_exp");
                cudaFree( exp_diff1 );
                cudaFree( exp_diff2 );

                //! (2) Pose smoothing
                float *pose_diff1, *pose_diff2;

                HANDLE_ERROR( cudaMalloc( (void**)&pose_diff1, sizeof(float)*6 ) );
                HANDLE_ERROR( cudaMalloc( (void**)&pose_diff2, sizeof(float)*6 ) );

                cudaMemcpy(pose_diff1, ov->taux, 6*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(pose_diff2, ov->taux, 6*sizeof(float), cudaMemcpyDeviceToDevice);

                if (t > 0) {
                    cublasSaxpy(handle, 6, &minus_one, ov->taux-6, 1, pose_diff1, 1);
                    elementwise_vector_multiplication<<<(6+NTHREADS-1)/NTHREADS, NTHREADS>>>(pose_diff1, drigid_l2weights_x2, 6);
                    cublasSaxpy(handle, 6, &plus_one, pose_diff1, 1, gradient+offset+T*ov->Kepsilon+t*6, 1);
                }

                if (t != T-1) {
                    cublasSaxpy(handle, 6, &minus_one, ov->taux+6, 1, pose_diff2, 1);
                    elementwise_vector_multiplication<<<(6+NTHREADS-1)/NTHREADS, NTHREADS>>>(pose_diff2, drigid_l2weights_x2, 6);
                    cublasSaxpy(handle, 6, &plus_one, pose_diff2, 1, gradient+offset+T*ov->Kepsilon+t*6, 1);
                }

                cudaFree( pose_diff1 );
                cudaFree( pose_diff2 );
                /*
*/
            }
        }



        if (use_expression)
        {
            // <!-- focus here -- update pseudo-hessian -->
            update_diagonal_of_hessian_wbounds<<<1, ov->Kepsilon>>>(f_epsilon_lb, f_epsilon_ub, nabla2F, ov->Kepsilon, Ktotal, offset + t*ov->Kepsilon*use_expression);

            if (use_exp_regularization && !use_slack) {
                update_diagonal_of_hessian_wvector<<<1, ov->Kepsilon>>>(eps_l2weights_x2, nabla2F, ov->Kepsilon, Ktotal, offset + t*ov->Kepsilon*use_expression);
            }

            //! Important -- we take care of the Hessian of expression-differential component in the end of this function
        }

        if (use_slack) {

            /**
             * Here we update the last component of the gradient
             *
             * In MATLAB language, we're doing:
             * 	nablaPhis = -(1./Gx_plus + 1./Gy_plus + 1./Gx_minus + 1./Gy_minus);
             * 	gradient(end) = gradient(end) + sum(nablaPhis)
             */
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 4*NLANDMARKS_51, &minus_one, vecOnes, 1, inv_Gx_minus_tmp, NLANDMARKS_51*4, &plus_one, gradient+Ktotal-1, 1);

            /**
             * Here we update the component of the hessian that is correspond to the second derivative
             * \frac{\partial^2 f_0}{\partial \theta \partial s}
             *
             * (See Step IV of Figure#45 in large notebook#1).
             */
            fill_for_nabla2F_dsdc<<<(NLANDMARKS_51*Ktotal_base+NTHREADS-1)/NTHREADS, NTHREADS>>>(nablaWx, nablaWy, Gx_minus_tmp, Gy_minus_tmp, Gx_plus_tmp, Gy_plus_tmp, Ktotal_base,  for_nabla2F_dsdc);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, Ktotal_base, NLANDMARKS_51, &plus_one, vecOnes, 1, for_nabla2F_dsdc, NLANDMARKS_51, &plus_one, nabla2F_dsdc, 1);
            fill_bottom_and_right_of_nabla2F<<<(Ktotal_base+NTHREADS-1)/NTHREADS, NTHREADS>>>(nabla2F_dsdc, nabla2F, Ktotal, Ktotal_base);


            /**
             * Here we update the last element of the hessian, which corresponds to  \frac{\partial^2 f_0}{\partial s^2}
             *
             * In MATLAB language, this is:
             * 	nabla2F_ds2 = (1./((Gx_plus).^2)+mineps) + (1./((Gx_minus).^2)+mineps)  + (1./((Gy_plus).^2)+mineps) + (1./((Gy_minus).^2)+mineps);
             */
            float *tmp_for_ds2;
            HANDLE_ERROR( cudaMalloc( (void**)&tmp_for_ds2, sizeof(float) ) );
            cublasSdot(handle, 4*NLANDMARKS_51, inv_Gx_minus_tmp, 1, inv_Gx_minus_tmp, 1, tmp_for_ds2); // last element of nabla2F
            cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 	 &plus_one, tmp_for_ds2, 1, &plus_one, nabla2F+Ktotal*Ktotal-1, 1, nabla2F+Ktotal*Ktotal-1, 1);
            HANDLE_ERROR( cudaFree( tmp_for_ds2 )  );
        }

        if (t == 0) {
            if (use_identity) {
                /**
                 * Here we update the hessian to incorporate identity (shape) constraints
                 * We simply update the diagonal of the (leading submatrix) of the hessian
                 * (See Step II of Figure #45 in large notebook#1).
                 */
                update_diagonal_of_hessian_wbounds<<<1, ov->Kalpha>>>(f_alpha_lb, f_alpha_ub, nabla2F, ov->Kalpha, Ktotal, 0);
            }
        }

        if (t == 0)
        {
            if (use_identity)
            {
                /**
                 * Here we update the objective -- we incorporate the identity (shape) coefficients
                 *                    ---------
                 *
                 * We start by neglogifying (i.e., computing the negative of the logarithm) of
                 * the f_alpha_lb and f_alpha_ub variables, which we don't need (in raw form) anymore
                 */
                neglogify<<<(ov->Kalpha+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_alpha_lb, ov->Kalpha);
                neglogify<<<(ov->Kalpha+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_alpha_ub, ov->Kalpha);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kalpha, &plus_one, vecOnes, 1, f_alpha_lb, ov->Kalpha, &plus_one, obj, 1);
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kalpha, &plus_one, vecOnes, 1, f_alpha_ub, ov->Kalpha, &plus_one, obj, 1);

            }
        }

        if (use_expression)
        {
            /**
             * Here we update the objective -- we incorporate the expression coefficients
             *                    ---------
             *
             * We start by neglogifying (i.e., computing the negative of the logarithm) of
             * the f_epsilon_lb and f_epsilon_ub variables, which we don't need (in raw form) anymore
             */
            neglogify<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_epsilon_lb, ov->Kepsilon);
            neglogify<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_epsilon_ub, ov->Kepsilon);

            // this is for objective
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kepsilon, &plus_one, vecOnes, 1, f_epsilon_lb, ov->Kepsilon, &plus_one, obj, 1);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kepsilon, &plus_one, vecOnes, 1, f_epsilon_ub, ov->Kepsilon, &plus_one, obj, 1);

            if (use_exp_regularization && !use_slack)
            {
                // <!-- focus here -- update objective -->
                float *tmp_exp;
                HANDLE_ERROR( cudaMalloc( (void**)&tmp_exp, sizeof(float)*ov->Kepsilon ) );

                cudaMemcpy(tmp_exp, eps_l2weights, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);

                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, ov->epsilons, ov->Kepsilon);
                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, ov->epsilons, ov->Kepsilon);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kepsilon, &plus_one, vecOnes, 1, tmp_exp, ov->Kepsilon, &plus_one, obj, 1);

                cudaFree( tmp_exp );
            }

            if (t >= 1 && T > 1 && use_temp_smoothing && !use_slack)
            {
                //! (1) Expression smoothing
                float *tmp_exp, *exp_diff;
                HANDLE_ERROR( cudaMalloc( (void**)&tmp_exp, sizeof(float)*ov->Kepsilon ) );
                HANDLE_ERROR( cudaMalloc( (void**)&exp_diff, sizeof(float)*ov->Kepsilon ) );
                cudaMemcpy(tmp_exp, deps_l2weights, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(exp_diff, ov->epsilons, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);
                cublasSaxpy(handle, ov->Kepsilon, &minus_one, ov->epsilons-ov->Kepsilon, 1, exp_diff, 1);

                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, exp_diff, ov->Kepsilon);
                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, exp_diff, ov->Kepsilon);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kepsilon, &plus_one, vecOnes, 1, tmp_exp, ov->Kepsilon, &plus_one, obj, 1);

                cudaFree( tmp_exp );
                cudaFree( exp_diff );

                //! (2) Pose smoothing
                float *tmp_pose, *pose_diff;
                HANDLE_ERROR( cudaMalloc( (void**)&tmp_pose, sizeof(float)*6 ) );
                HANDLE_ERROR( cudaMalloc( (void**)&pose_diff, sizeof(float)*6 ) );
                cudaMemcpy(tmp_pose, drigid_l2weights, 6*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(pose_diff, ov->taux, 6*sizeof(float), cudaMemcpyDeviceToDevice);
                cublasSaxpy(handle, 6, &minus_one, ov->taux-6, 1, pose_diff, 1);

                elementwise_vector_multiplication<<<(6+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_pose, pose_diff, 6);
                elementwise_vector_multiplication<<<(6+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_pose, pose_diff, 6);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 6, &plus_one, vecOnes, 1, tmp_pose, 6, &plus_one, obj, 1);

                cudaFree( tmp_pose );
                cudaFree( pose_diff );
                /*
                */
            }






            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /*
            {
                cv::Mat emptyFrame(cams_ptr->at(t).h_cy*2, cams_ptr->at(t).h_cx*2, CV_32FC3, cv::Scalar::all(255));

                float h_xl[NLANDMARKS_51], h_yl[NLANDMARKS_51];
                float h_xproj[NLANDMARKS_51], h_yproj[NLANDMARKS_51];

                float *d_xproj, *d_yproj;
                cudaMalloc((void**)&d_xproj, sizeof(float)*NLANDMARKS_51);
                cudaMalloc((void**)&d_yproj, sizeof(float)*NLANDMARKS_51);

                HANDLE_ERROR( cudaMemcpy( h_xl, xl+t*NLANDMARKS_51, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost ) );
                HANDLE_ERROR( cudaMemcpy( h_yl, yl+t*NLANDMARKS_51, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost ) );



                compute_xyproj<<<1, NLANDMARKS_51>>>(ov->taux, cams_ptr->at(t).h_phix, cams_ptr->at(t).h_phiy, cams_ptr->at(t).h_cx, cams_ptr->at(t).h_cy,
                        Rp, xl+t*NLANDMARKS_51, yl+t*NLANDMARKS_51,
                        d_xproj, d_yproj);

                cudaMemcpy(h_xproj, d_xproj, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_yproj, d_yproj, sizeof(float)*NLANDMARKS_51, cudaMemcpyDeviceToHost);

                for (uint ui=0; ui<NLANDMARKS_51; ++ui) {
                    cv::Point2f ptd(h_xl[ui], h_yl[ui]);
                    cv::Point2f ptp(h_xproj[ui], h_yproj[ui]);
                    cv::circle(emptyFrame, ptd, 3, cv::Scalar(0,0,255), cv::FILLED, 8, 0);
                    cv::circle(emptyFrame, ptp, 3, cv::Scalar(255,128,0), cv::FILLED, 8, 0);
                }

                cudaFree(d_xproj);
                cudaFree(d_yproj);

                cv::imshow("emptyFrame", emptyFrame);
                cv::waitKey(0);
            }
            */
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////
            /////////////////////////////////////////v
        }


        /**
         * Here we update the objective -- we incorporate the landmark constraints
         */
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 4*NLANDMARKS_51, &plus_one, vecOnes, 1, nlog_Gx_minus, 4*NLANDMARKS_51, &plus_one, obj, 1);

    }







    // update pseudo-hessian for expression differential component
    if (T > 1 && use_temp_smoothing && !use_slack)
    {
        //! (1) Expression smoothing
        int K_except_offset = Ktotal-ov->Kalpha;
        float *Hde; // part of Hessian corresponding to expression differential
        HANDLE_ERROR( cudaMalloc( (void**)&Hde, K_except_offset*K_except_offset*sizeof(float) ) );

        cudaMemset(Hde, 0, K_except_offset*K_except_offset*sizeof(float));

        update_diags_and_offdiags_for_expdiffs<<<(T*ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(Hde, T, ov->Kepsilon, K_except_offset, deps_l2weights);
        update_bottom_right_of_matrix<<<(K_except_offset*K_except_offset+NTHREADS-1)/NTHREADS, NTHREADS>>>(Hde, nabla2F, K_except_offset, offset);

        cudaFree( Hde );

        //! (2) Pose smoothing
        int K_except_pose = Ktotal-(ov->Kalpha+T*ov->Kepsilon);
        float *Hdp; // part of Hessian corresponding to pose differential
        HANDLE_ERROR( cudaMalloc( (void**)&Hdp, K_except_pose*K_except_pose*sizeof(float) ) );

        cudaMemset(Hdp, 0, K_except_pose*K_except_pose*sizeof(float));
        update_diags_and_offdiags_for_expdiffs<<<(T*6+NTHREADS-1)/NTHREADS, NTHREADS>>>(Hdp, T, 6, K_except_pose, drigid_l2weights);
        update_bottom_right_of_matrix<<<(K_except_pose*K_except_pose+NTHREADS-1)/NTHREADS, NTHREADS>>>(Hdp, nabla2F, K_except_pose, offset+T*ov->Kepsilon);
        cudaFree( Hdp );
        /*
        */
    }




    if (use_slack)
    {
        // this is for slack
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 	 &plus_one, ov->tau_logbarrier, 1, &plus_one, gradient+Ktotal-1, 1, gradient+Ktotal-1, 1);
    }
}



void Logbarrier_Initializer::copy_from_initialized(Logbarrier_Initializer& li_initialized)
{
    cudaMemcpy(xmeans, li_initialized.xmeans, T*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ymeans, li_initialized.ymeans, T*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(face_sizes, li_initialized.face_sizes, T*sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(angle_idx, li_initialized.angle_idx, T*sizeof(uint), cudaMemcpyHostToHost);

    cudaMemcpy(bounds_lx_cur, li_initialized.bounds_lx_cur, T*NLANDMARKS_51*sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(bounds_ux_cur, li_initialized.bounds_ux_cur, T*NLANDMARKS_51*sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(bounds_ly_cur, li_initialized.bounds_ly_cur, T*NLANDMARKS_51*sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(bounds_uy_cur, li_initialized.bounds_uy_cur, T*NLANDMARKS_51*sizeof(float), cudaMemcpyHostToHost);
    /*
    */
}


void Logbarrier_Initializer::initialize_with_orthographic_t(
        cusolverDnHandle_t& handleDn, cublasHandle_t &handle, uint t,
        const float *h_xl_t, const float *h_yl_t, const float h_face_size,
        OptimizationVariables* ov)
{
    float yaw, pitch, roll;
    oi.fit_model(handleDn, handle, h_xl_t, h_yl_t, &yaw, &pitch, &roll);

    float h_xorth[7];

    cudaMemcpy(h_xorth, oi.d_x_orthographic, 7*sizeof(float), cudaMemcpyDeviceToHost);

    if (h_xorth[2] < 0) {
        //        print_vector(oi.d_x_orthographic, 7, "orthvars_before");
        cv::Vec3f evec(DEG2RAD(roll), DEG2RAD(-pitch), DEG2RAD(-yaw));
        cv::Mat uout;
        cv::Vec3f h_u0(h_xorth[4], h_xorth[5], h_xorth[6]);

        cv::Mat Ru0;

        //    print_matrix(rc.R, 3,3);
        cv::Rodrigues(eulerAnglesToRotationMatrix(evec), h_u0);
        cv::Rodrigues(h_u0, Ru0);
        //    std::cout << Ru0 << std::endl;

        h_xorth[0] = -h_xorth[0];
        h_xorth[2] = -h_xorth[2];
        h_xorth[4] = h_u0[0];
        h_xorth[5] = h_u0[1];
        h_xorth[6] = h_u0[2];

        cudaMemcpy(oi.d_x_orthographic, h_xorth, 7*sizeof(float), cudaMemcpyHostToDevice);
        oi.fit_model(handleDn, handle, h_xl_t, h_yl_t, &yaw, &pitch, &roll, false);
        //        print_vector(oi.d_x_orthographic, 7, "orthvars_after");
    }


    compute_bounds(t, yaw, pitch, roll);

    float h_xmean_t = thrust::reduce(h_xl_t, h_xl_t+NLANDMARKS_51, 0.0f, thrust::plus<float>())/(float)NLANDMARKS_51;
    float h_ymean_t = thrust::reduce(h_yl_t, h_yl_t+NLANDMARKS_51, 0.0f, thrust::plus<float>())/(float)NLANDMARKS_51;
    cudaMemcpy(xmeans+t, &h_xmean_t, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ymeans+t, &h_ymean_t, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(face_sizes+t, &h_face_size, sizeof(float), cudaMemcpyHostToHost);

    ov->set_frame(t);

    initialize_kernel<<<1,1>>>(xmeans+t, ymeans+t,
                               cams_ptr->at(t).phix, cams_ptr->at(t).phiy, cams_ptr->at(t).cx, cams_ptr->at(t).cy,
                               oi.d_x_orthographic+2, oi.d_x_orthographic+3, oi.d_x_orthographic+4,
                               ov->taux, ov->tauy, ov->tauz, ov->u);

}






void Logbarrier_Initializer::evaluate_objective_function(cublasHandle_t &handle, OptimizationVariables* ov, float *obj)
{
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    float minus_one = -1.0f;

    cudaMemset(obj, 0, sizeof(float));

    if (!use_slack) {
        cudaMemset(ov->slack, 0, sizeof(float));
    } else {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha_, ov->tau_logbarrier, 1, ov->slack, 1, &beta_, obj, 1);
    }


    if (use_identity)
    {
        //     f_alpha_ub = -data.alpha_ub(1:K)+alpha';
        cudaMemcpy(f_alpha_ub, ov->alphas, sizeof(float)*ov->Kalpha, cudaMemcpyDeviceToDevice);
        cublasSaxpy(handle, ov->Kalpha, &minus_one, alpha_ub, 1, f_alpha_ub, 1);

        //     f_alpha_lb = -alpha'+data.alpha_lb(1:K);
        cudaMemcpy(f_alpha_lb, alpha_lb,  sizeof(float)*ov->Kalpha, cudaMemcpyDeviceToDevice);
        cublasSaxpy(handle, ov->Kalpha, &minus_one, ov->alphas, 1, f_alpha_lb, 1);
    }

    if (ov->Kbeta > 0)
    {
        cudaMemcpy(f_beta_ub, ov->betas, sizeof(float)*ov->Kbeta, cudaMemcpyDeviceToDevice);
        cublasSaxpy(handle, ov->Kbeta, &minus_one, beta_ub, 1, f_beta_ub, 1);

        cudaMemcpy(f_beta_lb, beta_lb,  sizeof(float)*ov->Kbeta, cudaMemcpyDeviceToDevice);
        cublasSaxpy(handle, ov->Kbeta, &minus_one, ov->betas, 1, f_beta_lb, 1);
    }

    uint length_vecOnes = max( ov->Kalpha, max(ov->Kepsilon, NLANDMARKS_51*4));

    for (uint t=0; t<T; ++t)
    {
        uint offset = ov->Kalpha*use_identity;

        uint epsilon_offset = (offset + t*ov->Kepsilon*use_expression)*NLANDMARKS_51;
        uint c_offset = (offset + T*ov->Kepsilon*use_expression + 6*t)*NLANDMARKS_51;

        ov->set_frame(t);
        rc.set_u_ptr(ov->u);
        rc.process_without_derivatives();
        compute_nonrigid_shape(handle, ov);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, NLANDMARKS_51, 3, &alpha_, rc.R, 3, p, 3, &beta_, Rp, 3);

        //        int bound_offset = NLANDMARKS_51*angle_idx[t];
        int bound_offset = NLANDMARKS_51*t;

        compute_Gs<<<1, NLANDMARKS_51>>>(ov->taux, cams_ptr->at(t).h_phix, cams_ptr->at(t).h_phiy, cams_ptr->at(t).h_cx, cams_ptr->at(t).h_cy,
                                         face_sizes[t], Rp,
                                         bounds_lx_cur + bound_offset, bounds_ly_cur + bound_offset, bounds_ux_cur + bound_offset, bounds_uy_cur + bound_offset,
                                         xl+t*NLANDMARKS_51, yl+t*NLANDMARKS_51,
                                         Gx_minus_tmp, Gy_minus_tmp, Gx_plus_tmp, Gy_plus_tmp,
                                         inv_Gx_minus_tmp, inv_Gy_minus_tmp, inv_Gx_plus_tmp, inv_Gy_plus_tmp,
                                         nlog_Gx_minus, nlog_Gy_minus, nlog_Gx_plus, nlog_Gy_plus,
                                         ov->slack);

        if (use_expression)
        {
            //     f_epsilon_ub = epsilon'-boundcoef*data.epsilon_ub(1:length(epsilon));
            cudaMemcpy(f_epsilon_ub, ov->epsilons, sizeof(float)*ov->Kepsilon, cudaMemcpyDeviceToDevice);
            cublasSaxpy(handle, ov->Kepsilon, &minus_one, epsilon_ub, 1, f_epsilon_ub, 1);

            //     f_epsilon_lb = boundcoef*data.epsilon_lb(1:length(epsilon))-epsilon';
            cudaMemcpy(f_epsilon_lb, epsilon_lb,  sizeof(float)*ov->Kepsilon, cudaMemcpyDeviceToDevice);
            cublasSaxpy(handle, ov->Kepsilon, &minus_one, ov->epsilons, 1, f_epsilon_lb, 1);
        }


        if (t == 0)
        {
            if (use_identity)
            {
                neglogify<<<(ov->Kalpha+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_alpha_lb, ov->Kalpha);
                neglogify<<<(ov->Kalpha+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_alpha_ub, ov->Kalpha);
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kalpha, &alpha_, vecOnes, 1, f_alpha_lb, ov->Kalpha, &alpha_, obj, 1);
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kalpha, &alpha_, vecOnes, 1, f_alpha_ub, ov->Kalpha, &alpha_, obj, 1);
            }
        }


        if (use_expression)
        {
            neglogify<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_epsilon_lb, ov->Kepsilon);
            neglogify<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(f_epsilon_ub, ov->Kepsilon);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kepsilon, &alpha_, vecOnes, 1, f_epsilon_lb, ov->Kepsilon, &alpha_, obj, 1);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kepsilon, &alpha_, vecOnes, 1, f_epsilon_ub, ov->Kepsilon, &alpha_, obj, 1);



            if (t >= 1 && T > 1 && use_temp_smoothing && !use_slack)
            {
                //! (1) Expression smoothing
                float plus_one = 1.0f;

                float *tmp_exp, *exp_diff;
                HANDLE_ERROR( cudaMalloc( (void**)&tmp_exp, sizeof(float)*ov->Kepsilon ) );
                HANDLE_ERROR( cudaMalloc( (void**)&exp_diff, sizeof(float)*ov->Kepsilon ) );
                cudaMemcpy(tmp_exp, deps_l2weights, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(exp_diff, ov->epsilons, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);
                cublasSaxpy(handle, ov->Kepsilon, &minus_one, ov->epsilons-ov->Kepsilon, 1, exp_diff, 1);

                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, exp_diff, ov->Kepsilon);
                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, exp_diff, ov->Kepsilon);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kepsilon, &plus_one, vecOnes, 1, tmp_exp, ov->Kepsilon, &plus_one, obj, 1);


                cudaFree( tmp_exp );
                cudaFree( exp_diff );


                //! (2) Pose smoothing
                float *tmp_pose, *pose_diff;
                HANDLE_ERROR( cudaMalloc( (void**)&tmp_pose, sizeof(float)*6 ) );
                HANDLE_ERROR( cudaMalloc( (void**)&pose_diff, sizeof(float)*6 ) );
                cudaMemcpy(tmp_pose, drigid_l2weights, 6*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(pose_diff, ov->taux, 6*sizeof(float), cudaMemcpyDeviceToDevice);
                cublasSaxpy(handle, 6, &minus_one, ov->taux-6, 1, pose_diff, 1);

                elementwise_vector_multiplication<<<(6+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_pose, pose_diff, 6);
                elementwise_vector_multiplication<<<(6+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_pose, pose_diff, 6);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 6, &plus_one, vecOnes, 1, tmp_pose, 6, &plus_one, obj, 1);

                cudaFree( tmp_pose );
                cudaFree( pose_diff );
            }


            if (use_exp_regularization && !use_slack)
            {
                float plus_one = 1.0f;
                // <!-- focus here -- update objective -->
                float *tmp_exp;
                HANDLE_ERROR( cudaMalloc( (void**)&tmp_exp, sizeof(float)*ov->Kepsilon ) );

                cudaMemcpy(tmp_exp, eps_l2weights, ov->Kepsilon*sizeof(float), cudaMemcpyDeviceToDevice);

                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, ov->epsilons, ov->Kepsilon);
                elementwise_vector_multiplication<<<(ov->Kepsilon+NTHREADS-1)/NTHREADS, NTHREADS>>>(tmp_exp, ov->epsilons, ov->Kepsilon);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, ov->Kepsilon, &plus_one, vecOnes, 1, tmp_exp, ov->Kepsilon, &plus_one, obj, 1);

                cudaFree( tmp_exp );
            }

        }

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 4*NLANDMARKS_51, &alpha_, vecOnes, 1, nlog_Gx_minus, 4*NLANDMARKS_51, &alpha_, obj, 1);

        //		print_vector(obj, 1, std::string("_OBJ_Stage4")+std::to_string(t));
    }
}














void Logbarrier_Initializer::set_bounds_from_host(
        const float *h_bounds_lx,
        const float *h_bounds_ly,
        const float *h_bounds_ux,
        const float *h_bounds_uy)
{
    HANDLE_ERROR( cudaMemcpy( bounds_lx, h_bounds_lx, sizeof(float)*NLANDMARKS_51*N_ANGLE_COMBINATIONS, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( bounds_ly, h_bounds_ly, sizeof(float)*NLANDMARKS_51*N_ANGLE_COMBINATIONS, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( bounds_ux, h_bounds_ux, sizeof(float)*NLANDMARKS_51*N_ANGLE_COMBINATIONS, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( bounds_uy, h_bounds_uy, sizeof(float)*NLANDMARKS_51*N_ANGLE_COMBINATIONS, cudaMemcpyHostToDevice ) );
}



void Logbarrier_Initializer::set_landmarks_from_host(uint t, const float *h_xl_t, const float *h_yl_t)
{
    HANDLE_ERROR( cudaMemcpy( xl+t*NLANDMARKS_51, h_xl_t, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( yl+t*NLANDMARKS_51, h_yl_t, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice ) );
}


void Logbarrier_Initializer::set_dictionaries_from_host(const float *h_AL, const float *h_EL, const OptimizationVariables *ov)
{
    HANDLE_ERROR( cudaMemcpy( AL, h_AL, sizeof(float)*NLANDMARKS_51*3*ov->Kalpha, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( EL, h_EL, sizeof(float)*NLANDMARKS_51*3*ov->Kepsilon, cudaMemcpyHostToDevice ) );
}


void Logbarrier_Initializer::compute_nonrigid_shape(cublasHandle_t &handle, const OptimizationVariables *ov, bool identity, bool expression)
{
    cudaMemcpy(p, p0L_mat, sizeof(float)*NLANDMARKS_51*3, cudaMemcpyDeviceToDevice);

    float alpha = 1.f;
    float beta  = 1.f;

    if (identity) {
        cublasSgemv(handle, CUBLAS_OP_N, NLANDMARKS_51*3, ov->Kalpha, &alpha, AL, NLANDMARKS_51*3, ov->alphas, 1, &beta, p, 1);
    }

    if (expression) {
        cublasSgemv(handle, CUBLAS_OP_N, NLANDMARKS_51*3, ov->Kepsilon, &alpha, EL, NLANDMARKS_51*3, ov->epsilons, 1, &beta, p, 1);
    }
}


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void Logbarrier_Initializer::compute_bounds(uint t, float yaw, float pitch, float roll,bool skip_this_frame)
{
    const float YAW_MAX = 40.0f;
    const float PITCH_MAX = 90.0f;
    const float ROLL_MAX = 42.5f;

    // We normalize input to nnet by dividing to YAW_MAX etc.
    // Before, we also clamp the values to lie in [-(YAW_MAX-4), (YAW_MAX-4)] etc.
    // We don't clamp exactly at the edges because at the edges the nnet may perform a bit weirdly
    if (std::abs(yaw) > YAW_MAX-4.0f)
        yaw = sgn<float>(yaw)*(YAW_MAX-4.0f);

    if (std::abs(pitch) > PITCH_MAX-4.0f)
        pitch = sgn<float>(pitch)*(PITCH_MAX-4.0f);

    if (std::abs(roll) > ROLL_MAX-4.0f)
        roll = sgn<float>(roll)*(ROLL_MAX-4.0f);

    yaw /= YAW_MAX;
    pitch /= PITCH_MAX;
    roll /= ROLL_MAX;


    /*
    // This is what's supposed to be but temporarily off (trying with ranges, see below

    float data_angles[3] = {yaw, pitch, roll};

    cv::Mat input_angles( 1, 3, CV_32FC1, data_angles);

    bound_estimator.setInput(cv::dnn::blobFromImage(input_angles));
    cv::Mat netOut = bound_estimator.forward();

    float *data_ptr = (float*) netOut.data;


    for (uint i=0; i<NLANDMARKS_51*4; ++i) {
        data_ptr[i] = data_ptr[i]*(bounds_ymax[i]-bounds_ymin[i])+bounds_ymin[i];
//        std::cout << data_ptr[i] << std::endl;
    }

    cudaMemcpy(bounds_lx_cur+t*NLANDMARKS_51, data_ptr, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice);
    cudaMemcpy(bounds_ux_cur+t*NLANDMARKS_51, data_ptr+NLANDMARKS_51, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice);
    cudaMemcpy(bounds_ly_cur+t*NLANDMARKS_51, data_ptr+2*NLANDMARKS_51, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice);
    cudaMemcpy(bounds_uy_cur+t*NLANDMARKS_51, data_ptr+3*NLANDMARKS_51, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice);

    */


    float h_bounds_lx[NLANDMARKS_51], h_bounds_ux[NLANDMARKS_51], h_bounds_ly[NLANDMARKS_51], h_bounds_uy[NLANDMARKS_51];
    for (uint i=0; i<NLANDMARKS_51; ++i)
    {
        float data_anglesx[4] = {yaw, pitch, roll, CONFIDENCE}; // 0.25 was good but probably too strict // 0.35 was also good and less strict
        float data_anglesy[4] = {yaw, pitch, roll, CONFIDENCE}; // 0.25 was good but probably too strict // 0.35 was also good and less strict
        cv::Mat input_anglesx( 1, 4, CV_32FC1, data_anglesx);
        cv::Mat input_anglesy( 1, 4, CV_32FC1, data_anglesy);
        vec_bound_estimator[i].setInput(cv::dnn::blobFromImage(input_anglesx));
        cv::Mat netOut = vec_bound_estimator[i].forward();

        float margin = -1.0f;

        if (config::USE_LOCAL_MODELS)
            margin = 0.175f;
        else
            margin = 0.03f;

        if (config::IGNORE_NOSE)
        {
            // this is uncommented if we want to ignore the nose
            if (10<=i && i<=18)
                margin = 100.0f;
        }

        if (config::IGNORE_SOME_LANDMARKS)
        {
            if (i == 0 || i == 1 || i == 8 || i == 9)
                margin = 1.50f;

            // this is uncommented if we want to ignore the nose
            if (10<=i && i<=18)
                margin = 0.40f;
        }

        float *data_ptr = (float*) netOut.data;

        if (_IGNORE_BOUNDS == 0 && !skip_this_frame) {
            h_bounds_lx[i] = (data_ptr[0]*(vec_bounds_ymax[i][0]-vec_bounds_ymin[i][0])+vec_bounds_ymin[i][0]);
            h_bounds_ux[i] = (data_ptr[1]*(vec_bounds_ymax[i][1]-vec_bounds_ymin[i][1])+vec_bounds_ymin[i][1]);

            h_bounds_ly[i] = (data_ptr[2]*(vec_bounds_ymax[i][2]-vec_bounds_ymin[i][2])+vec_bounds_ymin[i][2]);
            h_bounds_uy[i] = (data_ptr[3]*(vec_bounds_ymax[i][3]-vec_bounds_ymin[i][3])+vec_bounds_ymin[i][3]);

            float dx = h_bounds_ux[i]-h_bounds_lx[i];
            float dy = h_bounds_uy[i]-h_bounds_ly[i];

            if (margin != -1.0f) {
                h_bounds_lx[i] -= margin*dx;
                h_bounds_ly[i] -= margin*dy;
                h_bounds_ux[i] += margin*dx;
                h_bounds_uy[i] += margin*dy;
            }
        } else if (_IGNORE_BOUNDS || skip_this_frame) {
            h_bounds_lx[i] = -1000.0f;
            h_bounds_ux[i] = 1000.0f;
            h_bounds_ly[i] = -1000.0f;
            h_bounds_uy[i] = 1000.0f;
        }
    }

    cudaMemcpy(bounds_lx_cur+t*NLANDMARKS_51, h_bounds_lx, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice);
    cudaMemcpy(bounds_ux_cur+t*NLANDMARKS_51, h_bounds_ux, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice);
    cudaMemcpy(bounds_ly_cur+t*NLANDMARKS_51, h_bounds_ly, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice);
    cudaMemcpy(bounds_uy_cur+t*NLANDMARKS_51, h_bounds_uy, sizeof(float)*NLANDMARKS_51, cudaMemcpyHostToDevice);
}



Logbarrier_Initializer::~Logbarrier_Initializer()
{
    free( face_sizes );
    free( angle_idx );

    for (size_t i=0; i<vec_bounds_ymin.size(); ++i) {
        free(vec_bounds_ymin[i]);
        free(vec_bounds_ymax[i]);
    }

    HANDLE_ERROR( cudaFree( xmeans ) );
    HANDLE_ERROR( cudaFree( ymeans ) );
    HANDLE_ERROR( cudaFree( nabla2F ) );
    HANDLE_ERROR( cudaFree( gradient ) );



    if (f_beta_lb != NULL)
        HANDLE_ERROR( cudaFree( f_beta_lb ) );
    if (beta_lb != NULL)
        HANDLE_ERROR( cudaFree( beta_lb ) );
    if (beta_ub != NULL)
        HANDLE_ERROR( cudaFree( beta_ub ) );


    HANDLE_ERROR( cudaFree( f_alpha_lb ) );
    HANDLE_ERROR( cudaFree( p ) );

    HANDLE_ERROR( cudaFree( p0L_mat ) );

    HANDLE_ERROR( cudaFree( Rp ) );

    HANDLE_ERROR( cudaFree( AL ) );
    HANDLE_ERROR( cudaFree( EL ) );

    HANDLE_ERROR( cudaFree( xl ) );
    HANDLE_ERROR( cudaFree( yl ) );

    HANDLE_ERROR( cudaFree( bounds_ux ) );
    HANDLE_ERROR( cudaFree( bounds_uy ) );
    HANDLE_ERROR( cudaFree( bounds_lx ) );
    HANDLE_ERROR( cudaFree( bounds_ly ) );


    HANDLE_ERROR( cudaFree( bounds_ux_cur ) );
    HANDLE_ERROR( cudaFree( bounds_uy_cur ) );
    HANDLE_ERROR( cudaFree( bounds_lx_cur ) );
    HANDLE_ERROR( cudaFree( bounds_ly_cur ) );

    HANDLE_ERROR( cudaFree( Gxs_ALL ) );


    HANDLE_ERROR( cudaFree( alpha_lb ) );
    HANDLE_ERROR( cudaFree( alpha_ub ) );
    HANDLE_ERROR( cudaFree( epsilon_lb ) );
    HANDLE_ERROR( cudaFree( epsilon_ub ) );
    HANDLE_ERROR( cudaFree( epsilon_lb_finetune ) );
    HANDLE_ERROR( cudaFree( epsilon_ub_finetune ) );
    HANDLE_ERROR( cudaFree( epsilon_lb_regular ) );
    HANDLE_ERROR( cudaFree( epsilon_ub_regular ) );

    HANDLE_ERROR( cudaFree( eps_l2weights ) );
    HANDLE_ERROR( cudaFree( eps_l2weights_x2 ) );

    HANDLE_ERROR( cudaFree( deps_l2weights ) );
    HANDLE_ERROR( cudaFree( deps_l2weights_x2 ) );

    HANDLE_ERROR( cudaFree( drigid_l2weights ) );
    HANDLE_ERROR( cudaFree( drigid_l2weights_x2 ) );





    HANDLE_ERROR( cudaFree( for_nablaPhi_Gx_minus ) );
    HANDLE_ERROR( cudaFree( for_nablaPhi_Gy_minus ) );
    HANDLE_ERROR( cudaFree( for_nablaPhi_Gx_plus ) );
    HANDLE_ERROR( cudaFree( for_nablaPhi_Gy_plus ) );


    HANDLE_ERROR( cudaFree( for_nabla2F_dsdc ) );

    HANDLE_ERROR( cudaFree( nablaWx ) );
    HANDLE_ERROR( cudaFree( nablaWy ) );

    HANDLE_ERROR( cudaFree( nabla2F_dsdc ) );

    HANDLE_ERROR( cudaFree( vecOnes ) );
    return;

}
