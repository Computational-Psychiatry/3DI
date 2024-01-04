#include "video_fitter.h"
#include "preprocessing.h"
#include "model_fitter.h"

#include <experimental/filesystem>
#include <random>
#include <deque>

#ifdef VISUALIZE_3D
#include "GLfuncs.h"
#endif


using cv::Mat;

VideoFitter::VideoFitter(Camera &cam0,
                         const ushort Kalpha, const ushort Kbeta, const ushort Kepsilon,
                         const ushort Kalpha_L, const ushort Kbeta_L, const ushort Kepsilon_L,
                         size_t _nframes,
                         bool _use_temp_smoothing,
                         bool _use_exp_regularization,
                         float *_h_X0, float *_h_Y0, float *_h_Z0, float *_h_tex_mu):
    T(_nframes),
    use_temp_smoothing(_use_temp_smoothing), use_exp_regularization(_use_exp_regularization),
    detection_net(cv::dnn::readNetFromCaffe(config::FACE_DETECTOR_DPATH, config::FACE_DETECTOR_MPATH)),
    landmark_net(cv::dnn::readNetFromTensorflow(config::LANDMARK_MPATH)),
    leye_net(cv::dnn::readNetFromTensorflow(config::LANDMARK_LEYE_MPATH)),
    reye_net(cv::dnn::readNetFromTensorflow(config::LANDMARK_REYE_MPATH)),
    mouth_net(cv::dnn::readNetFromTensorflow(config::LANDMARK_MOUTH_MPATH)),
    correction_net(cv::dnn::readNetFromTensorflow(config::LANDMARK_CORRECTION_MPATH)),
    r(T, Kalpha, Kbeta, Kepsilon, Kalpha>0, Kbeta>0, Kepsilon>0, _h_X0, _h_Y0, _h_Z0, _h_tex_mu),
    ov(T, Kalpha, Kbeta, Kepsilon, Kalpha>0, Kbeta>0, Kepsilon>0),
    ov_lb(T, Kalpha_L, Kbeta_L, Kepsilon_L, Kalpha_L>0, Kbeta>0, Kepsilon_L>0, true),
    ov_lb_linesearch(T, Kalpha_L, Kbeta_L, Kepsilon_L, Kalpha_L>0, Kbeta>0, Kepsilon_L>0, true),
    ov_linesearch(T, Kalpha, Kbeta, Kepsilon, Kalpha>0, Kbeta>0, Kepsilon>0),
    li_init(NULL, &ov_lb, handleDn, 1.0f, Kalpha_L>0, Kbeta_L>0, Kepsilon_L>0, r, true, config::CONFIDENCE_RANGE, use_temp_smoothing, use_exp_regularization),
    li(NULL, &ov, handleDn, 1.0f, Kalpha>0, Kbeta>0, Kepsilon>0, r, false, config::CONFIDENCE_RANGE, use_temp_smoothing, use_exp_regularization),
    o(&ov, Kalpha, Kbeta, Kepsilon, Kalpha>0, Kbeta>0, Kepsilon>0),
    dc(true, Kalpha>0, Kbeta>0, Kepsilon>0),
    rc(ov.u),
    rc_linesearch(ov_linesearch.u),
    s(handleDn, ov.Ktotal),
    s_lb(handleDn, ov_lb.Ktotal),
    s_lambda(handleDn, 3)
{
    cusolverDnCreate(&handleDn);
    cublasCreate(&handle);

    for (size_t t=0; t<T; ++t)
        cams.push_back(Camera(cam0));

    
    detection_net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    landmark_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    landmark_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    leye_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    leye_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    reye_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    reye_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    mouth_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    mouth_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Check python code in the end of this file to see how this kernel is generated
//    float h_Kernel[7] = {1.19794565e-08, 2.63865077e-04, 1.06450766e-01, 7.86570668e-01, 1.06450766e-01, 2.63865077e-04, 1.19794565e-08};
//    float h_Kernel[15] = {0.00884695, 0.0182159 , 0.0335624 , 0.05533503, 0.08163802, 0.10777792, 0.12732457, 0.13459834, 0.12732457, 0.10777792, 0.08163802, 0.05533503, 0.0335624 , 0.0182159 , 0.00884695};
//    float h_Kernel[15] = {0.00043641, 0.00221626, 0.00876548, 0.02699957, 0.06476859, 0.12100367, 0.1760593 , 0.19950134, 0.1760593 , 0.12100367, 0.06476859, 0.02699957, 0.00876548, 0.00221626, 0.00043641};
/**/
    float *h_Kernel; // [2*KERNEL_RADIUS+1];

    float h_Kernel_200sgm[2*KERNEL_RADIUS+1] = {4.9640312e-06, 8.9220186e-05, 1.0281866e-03, 7.5973268e-03, 3.5993993e-02, 1.0934010e-01, 2.1296541e-01, 2.6596162e-01, 2.1296541e-01, 1.0934010e-01, 3.5993993e-02, 7.5973268e-03, 1.0281866e-03, 8.9220186e-05, 4.9640312e-06};
    float h_Kernel_100sgm[2*KERNEL_RADIUS+1] = {9.1347208e-12, 6.0758834e-09, 1.4867194e-06, 1.3383022e-04, 4.4318484e-03, 5.3990968e-02, 2.4197073e-01, 3.9894229e-01, 2.4197073e-01, 5.3990968e-02, 4.4318484e-03, 1.3383022e-04, 1.4867194e-06, 6.0758834e-09, 9.1347208e-12};
    float h_Kernel_075sgm[2*KERNEL_RADIUS+1] = {6.4550189e-20, 6.7361578e-15, 1.1880850e-10, 3.5416286e-07, 1.7843490e-04, 1.5194189e-02, 2.1867350e-01, 5.3190696e-01, 2.1867350e-01, 1.5194189e-02, 1.7843490e-04, 3.5416286e-07, 1.1880850e-10, 6.7361578e-15, 6.4550189e-20};
    float h_Kernel_050sgm[2*KERNEL_RADIUS+1] = {2.15799964e-43, 4.23189662e-32, 1.51709817e-22, 9.96126162e-15, 1.19794565e-08, 2.63865077e-04, 1.06450766e-01, 7.86570668e-01, 1.06450766e-01, 2.63865077e-04, 1.19794565e-08, 9.96126162e-15, 1.51709817e-22, 4.23189662e-32, 2.15799964e-43};

    if (config::KERNEL_SIGMA == 2.0f) h_Kernel = h_Kernel_200sgm;
    if (config::KERNEL_SIGMA == 1.0f) h_Kernel = h_Kernel_100sgm;
    if (config::KERNEL_SIGMA == 0.75f) h_Kernel = h_Kernel_075sgm;
    if (config::KERNEL_SIGMA == 0.50f) h_Kernel = h_Kernel_050sgm;

    setConvolutionKernel(h_Kernel);

    li_init.cams_ptr = &cams;
    li.cams_ptr = &cams;

    o.ov_ptr = &ov;

    ov.set_frame(0);
    ov_linesearch.set_frame(0);

    cudaMalloc((void**)&search_dir_Lintensity, sizeof(float)*1);
    cudaMalloc((void**)&dg_Lintensity, sizeof(float)*1);
    cudaMalloc( (void**)&d_tmp, sizeof(float));

    cudaMalloc((void**)&d_cropped_face,  sizeof(float)*r.T*DIMX*DIMY);
    cudaMalloc((void**)&d_buffer_face,  sizeof(float)*DIMX*DIMY);


}




VideoOutput VideoFitter::fit_video_frames_auto(const std::string& filepath, LandmarkData& landmark_data, int *min_x_, int *max_x_, int *min_y_, int *max_y_,
                                               const std::string& exp0_path, const std::string& pose0_path, const std::string& illum0_path )
{
    if (min_x_ != NULL) min_x = *min_x_;
    if (min_y_ != NULL) min_y = *min_y_;
    if (max_x_ != NULL) max_x = *max_x_;
    if (max_y_ != NULL) max_y = *max_y_;

    InputData id(T);

    VideoOutput out(ov.Kepsilon);

    if (exp0_path != "" && pose0_path != "" && illum0_path != "")
    {
        vector<vector<float> > exp_coeffs0 = read2DVectorFromFile_unknown_size<float>(exp0_path);
        vector<vector<float> > poses0 = read2DVectorFromFile_unknown_size<float>(pose0_path);
        vector<vector<float> > illums0 = read2DVectorFromFile_unknown_size<float>(illum0_path);

        for (size_t t=0; t<std::min<size_t>(exp_coeffs0.size(), poses0.size()); ++t)
        {
            out.add_exp_coeffs(t, exp_coeffs0[t]);
            out.add_pose(t, poses0[t]);
            out.add_illum(t, illums0[t]);
        }
    }

    cudaEvent_t start, stop;
    cudaEvent_t start_t, stop_t;

    float h_lambdas[3] = {-7.3627f, 51.1364f, 100.1784f};
    float h_Lintensity = 0.005f;

    cudaMemcpy(ov.lambda, h_lambdas, sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(ov.Lintensity, &h_Lintensity, sizeof(float), cudaMemcpyHostToDevice);

    cv::VideoCapture capture(filepath);

    int Nframes = capture.get(cv::CAP_PROP_FRAME_COUNT);
    FPS = capture.get(cv::CAP_PROP_FPS);

    Nframes = std::min<int>(Nframes, config::MAX_VID_FRAMES_TO_PROCESS);
    Nframes = std::min<int>(Nframes, landmark_data.get_num_frames());

    cv::Mat frame;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    cudaEventCreate( &start_t );
    cudaEventCreate( &stop_t );

    cv::VideoWriter videoOutput;

    vector<vector<float> > all_angles;

    float REF_EXTRA = 5.0;

    cudaEventRecord( start_t, 0 );

    uint tot_processed = 0;
    float ref_face_size = -1.0f;

    cv::Rect ROI(-1, -1, -1, -1);

    for( int fi=0; fi<Nframes; ++fi)
    {
        if (fi % config::PRINT_EVERY_N_FRAMES == 0)
            std::cout << "\tframe# " << fi << "/" << Nframes << std::endl;

        all_angles.push_back(std::vector<float>(9, 0.0f));
        float h_us[3] = {NAN, NAN, NAN};
        float h_tau[3] = {NAN, NAN, NAN};
        float yaw(NAN), pitch(NAN), roll(NAN);

        bool frame_successful = false;

        try {
            capture >> frame;
            if (fi < FPS*config::SKIP_FIRST_N_SECS)
                continue;

            if(frame.empty())
                break;

            cams[0].update_camera(1.0f);
            if (cams[0].cam_remap) {
                cv::remap(frame, frame, cams[0].map1, cams[0].map2, cv::INTER_LINEAR);
            }

            float cur_face_size = (float) landmark_data.get_face_size(fi);

            std::vector<float> xp_vec(landmark_data.get_xpvec(fi));
            std::vector<float> yp_vec(landmark_data.get_ypvec(fi));

            float minxl = *std::min_element(std::begin(xp_vec), std::end(xp_vec));
            float maxxl = *std::max_element(std::begin(xp_vec), std::end(xp_vec));
            float minyl = *std::min_element(std::begin(yp_vec), std::end(yp_vec));
            float maxyl = *std::max_element(std::begin(yp_vec), std::end(yp_vec));
            if (minxl > 0.0f && minyl > 0.0f && maxxl < frame.cols && maxyl < frame.rows)
            {
                if (config::PAINT_INNERMOUTH_BLACK) {
                    paint_innermouth_black(frame, xp_vec, yp_vec);
                }

                id.add_data(frame, xp_vec, yp_vec, fi, cur_face_size);
                if (id.frames.size() < T)
                    continue;
            }
            else
            {
                xp_vec.clear();
                yp_vec.clear();
                ROI.x = -1;
                ROI.y = -1;
                ROI.width = -1;
                ROI.height = -1;
//                std::cout << "Problem during landmark detection -- skipping ..." << std::endl;
                id.clear();
                throw std::runtime_error("Problem during landmark detection -- skipping ...");
            }

            if (xp_vec.size() == 0) {
                id.clear();
                throw std::runtime_error("failed to detect landmarks");
            }

            // @@@ <!-- probably needs to be done for all cameras -->
            for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
            {
                cams[rel_fid].update_camera(1.0f);
                /** Test if fitting landmark fails, skip if necessary */
                li_init.set_landmarks_from_host(rel_fid, &(id.xp_origs[rel_fid])[0], &(id.yp_origs[rel_fid])[0]);
                li_init.initialize_with_orthographic_t(handleDn, handle, rel_fid,
                                                       &(id.xp_origs[rel_fid])[0], &(id.yp_origs[rel_fid])[0], id.face_sizes[rel_fid], &ov_lb);
                li.set_landmarks_from_host(rel_fid, &(id.xp_origs[rel_fid])[0], &(id.yp_origs[rel_fid])[0]);
                li.initialize_with_orthographic_t(handleDn, handle, rel_fid,
                                                  &(id.xp_origs[rel_fid])[0], &(id.yp_origs[rel_fid])[0], id.face_sizes[rel_fid], &ov);
            }
            li_init.set_minimal_slack(handle, &ov_lb);

            ov_lb.set_frame(0);
            ov_lb_linesearch.set_frame(0);

            ov.set_frame(0);
            ov_linesearch.set_frame(0);


            li_init.fit_model(handleDn, handle, &ov_lb, &ov_lb_linesearch);
            li.copy_from_initialized(li_init);


            ov_lb.set_frame(0);
            ov_lb_linesearch.set_frame(0);

            ov.set_frame(0);
            ov_linesearch.set_frame(0);


            HANDLE_ERROR( cudaMemcpy( ov.taux, ov_lb.taux, sizeof(float)*6*ov.T, cudaMemcpyDeviceToDevice ) );
            HANDLE_ERROR( cudaMemcpy( ov.alphas, ov_lb.alphas, sizeof(float)*ov_lb.Kalpha, cudaMemcpyDeviceToDevice ) );
            HANDLE_ERROR( cudaMemcpy( ov_linesearch.taux, ov_lb.taux, sizeof(float)*6*ov.T, cudaMemcpyDeviceToDevice ) );
            HANDLE_ERROR( cudaMemcpy( ov_linesearch.alphas, ov_lb.alphas, sizeof(float)*ov_lb.Kalpha, cudaMemcpyDeviceToDevice ) );

            for (uint rel_fid=0; rel_fid<ov.T; ++rel_fid) {
                HANDLE_ERROR( cudaMemcpy( ov.epsilons+rel_fid*ov.Kepsilon, ov_lb.epsilons+rel_fid*ov_lb.Kepsilon, sizeof(float)*ov_lb.Kepsilon, cudaMemcpyDeviceToDevice ) );
                HANDLE_ERROR( cudaMemcpy( ov_linesearch.epsilons+rel_fid*ov.Kepsilon, ov_lb.epsilons+rel_fid*ov_lb.Kepsilon, sizeof(float)*ov_lb.Kepsilon, cudaMemcpyDeviceToDevice ) );

                float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
                id.get_resized_landmarks(rel_fid, cams[rel_fid].resize_coef, xp, yp);

                // @@@ repeat this at each N-frame processing
                r.set_x0_short_y0_short(rel_fid, xp, yp);
            }


            //print_vector(ov_lb.alphas, li_init.Ktotal, "ov_lb");
            //print_vector(ov.alphas, li_init.Ktotal, "ov");
            li.fit_model(handleDn, handle, &ov, &ov_linesearch);

            if (!li_init.fit_success) {
                id.clear();
                throw std::runtime_error("failed to fit logbarrier initializer");
            }

            float RE = REF_EXTRA;
            float RE2 = REF_EXTRA/2.0f;
            float RSFS = config::REF_FACE_SIZE;

            std::vector<float> ref_sizes({RSFS, RSFS+RE+5.0f, RSFS+RE+3.0f, RSFS+RE+9.0f, RSFS+RE+7.0f,
                                          RSFS+RE-9.0f, RSFS+RE-6.0f, RSFS+RE-3.0f, RSFS+RE2+5.0f, RSFS+RE2+3.0f, RSFS+RE2+9.0f,
                                          RSFS+RE2+7.0f, RSFS+RE2-9.0f, RSFS+RE2-6.0f, RSFS+RE2-3.0f});
            ref_sizes.resize(config::NRES_COEFS);

            bool success = true;
            for (float& cur_ref_size : ref_sizes)
            {
                success = success && update_shape_single_ref_size(id, cur_ref_size, out);
            }
            cudaEventRecord( stop_t, 0 );
            cudaEventSynchronize( stop_t );

            frame_successful = true;
            float   elapsedTime_t;

            cudaEventElapsedTime( &elapsedTime_t, start_t, stop_t );

            rc.compute_euler_angles(yaw, pitch, roll);
            cudaMemcpy(h_tau, ov.taux, 3*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_us, ov.u, 3*sizeof(float), cudaMemcpyDeviceToHost);


            /*
            for (size_t rel_fid=0; rel_fid<ov.T; ++rel_fid)
            {
                ov.set_frame(rel_fid);
                print_vector(ov.epsilons, ov.Kepsilon, "");
            }
            ov.set_frame(0);

            for (size_t rel_fid=0; rel_fid<ov.T; ++rel_fid)
            {
                ov.set_frame(rel_fid);
                print_vector(ov.taux, 6, "");
            }
            ov.set_frame(0);
            */

            tot_processed++;

            //! PUTBACK
            if (tot_processed == Nframes)
                break;
        //try { 
        } catch (std::exception& e) {
            if (config::PRINT_DEBUG)
            {
                std::cout << " PROBLEM -- SKIPPING" << std::endl;
                std::cout << e.what() << std::endl;
            }
            ov.reset();
            ov_lb.reset();
            //!cv::waitKey(1);
        }
    }

    // get stop time, and display the timing results
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    ushort pixel_idx[config::Nredundant];
    bool rend_flag[config::N_TRIANGLES*NTMP];

    cudaMemcpy( pixel_idx, r.d_pixel_idx,  sizeof(ushort)*config::Nredundant, cudaMemcpyDeviceToHost );
    cudaMemcpy( rend_flag, r.d_rend_flag,  sizeof(bool)*config::N_TRIANGLES*NTMP, cudaMemcpyDeviceToHost );

    return out;
}






bool VideoFitter::update_shape_single_ref_size(InputData &id, float ref_face_size, VideoOutput& out)
{

    for (size_t i=0; i<T; ++i) {
        cams[i].update_camera(ref_face_size/id.face_sizes[i]);
    }

    bool success;
    if (config::FINETUNE_ONLY)
        success = fit_to_video_frame_finetune_only(id, out);
    else
        success = fit_to_video_frame(id, out);

    float exp_coeffs[config::K_EPSILON];
    float pose[6];
    float illum_coeffs[4];

    for (size_t rel_fid=0; rel_fid<T; ++rel_fid) {
        ov.set_frame(rel_fid);
        cudaMemcpy(exp_coeffs, ov.epsilons, sizeof(float)*config::K_EPSILON, cudaMemcpyDeviceToHost);
        cudaMemcpy(pose, ov.taux, sizeof(float)*6, cudaMemcpyDeviceToHost);
        cudaMemcpy(illum_coeffs, ov.lambdas, sizeof(float)*3, cudaMemcpyDeviceToHost);
        cudaMemcpy(illum_coeffs+3, ov.Lintensity, sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> exp_coeffs_vec(exp_coeffs, exp_coeffs+config::K_EPSILON);
        std::vector<float> pose_vec(pose, pose+6);

        out.add_exp_coeffs(id.abs_frame_ids[rel_fid], exp_coeffs_vec);

        if (!config::FINETUNE_ONLY)
        {
            out.add_pose(id.abs_frame_ids[rel_fid], pose_vec);
            out.add_illum(id.abs_frame_ids[rel_fid], std::vector<float>(illum_coeffs, illum_coeffs+4));
        }
    }
    ov.set_frame(0);

    return success;
}




bool VideoFitter::update_shape_single_ref_size_finetune_only(InputData &id, float ref_face_size, VideoOutput& out)
{

    for (size_t i=0; i<T; ++i) {
        cams[i].update_camera(ref_face_size/id.face_sizes[i]);
    }

    bool success = fit_to_video_frame(id, out);

    float exp_coeffs[config::K_EPSILON];
    float pose[6];
    float illum_coeffs[4];

    for (size_t rel_fid=0; rel_fid<T; ++rel_fid) {
        ov.set_frame(rel_fid);
        cudaMemcpy(exp_coeffs, ov.epsilons, sizeof(float)*config::K_EPSILON, cudaMemcpyDeviceToHost);
        cudaMemcpy(pose, ov.taux, sizeof(float)*6, cudaMemcpyDeviceToHost);
        cudaMemcpy(illum_coeffs, ov.lambdas, sizeof(float)*3, cudaMemcpyDeviceToHost);
        cudaMemcpy(illum_coeffs+3, ov.Lintensity, sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> exp_coeffs_vec(exp_coeffs, exp_coeffs+config::K_EPSILON);
        std::vector<float> pose_vec(pose, pose+6);

        out.add_exp_coeffs(id.abs_frame_ids[rel_fid], exp_coeffs_vec);
        out.add_pose(id.abs_frame_ids[rel_fid], pose_vec);
        out.add_illum(id.abs_frame_ids[rel_fid], std::vector<float>(illum_coeffs, illum_coeffs+4));
    }
    ov.set_frame(0);

    return success;
}


// @@@ Change name to "fit_to_video_frames"
bool VideoFitter::fit_to_video_frame(InputData &id, VideoOutput& out)
{
    bool success = false;
    bool we_have_good_landmarks = true;

    for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        we_have_good_landmarks = we_have_good_landmarks && id.xp_origs[rel_fid][0] != 1.0f;

    ov.reset_tau_logbarrier();
    ov_linesearch.reset_tau_logbarrier();

    ov_lb.reset();
    ov_lb_linesearch.reset();

    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);


    cudaMemset(o.d_TEX_ID_NREF, 0, sizeof(float)*Nrender_estimated*3850*ov.T);
    cudaMemset(d_cropped_face, 0, sizeof(float)*DIMX*r.T*DIMY);
    cudaMemset(d_buffer_face, 0, sizeof(float)*DIMX*DIMY);



    bool is_bb_OK = true;
    if (we_have_good_landmarks)
    {
        for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        {
            float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
            id.get_resized_landmarks(rel_fid, cams[rel_fid].resize_coef, xp, yp);

            float face_size = compute_face_size(xp, yp);

            cudaMemset(ov_lb.slack, 0, sizeof(float));
            cudaMemset(ov_lb_linesearch.slack, 0, sizeof(float));

            ov_lb.reset_tau_logbarrier();
            ov_lb_linesearch.reset_tau_logbarrier();
            ov.reset_tau_logbarrier();
            ov_linesearch.reset_tau_logbarrier();

            // --->***<--- // PLACE COMPUTATION OF IOD HERE
            li.set_landmarks_from_host(rel_fid, xp, yp);
            li_init.set_landmarks_from_host(rel_fid, xp, yp);
            li_init.initialize_with_orthographic_t(handleDn, handle, rel_fid, xp, yp, face_size, &ov_lb);
            is_bb_OK = is_bb_OK && check_if_bb_OK(xp, yp);
        }
    }

    li_init.set_minimal_slack(handle, &ov_lb);
    li_init.fit_model(handleDn, handle, &ov_lb, &ov_lb_linesearch);

    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);

    if (li_init.fit_success && is_bb_OK && we_have_good_landmarks)
    {
        li.copy_from_initialized(li_init);

        HANDLE_ERROR( cudaMemcpy( ov.taux, ov_lb.taux, sizeof(float)*6*ov.T, cudaMemcpyDeviceToDevice ) );
        HANDLE_ERROR( cudaMemcpy( ov.alphas, ov_lb.alphas, sizeof(float)*ov_lb.Kalpha, cudaMemcpyDeviceToDevice ) );

        for (uint rel_fid=0; rel_fid<ov.T; ++rel_fid) {
            HANDLE_ERROR( cudaMemcpy( ov.epsilons+rel_fid*ov.Kepsilon, ov_lb.epsilons+rel_fid*ov_lb.Kepsilon, sizeof(float)*ov_lb.Kepsilon, cudaMemcpyDeviceToDevice ) );

            float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
            id.get_resized_landmarks(rel_fid, cams[rel_fid].resize_coef, xp, yp);

            // @@@ repeat this at each N-frame processing
            r.set_x0_short_y0_short(rel_fid, xp, yp);
        }

    }
    else
    {
        for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        {
            li.compute_bounds(rel_fid, 1.0, 1.0, 1.0, true);

            if (config::PRINT_WARNINGS)
               std::cout << "Failed fitting at this frame -- will ignore bounds" << std::endl;

            float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
            id.get_resized_landmarks(rel_fid, cams[rel_fid].resize_coef, xp, yp);

            // @@@ repeat this at each N-frame processing
            r.set_x0_short_y0_short(rel_fid, xp, yp);
        }
    }

    //    print_vector(ov_lb.taux, 6*3, "taus_lb");
    //    print_vector(ov.taux, 6*3, "taus");



    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);

    for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
    {
        cv::Mat inputImage;
        id.get_resized_frame(rel_fid, cams[rel_fid].resize_coef, inputImage);

        cv::Mat cropped_face_upright = inputImage(cv::Rect(r.x0_short[rel_fid], r.y0_short[rel_fid], DIMX, DIMY)).clone();
        // Note that we transpose the image here to make it column major as the rest
        cv::Mat cropped_face_mat = cv::Mat(cropped_face_upright.t()).clone();

        HANDLE_ERROR( cudaMemcpy( d_cropped_face+rel_fid*(DIMX*DIMY), cropped_face_mat.data, sizeof(float)*DIMX*DIMY, cudaMemcpyHostToDevice ) );

        convolutionRowsGPU( d_buffer_face, d_cropped_face+rel_fid*(DIMX*DIMY), DIMX, DIMY );
        convolutionColumnsGPU(d_cropped_face+rel_fid*(DIMX*DIMY), d_buffer_face, DIMX, DIMY );
    }


//    if (id.abs_frame_ids[0] % 90 == 0)
    {
        for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        {
            ov.set_frame(rel_fid);
            ov_linesearch.set_frame(rel_fid);

            float h_Lintensity = 0.005f;
            cudaMemcpy(ov.Lintensity, &h_Lintensity, sizeof(float), cudaMemcpyHostToDevice);

            fit_3DMM_lambdas(rel_fid, r, o, handleDn, handle, d_cropped_face, d_buffer_face,
                             ov, ov_linesearch, cams[rel_fid], rc, rc_linesearch, dc, s_lambda,  d_tmp, false, true);
//            print_vector(ov.lambda, 3, "");
//            print_vector(ov.Lintensity, 1, "");
        }
    }

    ov.set_frame(0);
    ov_linesearch.set_frame(0);

    cudaMemcpy(ov_linesearch.lambdas, ov.lambdas, ov.T*3*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ov_linesearch.Lintensity, ov.Lintensity, ov.T*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ov_linesearch.betas, ov.betas, ov.Ktotal*sizeof(float), cudaMemcpyDeviceToDevice);

    // The function below is multiframe
    fit_3DMM_shape_rigid(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face,
                         ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);


    if (config::TWOSTAGE_ILLUM) {
        for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        {

            ov.set_frame(rel_fid);
            ov_linesearch.set_frame(rel_fid);

            fit_3DMM_lambdas(rel_fid, r, o, handleDn, handle, d_cropped_face, d_buffer_face,
                             ov, ov_linesearch, cams[rel_fid], rc, rc_linesearch, dc, s_lambda,  d_tmp, false, true);
        }

        ov.set_frame(0);
        ov_linesearch.set_frame(0);

        cudaMemcpy(ov_linesearch.lambdas, ov.lambdas, ov.T*3*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ov_linesearch.Lintensity, ov.Lintensity, ov.T*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ov_linesearch.betas, ov.betas, ov.Ktotal*sizeof(float), cudaMemcpyDeviceToDevice);

        // The function below is multiframe
        fit_3DMM_shape_rigid(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face,
                             ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);

        for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        {

            ov.set_frame(rel_fid);
            ov_linesearch.set_frame(rel_fid);

            fit_3DMM_lambdas(rel_fid, r, o, handleDn, handle, d_cropped_face, d_buffer_face,
                             ov, ov_linesearch, cams[rel_fid], rc, rc_linesearch, dc, s_lambda,  d_tmp, false, true);
        }

        ov.set_frame(0);
        ov_linesearch.set_frame(0);

        cudaMemcpy(ov_linesearch.lambdas, ov.lambdas, ov.T*3*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ov_linesearch.Lintensity, ov.Lintensity, ov.T*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ov_linesearch.betas, ov.betas, ov.Ktotal*sizeof(float), cudaMemcpyDeviceToDevice);

        // The function below is multiframe
        fit_3DMM_shape_rigid(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face,
                             ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);
    }


    if (config::FINETUNE_EXPRESSIONS)
        fit_3DMM_epsilon_finetune(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);

    //HANDLE_ERROR( cudaMemcpy( li.epsilon_lb, li.epsilon_lb_finetune, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );
    //HANDLE_ERROR( cudaMemcpy( li.epsilon_ub, li.epsilon_ub_finetune, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );

    //fit_3DMM_rigid_alone(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);
    //fit_3DMM_epsilon_finetune(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);

    /*
    */

    success = true;


    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);

    return success;
}






// @@@ Change name to "fit_to_video_frames"
bool VideoFitter::fit_to_video_frame_finetune_only(InputData &id, VideoOutput& out)
{
    bool success = false;
    bool we_have_good_landmarks = true;

    for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        we_have_good_landmarks = we_have_good_landmarks && id.xp_origs[rel_fid][0] != 1.0f;

    ov.reset_tau_logbarrier();
    ov_linesearch.reset_tau_logbarrier();

    ov_lb.reset();
    ov_lb_linesearch.reset();

    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);


    cudaMemset(o.d_TEX_ID_NREF, 0, sizeof(float)*Nrender_estimated*3850*ov.T);
    cudaMemset(d_cropped_face, 0, sizeof(float)*DIMX*r.T*DIMY);
    cudaMemset(d_buffer_face, 0, sizeof(float)*DIMX*DIMY);


    bool is_bb_OK = true;
    if (we_have_good_landmarks)
    {
        for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        {
            float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
            id.get_resized_landmarks(rel_fid, cams[rel_fid].resize_coef, xp, yp);

            float face_size = compute_face_size(xp, yp);

            cudaMemset(ov_lb.slack, 0, sizeof(float));
            cudaMemset(ov_lb_linesearch.slack, 0, sizeof(float));

            ov_lb.reset_tau_logbarrier();
            ov_lb_linesearch.reset_tau_logbarrier();
            ov.reset_tau_logbarrier();
            ov_linesearch.reset_tau_logbarrier();

            // --->***<--- // PLACE COMPUTATION OF IOD HERE
            li.set_landmarks_from_host(rel_fid, xp, yp);
            li_init.set_landmarks_from_host(rel_fid, xp, yp);
            li_init.initialize_with_orthographic_t(handleDn, handle, rel_fid, xp, yp, face_size, &ov_lb);
            is_bb_OK = is_bb_OK && check_if_bb_OK(xp, yp);
        }
    }

    li_init.set_minimal_slack(handle, &ov_lb);
    li_init.fit_model(handleDn, handle, &ov_lb, &ov_lb_linesearch);

    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);

    if (li_init.fit_success && is_bb_OK && we_have_good_landmarks)
    {
        li.copy_from_initialized(li_init);

        HANDLE_ERROR( cudaMemcpy( ov.taux, ov_lb.taux, sizeof(float)*6*ov.T, cudaMemcpyDeviceToDevice ) );
        HANDLE_ERROR( cudaMemcpy( ov.alphas, ov_lb.alphas, sizeof(float)*ov_lb.Kalpha, cudaMemcpyDeviceToDevice ) );

        for (uint rel_fid=0; rel_fid<ov.T; ++rel_fid) {
            HANDLE_ERROR( cudaMemcpy( ov.epsilons+rel_fid*ov.Kepsilon, ov_lb.epsilons+rel_fid*ov_lb.Kepsilon, sizeof(float)*ov_lb.Kepsilon, cudaMemcpyDeviceToDevice ) );

            float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
            id.get_resized_landmarks(rel_fid, cams[rel_fid].resize_coef, xp, yp);

            // @@@ repeat this at each N-frame processing
            r.set_x0_short_y0_short(rel_fid, xp, yp);
        }

    }
    else
    {
        for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
        {
            li.compute_bounds(rel_fid, 1.0, 1.0, 1.0, true);

            if (config::PRINT_WARNINGS)
               std::cout << "Failed fitting at this frame -- will ignore bounds" << std::endl;

            float xp[NLANDMARKS_51], yp[NLANDMARKS_51];
            id.get_resized_landmarks(rel_fid, cams[rel_fid].resize_coef, xp, yp);

            // @@@ repeat this at each N-frame processing
            r.set_x0_short_y0_short(rel_fid, xp, yp);
        }
    }

    //    print_vector(ov_lb.taux, 6*3, "taus_lb");
    //    print_vector(ov.taux, 6*3, "taus");



    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);

    for (size_t rel_fid=0; rel_fid<T; ++rel_fid)
    {
        cv::Mat inputImage;
        id.get_resized_frame(rel_fid, cams[rel_fid].resize_coef, inputImage);

        cv::Mat cropped_face_upright = inputImage(cv::Rect(r.x0_short[rel_fid], r.y0_short[rel_fid], DIMX, DIMY)).clone();
        // Note that we transpose the image here to make it column major as the rest
        cv::Mat cropped_face_mat = cv::Mat(cropped_face_upright.t()).clone();

        HANDLE_ERROR( cudaMemcpy( d_cropped_face+rel_fid*(DIMX*DIMY), cropped_face_mat.data, sizeof(float)*DIMX*DIMY, cudaMemcpyHostToDevice ) );

        convolutionRowsGPU( d_buffer_face, d_cropped_face+rel_fid*(DIMX*DIMY), DIMX, DIMY );
        convolutionColumnsGPU(d_cropped_face+rel_fid*(DIMX*DIMY), d_buffer_face, DIMX, DIMY );
    }


    vector<float> cur_illum = out.illum_coeffs[id.abs_frame_ids[0]];
    vector<float> cur_exp = out.exp_coefs[id.abs_frame_ids[0]][0];
    vector<float> cur_pose = out.poses[id.abs_frame_ids[0]][0];



    cudaMemcpy(ov_linesearch.lambdas, &cur_illum[0], 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ov_linesearch.Lintensity, &cur_illum[0]+3, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ov.lambdas, &cur_illum[0], 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ov.Lintensity, &cur_illum[0]+3, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(ov_linesearch.taux, &cur_pose[0], 6*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ov.taux, &cur_pose[0], 6*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(ov_linesearch.epsilons, &cur_exp[0], config::K_EPSILON*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ov.epsilons, &cur_exp[0], config::K_EPSILON*sizeof(float), cudaMemcpyHostToDevice);

    if (config::FINETUNE_EXPRESSIONS)
        fit_3DMM_epsilon_finetune(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);

    //HANDLE_ERROR( cudaMemcpy( li.epsilon_lb, li.epsilon_lb_finetune, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );
    //HANDLE_ERROR( cudaMemcpy( li.epsilon_ub, li.epsilon_ub_finetune, sizeof(float)*ov.Kepsilon, cudaMemcpyDeviceToDevice ) );

    //fit_3DMM_rigid_alone(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);
    //fit_3DMM_epsilon_finetune(0, r, o, li, handleDn, handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, cams, rc, rc_linesearch, dc, s, d_tmp, false);

    /*
    */

    success = true;


    ov.set_frame(0);
    ov_linesearch.set_frame(0);
    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);

    return success;
}











bool VideoFitter::output_facial_parts(VideoOutput& out, std::string& video_path, std::string& output_path_sleye,
                                  std::string& output_path_sreye, std::string& output_path_smouth,
                                  vector<vector<float> >* all_exps, vector<vector<float> >* all_poses)
{
    cv::VideoCapture vidIn(video_path);
    cv::VideoWriter vidOut_m, vidOut_le, vidOut_re;

    int width = vidIn.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = vidIn.get(cv::CAP_PROP_FRAME_HEIGHT);

    int partSize = 150;



    if (!vidOut_m.isOpened())
        vidOut_m.open(output_path_smouth, cv::VideoWriter::fourcc('D','I','V','X'), FPS, cv::Size(partSize,partSize), true);

    if (!vidOut_le.isOpened())
        vidOut_le.open(output_path_sleye, cv::VideoWriter::fourcc('D','I','V','X'), FPS, cv::Size(partSize,partSize), true);

    if (!vidOut_re.isOpened())
        vidOut_re.open(output_path_sreye, cv::VideoWriter::fourcc('D','I','V','X'), FPS, cv::Size(partSize,partSize), true);

    vidOut_le.set(cv::VIDEOWRITER_PROP_QUALITY, 100);
    vidOut_re.set(cv::VIDEOWRITER_PROP_QUALITY, 100);
    vidOut_m.set(cv::VIDEOWRITER_PROP_QUALITY, 100);

    size_t T = out.exp_coefs.size();

    float *h_Xe = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ye = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ze = (float*)malloc(config::NPTS*sizeof(float));

    float *h_Xr = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Yr = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Zr = (float*)malloc(config::NPTS*sizeof(float));

    float *d_Xr, *d_Yr, *d_Zr;

    cudaMalloc((void**)&d_Xr,  sizeof(float)*config::NPTS);
    cudaMalloc((void**)&d_Yr,  sizeof(float)*config::NPTS);
    cudaMalloc((void**)&d_Zr,  sizeof(float)*config::NPTS);

    cv::Mat frame;


    int Nframes = vidIn.get(cv::CAP_PROP_FRAME_COUNT);
    FPS = vidIn.get(cv::CAP_PROP_FPS);

    Nframes = std::min<int>(Nframes, config::MAX_VID_FRAMES_TO_PROCESS);

    if (all_exps != NULL)
        Nframes = std::min<int>(Nframes, all_exps->size());

    std::deque<float> xlcs, xrcs;
    std::deque<float> ylcs, yrcs;
    std::deque<float> xms, yms;
    std::deque<float> ciods;
    std::deque<float> rolls;


    float *h_xp = (float*)malloc( config::NPTS*sizeof(float) );
    float *h_yp = (float*)malloc( config::NPTS*sizeof(float) );

    for (size_t fi=0; fi<Nframes; ++fi)
    {
        cams[0].update_camera(1.0f);

        std::vector<float> exp_coeffs, pose;
        if (NULL != all_exps) {
            if (fi >= all_exps->size())
                break;
            exp_coeffs = all_exps->at(fi);
        } else
            exp_coeffs = out.compute_avg_exp_coeffs(fi);

        if (NULL != all_poses) {
            if (fi >= all_poses->size())
                break;
            pose = all_poses->at(fi);
        } else
            pose = out.compute_avg_pose(fi);

        std::vector<float> npose(pose);

        npose[3] = 0.1f;
        npose[4] = 0.0f;
        npose[5] = 0.0f;


        vidIn >> frame;

        if (!config::PREPEND_BLANK_FRAMES)
        {
            if (fi < config::SKIP_FIRST_N_SECS*FPS) {
                continue;
            }
        }

        if (isnan(exp_coeffs[0]) || isnan(pose[0])) {
            cv::Mat blank(partSize, partSize, CV_8UC3, cv::Scalar::all(255));
            vidOut_m << blank;
            vidOut_le << blank;
            vidOut_re << blank;
            continue;
        }


        if (cams[0].cam_remap) {
            cv::remap(frame, frame, cams[0].map1, cams[0].map2, cv::INTER_LINEAR);
        }

        ov.set_frame(0);
//        cudaMemcpy(ov.epsilons, &exp_coeffs[0], r.Kepsilon*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.tauxs, &npose[0], 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.u, &npose[0]+3, 3*sizeof(float), cudaMemcpyHostToDevice);

        rc.set_u_ptr(ov.u);
        rc.process();


        r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);
        cudaMemcpy(h_Xe, r.X0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ye, r.Y0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ze, r.Z0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
//        cv::Mat im3d = drawFace_fromptr(r.tl_vector, h_Xe, h_Ye, h_Ze);
        r.compute_nonrigid_shape_identity_and_rotation(handle, ov, rc.R, d_Xr, d_Yr, d_Zr);

        view_transform_3d_pts_and_render_2d<<<(config::NPTS+NTHREADS-1)/NTHREADS, NTHREADS>>>(r.X0, r.Y0, r.Z0,
                                                                                      rc.R, ov.taux, ov.tauy, ov.tauz,
                                                                                      cams[0].phix, cams[0].phiy, cams[0].cx, cams[0].cy,
                                                                                      d_Xr, d_Yr, d_Zr,
                                                                                      r.d_xp, r.d_yp, config::NPTS);

        float xleo, xreo, xlei, xrei, yleo, yreo, ylei, yrei;
        cudaMemcpy(&xleo, r.d_xp+config::LIS[19], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&xreo, r.d_xp+config::LIS[28], sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> xp_vec, yp_vec;
        for (uint pi=0; pi<config::LIS.size(); ++pi)
        {
            xp_vec.push_back(h_xp[config::LIS[pi]]);
            yp_vec.push_back(h_yp[config::LIS[pi]]);
        }

//        float canonical_iod = fabs(xreo-xleo);

        float stdx = std::sqrt(variance<float>(xp_vec));
        float stdy = std::sqrt(variance<float>(yp_vec));

        float canonical_iod = 3*(stdx+stdy)/2.0f;

        ov.set_frame(0);
//        cudaMemcpy(ov.epsilons, &exp_coeffs[0], r.Kepsilon*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.tauxs, &pose[0], 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.u, &pose[0]+3, 3*sizeof(float), cudaMemcpyHostToDevice);

        rc.set_u_ptr(ov.u);
        rc.process();

        r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);
        cudaMemcpy(h_Xe, r.X0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ye, r.Y0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ze, r.Z0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        r.compute_nonrigid_shape_identity_and_rotation(handle, ov, rc.R, d_Xr, d_Yr, d_Zr);

        view_transform_3d_pts_and_render_2d<<<(config::NPTS+NTHREADS-1)/NTHREADS, NTHREADS>>>(r.X0, r.Y0, r.Z0,
                                                                                      rc.R, ov.taux, ov.tauy, ov.tauz,
                                                                                      cams[0].phix, cams[0].phiy, cams[0].cx, cams[0].cy,
                                                                                      d_Xr, d_Yr, d_Zr,
                                                                                      r.d_xp, r.d_yp, config::NPTS);

        cudaMemcpy(h_xp, r.d_xp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_yp, r.d_yp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);

        float xmc(0.0f), ymc(0.0f);
        for (uint pi=31; pi<config::LIS.size(); ++pi)
        {
            xmc += h_xp[config::LIS[pi]];
            ymc += h_yp[config::LIS[pi]];
        }

        xmc /= 20.0;
        ymc /= 20.0;

        cudaMemcpy(&xleo, r.d_xp+config::LIS[19], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&xreo, r.d_xp+config::LIS[28], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&yleo, r.d_yp+config::LIS[19], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&yreo, r.d_yp+config::LIS[28], sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(&xlei, r.d_xp+config::LIS[22], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&xrei, r.d_xp+config::LIS[25], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ylei, r.d_yp+config::LIS[22], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&yrei, r.d_yp+config::LIS[25], sizeof(float), cudaMemcpyDeviceToHost);


        cv::Point2f ple((xleo+xlei)/2.0f, (yleo+ylei)/2.0f);
        cv::Point2f pre((xreo+xrei)/2.0f, (yreo+yrei)/2.0f);

        double tri_h = sqrtf((ple.x-pre.x)*(ple.x-pre.x)+(ple.y-pre.y)*(ple.y-pre.y));
        double tri_dy = fabs((ple.y-pre.y));
        double angle = RAD2DEG(asin(tri_dy/tri_h));

        if (pre.y>ple.y) {
            angle *= -1.0f;
        }

        xlcs.push_back(ple.x);
        xrcs.push_back(pre.x);
        ylcs.push_back(ple.y);
        yrcs.push_back(pre.y);
        xms.push_back(xmc);
        yms.push_back(ymc);
        ciods.push_back(canonical_iod);
        rolls.push_back(angle);

        double smoothing_tw = 0.001f; // in secs
        if (xlcs.size() > 1 && xlcs.size() >= std::round(smoothing_tw*FPS)) {
            xlcs.pop_front();
            xrcs.pop_front();
            ylcs.pop_front();
            yrcs.pop_front();
            xms.pop_front();
            yms.pop_front();
            ciods.pop_front();
            rolls.pop_front();
        }

        float xlc = std::accumulate(xlcs.begin(), xlcs.end(), 0.0f)/xlcs.size();
        float xrc = std::accumulate(xrcs.begin(), xrcs.end(), 0.0f)/xrcs.size();
        float ylc = std::accumulate(ylcs.begin(), ylcs.end(), 0.0f)/ylcs.size();
        float yrc = std::accumulate(yrcs.begin(), yrcs.end(), 0.0f)/yrcs.size();
        float xm = std::accumulate(xms.begin(), xms.end(), 0.0f)/xms.size();
        float ym = std::accumulate(yms.begin(), yms.end(), 0.0f)/yms.size();
        float ciod = std::accumulate(ciods.begin(), ciods.end(), 0.0f)/ciods.size();
        float roll = std::accumulate(rolls.begin(), rolls.end(), 0.0f)/rolls.size();

        float eye_size = ciod*1.55;
        float mouth_size = ciod*1.85;

        cv::Point2f tl_m(xm-mouth_size, ym-mouth_size);
        cv::Point2f br_m(xm+mouth_size, ym+mouth_size);

        cv::Point2f tl_le(xlc-eye_size, ylc-eye_size);
        cv::Point2f br_le(xlc+eye_size, ylc+eye_size);

        cv::Point2f tl_re(xrc-eye_size, yrc-eye_size);
        cv::Point2f br_re(xrc+eye_size, yrc+eye_size);

        std::vector<cv::Point2f> src_m, dst_m;
        std::vector<cv::Point2f> src_le, dst_le;
        std::vector<cv::Point2f> src_re, dst_re;
        src_m.push_back(tl_m);
        src_m.push_back(br_m);
        src_le.push_back(tl_le);
        src_le.push_back(br_le);
        src_re.push_back(tl_re);
        src_re.push_back(br_re);

        std::vector<cv::Point2f> dst;
        dst.push_back(cv::Point2f(0, 0));
        dst.push_back(cv::Point2f(300, 300));

        cv::Mat trMat_m  = cv::estimateRigidTransform(src_m, dst, false);
        cv::Mat trMat_le = cv::estimateRigidTransform(src_le, dst, false);
        cv::Mat trMat_re = cv::estimateRigidTransform(src_re, dst, false);

        cv::Mat newFrame_m, newFrame_le, newFrame_re;
        warpAffine(frame, newFrame_m, trMat_m, cv::Size(300, 300));
        warpAffine(frame, newFrame_le, trMat_le, cv::Size(300, 300));
        warpAffine(frame, newFrame_re, trMat_re, cv::Size(300, 300));

        cv::Mat newFrameRot_m, newFrameRot_le, newFrameRot_re;

        cv::Point2f center((newFrame_m.cols-1)/2.0, (newFrame_m.rows-1)/2.0);
        cv::Mat rot = cv::getRotationMatrix2D(center, -roll, 1.0);

        cv::warpAffine(newFrame_m, newFrameRot_m, rot, newFrame_m.size());
        cv::warpAffine(newFrame_le, newFrameRot_le, rot, newFrame_le.size());
        cv::warpAffine(newFrame_re, newFrameRot_re, rot, newFrame_re.size());

        cv::Rect crop_region(75, 75, partSize, partSize);

        vidOut_m << newFrameRot_m(crop_region);
        vidOut_le << newFrameRot_le(crop_region);
        vidOut_re << newFrameRot_re(crop_region);
    }

    free( h_xp );
    free( h_yp );

    cudaFree( d_Xr );
    cudaFree( d_Yr );
    cudaFree( d_Zr );

    free(h_Xr);
    free(h_Yr);
    free(h_Zr);

    free(h_Xe);
    free(h_Ye);
    free(h_Ze);

    return true;
}






bool VideoFitter::output_landmarks(VideoOutput& out, std::string& video_path, std::string& output_landmarks_path,
                                  vector<vector<float> >* all_exps, vector<vector<float> >* all_poses)
{
    cv::VideoCapture vidIn(video_path);

    int width = vidIn.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = vidIn.get(cv::CAP_PROP_FRAME_HEIGHT);

    size_t T = out.exp_coefs.size();

    float *h_Xe = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ye = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ze = (float*)malloc(config::NPTS*sizeof(float));

    float *h_Xr = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Yr = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Zr = (float*)malloc(config::NPTS*sizeof(float));

    float *d_Xr, *d_Yr, *d_Zr;

    cudaMalloc((void**)&d_Xr,  sizeof(float)*config::NPTS);
    cudaMalloc((void**)&d_Yr,  sizeof(float)*config::NPTS);
    cudaMalloc((void**)&d_Zr,  sizeof(float)*config::NPTS);

    cv::Mat frame;

    int Nframes = vidIn.get(cv::CAP_PROP_FRAME_COUNT);
    FPS = vidIn.get(cv::CAP_PROP_FPS);

    Nframes = std::min<int>(Nframes, config::MAX_VID_FRAMES_TO_PROCESS);

    if (all_exps != NULL)
        Nframes = std::min<int>(Nframes, all_exps->size());

    std::deque<float> xlcs, xrcs;
    std::deque<float> ylcs, yrcs;

    float *h_xp = (float*)malloc( config::NPTS*sizeof(float) );
    float *h_yp = (float*)malloc( config::NPTS*sizeof(float) );
    vector<vector<float> > all_lmks;
    for (size_t fi=0; fi<Nframes; ++fi)
    {
        vector<float> lmks_combined;
        cams[0].update_camera(1.0f);

        std::vector<float> exp_coeffs, pose;
        if (NULL != all_exps) {
            if (fi >= all_exps->size())
                break;
            exp_coeffs = all_exps->at(fi);
        } else
            exp_coeffs = out.compute_avg_exp_coeffs(fi);

        if (NULL != all_poses) {
            if (fi >= all_poses->size())
                break;
            pose = all_poses->at(fi);
        } else
            pose = out.compute_avg_pose(fi);

        std::vector<float> npose(pose);

        //npose[3] = 0.1f;
        //npose[4] = 0.0f;
        //npose[5] = 0.0f;


        if (isnan(exp_coeffs[0]) || isnan(pose[0])) {
            std::cout << exp_coeffs[0] << '\t' << pose[0] << std::endl;
            for (size_t i=0; i<NLANDMARKS_51; ++i)
            {
                lmks_combined.push_back(0);
                lmks_combined.push_back(0);
            }
            all_lmks.push_back(lmks_combined);
            continue;
        }

        ov.set_frame(0);

        cudaMemcpy(ov.epsilons, &exp_coeffs[0], r.Kepsilon*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.tauxs, &npose[0], 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.u, &npose[0]+3, 3*sizeof(float), cudaMemcpyHostToDevice);

        rc.set_u_ptr(ov.u);
        rc.process();

        r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);
        cudaMemcpy(h_Xe, r.X0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ye, r.Y0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ze, r.Z0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
//        cv::Mat im3d = drawFace_fromptr(r.tl_vector, h_Xe, h_Ye, h_Ze);

        view_transform_3d_pts_and_render_2d<<<(config::NPTS+NTHREADS-1)/NTHREADS, NTHREADS>>>(r.X0, r.Y0, r.Z0,
                                                                                      rc.R, ov.taux, ov.tauy, ov.tauz,
                                                                                      cams[0].phix, cams[0].phiy, cams[0].cx, cams[0].cy,
                                                                                      d_Xr, d_Yr, d_Zr,
                                                                                      r.d_xp, r.d_yp, config::NPTS);


        cudaMemcpy(h_xp, r.d_xp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_yp, r.d_yp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<float> xp_vec, yp_vec;
        for (uint pi=0; pi<config::LIS.size(); ++pi)
        {
            xp_vec.push_back(h_xp[config::LIS[pi]]);
            yp_vec.push_back(h_yp[config::LIS[pi]]);
            lmks_combined.push_back(h_xp[config::LIS[pi]]);
            lmks_combined.push_back(h_yp[config::LIS[pi]]);
        }
        all_lmks.push_back(lmks_combined);
    }

    write_2d_vector<float>(output_landmarks_path, all_lmks);

    free( h_xp );
    free( h_yp );

    cudaFree( d_Xr );
    cudaFree( d_Yr );
    cudaFree( d_Zr );

    free(h_Xr);
    free(h_Yr);
    free(h_Zr);

    free(h_Xe);
    free(h_Ye);
    free(h_Ze);

    return true;
}












bool VideoFitter::output_landmarks_expression_variation(VideoOutput& out, std::string& input_path, std::string& output_landmarks_txt,
                                  vector<vector<float> >* all_exps, vector<vector<float> >* all_poses)
{
    cv::VideoCapture vidIn(input_path);

    float *h_Xe = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ye = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ze = (float*)malloc(config::NPTS*sizeof(float));

    float *h_Xm = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ym = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Zm = (float*)malloc(config::NPTS*sizeof(float));

    cv::Mat frame;


    int Nframes = vidIn.get(cv::CAP_PROP_FRAME_COUNT);
    FPS = vidIn.get(cv::CAP_PROP_FPS);

    Nframes = std::min<int>(Nframes, config::MAX_VID_FRAMES_TO_PROCESS);

    if (all_exps != NULL)
        Nframes = std::min<int>(Nframes, all_exps->size());

    std::vector<std::vector<float> > landmarks;

    for (size_t fi=0; fi<Nframes; ++fi)
    {
        cams[0].update_camera(1.0f);

        std::vector<float> exp_coeffs, pose;
        if (NULL != all_exps) {
            if (fi >= all_exps->size())
                break;
            exp_coeffs = all_exps->at(fi);
        } else
            exp_coeffs = out.compute_avg_exp_coeffs(fi);

        if (NULL != all_poses) {
            if (fi >= all_poses->size())
                break;
            pose = all_poses->at(fi);
        } else
            pose = out.compute_avg_pose(fi);

        std::vector<float> npose(pose);

        npose[3] = 0.1f;
        npose[4] = 0.0f;
        npose[5] = 0.0f;

        vidIn >> frame;

        if (isnan(exp_coeffs[0]) || isnan(pose[0])) 
        {
            std::vector<float> clandmarks;
            for (uint pi=0; pi<NLANDMARKS_51; ++pi)
            {
                clandmarks.push_back(NAN);
                clandmarks.push_back(NAN);
                clandmarks.push_back(NAN);
            }
            landmarks.push_back(clandmarks);
            continue;
        }

        ov.set_frame(0);
        cudaMemcpy(ov.epsilons, &exp_coeffs[0], r.Kepsilon*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.tauxs, &npose[0], 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.u, &npose[0]+3, 3*sizeof(float), cudaMemcpyHostToDevice);

        rc.set_u_ptr(ov.u);
        rc.process();

        r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);
        cudaMemcpy(h_Xe, r.X0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ye, r.Y0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ze, r.Z0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(h_Xm, r.X0_mean, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ym, r.Y0_mean, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Zm, r.Z0_mean, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> clandmarks;
        for (uint pi=0; pi<NLANDMARKS_51; ++pi)
        {
            float dx = h_Xe[config::LIS[pi]]-h_Xm[config::LIS[pi]];
            float dy = h_Ye[config::LIS[pi]]-h_Ym[config::LIS[pi]];
            float dz = h_Ze[config::LIS[pi]]-h_Zm[config::LIS[pi]];
            clandmarks.push_back(dx);
            clandmarks.push_back(dy);
            clandmarks.push_back(dz);
        }

        landmarks.push_back(clandmarks);
    }

    write_2d_vector<float>(output_landmarks_txt, landmarks);

    free(h_Xm);
    free(h_Ym);
    free(h_Zm);

    free(h_Xe);
    free(h_Ye);
    free(h_Ze);

    return true;
}















bool VideoFitter::visualize_3dmesh(VideoOutput& out,
                                   std::string& input_path, std::string& output_path,
                                   vector<vector<float> >* all_exps, vector<vector<float> >* all_poses)
{
    cv::VideoCapture vidIn(input_path);
    cv::VideoWriter vidOut;

    if (output_path != "") {
        if (!vidOut.isOpened()) {
            initGLwindow();

            vidOut.open(output_path, cv::VideoWriter::fourcc('D','I','V','X'), FPS, cv::Size(3240,1080), true);
        }
    }


    size_t T = out.exp_coefs.size();

    float *h_Xe = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ye = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Ze = (float*)malloc(config::NPTS*sizeof(float));

    float *h_Xr = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Yr = (float*)malloc(config::NPTS*sizeof(float));
    float *h_Zr = (float*)malloc(config::NPTS*sizeof(float));

    float *d_Xr, *d_Yr, *d_Zr;

    cudaMalloc((void**)&d_Xr,  sizeof(float)*config::NPTS);
    cudaMalloc((void**)&d_Yr,  sizeof(float)*config::NPTS);
    cudaMalloc((void**)&d_Zr,  sizeof(float)*config::NPTS);

    cv::Mat frame;


    int Nframes = vidIn.get(cv::CAP_PROP_FRAME_COUNT);
    FPS = vidIn.get(cv::CAP_PROP_FPS);

    Nframes = std::min<int>(Nframes, config::MAX_VID_FRAMES_TO_PROCESS);

    if (all_exps != NULL)
        Nframes = std::min<int>(Nframes, all_exps->size());


    for (size_t fi=0; fi<Nframes; ++fi)
    {
        std::vector<float> exp_coeffs, pose;
        if (NULL != all_exps) {
            if (fi >= all_exps->size())
                break;
            exp_coeffs = all_exps->at(fi);
        } else
            exp_coeffs = out.compute_avg_exp_coeffs(fi);

        if (NULL != all_poses) {
            if (fi >= all_poses->size())
                break;
            pose = all_poses->at(fi);
        } else
            pose = out.compute_avg_pose(fi);


        vidIn >> frame;

	if (!config::PREPEND_BLANK_FRAMES)
	{
            if (fi < config::SKIP_FIRST_N_SECS*FPS) {
                continue;
            }
	}

        if (isnan(exp_coeffs[0]) || isnan(pose[0])) {
            cv::Mat blank(1080, 3240, CV_8UC3, cv::Scalar::all(255));
            vidOut << blank;
            continue;
        }

        ov.set_frame(0);
        cudaMemcpy(ov.epsilons, &exp_coeffs[0], r.Kepsilon*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.u, &pose[0]+3, 3*sizeof(float), cudaMemcpyHostToDevice);
        rc.set_u_ptr(ov.u);
        rc.process();


        r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);
        cudaMemcpy(h_Xe, r.X0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ye, r.Y0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ze, r.Z0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cv::Mat im3d = drawFace_fromptr(r.tl_vector, h_Xe, h_Ye, h_Ze);
        //r.compute_nonrigid_shape_identity_and_rotation(handle, ov, rc.R, d_Xr, d_Yr, d_Zr);
        // r.compute_nonrigid_shape_expression_and_rotation(handle, ov, rc.R, d_Xr, d_Yr, d_Zr);
        rotate_3d_pts<<<(config::NPTS+NTHREADS-1)/NTHREADS, NTHREADS>>>(r.X0, r.Y0, r.Z0, rc.R, config::NPTS);

        cudaMemcpy(h_Xe, r.X0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ye, r.Y0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ze, r.Z0, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_Xr, d_Xr, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_Yr, d_Yr, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_Zr, d_Zr, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);


        //cv::Mat im3d_r = drawFace_fromptr(r.tl_vector, h_Xr, h_Yr, h_Zr);
        cv::Mat im3d_r = drawFace_fromptr(r.tl_vector, h_Xe, h_Ye, h_Ze);
        cv::cvtColor(im3d, im3d, cv::COLOR_RGBA2RGB);
        cv::cvtColor(im3d_r, im3d_r, cv::COLOR_RGBA2RGB);

        const int buffer = 0; /* 100*/
        int cmin_x = std::max<int>(0, min_x);
        int cmin_y = std::max<int>(0, min_y);
        int facerect_width = ((max_x+buffer)-(cmin_x-buffer));
        int facerect_height = ((max_y+buffer)-(cmin_y-buffer));
        facerect_width = std::min<int>(facerect_width, frame.cols-cmin_x);
        facerect_height = std::min<int>(facerect_height, frame.rows-cmin_y);

        cv::Mat face_mat = frame(cv::Rect((cmin_x-buffer), (cmin_y-buffer), facerect_width, facerect_height)).clone();

        cv::resize(face_mat, face_mat, cv::Size(1080, 1080));
        cv::resize(im3d, im3d, cv::Size(1080, 1080));
        cv::resize(im3d_r, im3d_r, cv::Size(1080, 1080));

        cv::Mat tmp_combined, combined;
        cv::hconcat(face_mat, im3d_r, tmp_combined);
        cv::hconcat(tmp_combined, im3d, combined);

        vidOut << combined;
    }

    cudaFree( d_Xr );
    cudaFree( d_Yr );
    cudaFree( d_Zr );

    free(h_Xr);
    free(h_Yr);
    free(h_Zr);

    free(h_Xe);
    free(h_Ye);
    free(h_Ze);

    return true;
}





bool VideoFitter::visualize_texture(VideoOutput &out, std::string &input_path, std::string &output_path, vector<vector<float> >* all_exps,
                                    vector<vector<float> >* all_poses, vector<vector<float> > *all_illums)
{
    cv::VideoCapture vidIn(input_path);
    cv::VideoWriter vidOut;

    cv::Mat frame;

    int Nframes = vidIn.get(cv::CAP_PROP_FRAME_COUNT);
    FPS = vidIn.get(cv::CAP_PROP_FPS);

    int width = vidIn.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = vidIn.get(cv::CAP_PROP_FRAME_HEIGHT);


    if (output_path != "") {
        if (!vidOut.isOpened()) {
            vidOut.open(output_path, cv::VideoWriter::fourcc('M','J','P','G'), FPS, cv::Size(2*width, height), false);
        }
    }

    float h_lambdas[3] = {-7.3627f, 51.1364f, 100.1784f};
    float h_Lintensity = 1000.0f;

    cv::Mat white_bg(DIMY, DIMX, CV_32FC1, cv::Scalar::all(1));

    Nframes = std::min<int>(Nframes, config::MAX_VID_FRAMES_TO_PROCESS);

    if (all_exps != NULL)
        Nframes = std::min<int>(Nframes, all_exps->size());

    float *h_xp = (float*)malloc( config::NPTS*sizeof(float) );
    float *h_yp = (float*)malloc( config::NPTS*sizeof(float) );

    std::vector<int> xoffs, yoffs;

    r.compute_texture(handle, ov, o);
    for (size_t fi=0; fi<Nframes; ++fi)
    {
        std::vector<float> exp_coeffs, pose, illum;
        if (NULL != all_exps)
            exp_coeffs = all_exps->at(fi);
        else
            exp_coeffs = out.compute_avg_exp_coeffs(fi);

        if (NULL != all_poses) {
            if (fi >= all_poses->size())
                break;
            pose = all_poses->at(fi);
        } else
            pose = out.compute_avg_pose(fi);

        if (NULL != all_illums) {
            if (fi >= all_illums->size())
                break;
            illum = all_illums->at(fi);
        } else {
            std::vector<float> tmp_illum = out.get_illum(fi);
//            illum = out.compute_avg_illum_last_k_frames(fi, 160);
            illum = out.get_illum(fi);
        }

        vidIn >> frame;

        if (cams[0].cam_remap) 
            cams[0].undistort(frame, frame);

        if (frame.channels() == 3)
            cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);

        cv::Mat result(height, 2*width, CV_8U, cv::Scalar::all(255));

        if (!config::PREPEND_BLANK_FRAMES)
        {
            if (fi < config::SKIP_FIRST_N_SECS*FPS) {
                continue;
            }
        }

        if (isnan(exp_coeffs[0]) || exp_coeffs[0] == 0.0f) {
            vidOut << result;
            continue;
        }

//        std::vector<float> illum_coefs = out.illum_coeffs[fi];


        ov.set_frame(0);
        cudaMemcpy(ov.epsilons, &exp_coeffs[0], r.Kepsilon*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.taux, &pose[0], 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.u, &pose[0]+3, 3*sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(ov.lambda, &illum[0], sizeof(float)*3, cudaMemcpyHostToDevice);
        cudaMemcpy(ov.Lintensity, &illum[0]+3, sizeof(float), cudaMemcpyHostToDevice);
//        cudaMemcpy(ov.lambda, &h_lambdas[0], sizeof(float)*3, cudaMemcpyHostToDevice);
//        cudaMemcpy(ov.Lintensity, &h_Lintensity, sizeof(float), cudaMemcpyHostToDevice);
        //print_vector(ov.epsilons, r.Kepsilon, "eps");
        //print_vector(ov.taux, 6, "pose");

        rc.set_u_ptr(ov.u);
        rc.process();

        r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);

        cudaMemcpy(h_xp, r.d_xp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_yp, r.d_yp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        int minx = (int)(*std::min_element(h_xp, h_xp+config::NPTS));
        int maxx = (int)(*std::max_element(h_xp, h_xp+config::NPTS));

        float facew = (float) (maxx-minx);

        if (facew > 120)
        {
            cams[0].update_camera(config::REF_FACE_SIZE/facew);
            r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);

            cudaMemcpy(h_xp, r.d_xp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_yp, r.d_yp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        }

        r.set_x0_short_y0_short(0, h_xp, h_yp, config::NPTS, false);

        xoffs.push_back((int)(*std::min_element(h_xp, h_xp+config::NPTS)/cams[0].resize_coef));
        xoffs.push_back((int)(*std::max_element(h_xp, h_xp+config::NPTS)/cams[0].resize_coef));
        yoffs.push_back((int)(*std::min_element(h_yp, h_yp+config::NPTS)/cams[0].resize_coef));
        yoffs.push_back((int)(*std::max_element(h_yp, h_yp+config::NPTS)/cams[0].resize_coef));

        ushort Nunique;
        cudaMemcpy(r.d_texIm, white_bg.data, sizeof(float)*DIMX*DIMY, cudaMemcpyHostToDevice);
        r.render(0, o, ov, rc.R, handle, &Nunique, d_cropped_face, d_buffer_face, false, false);

        int cur_width = std::round(((double) width)*cams[0].resize_coef);
        int cur_height = std::round(((double) height)*cams[0].resize_coef);

        cv::Mat rtexture(cur_height, cur_width, CV_32F, cv::Scalar::all(1));

        int x_offset = r.x0_short[0];
        int y_offset = r.y0_short[0];
        cv::Mat rtexture_uchar;

        {
            cv::Mat rectMat(DIMX, DIMY, CV_32F);


            cudaMemcpy(rectMat.data, r.d_texIm, sizeof(float)*NTOTAL_PIXELS, cudaMemcpyDeviceToHost);
            cv::transpose(rectMat, rectMat);

            int rect_width = DIMX;
            int rect_height = DIMY;

            if (rtexture.rows - y_offset < DIMY) {
                rect_height = rtexture.rows - y_offset-1;
            }

            if (rtexture.cols - x_offset < DIMX) {
                rect_width = rtexture.cols - x_offset-1;
            }

            rectMat = rectMat(cv::Rect(0,0,rect_width, rect_height));

            rectMat.copyTo(rtexture(cv::Rect(x_offset, y_offset, rect_width, rect_height)));

            double minSrc, maxSrc;
            cv::minMaxLoc(rtexture, &minSrc, &maxSrc);
            rtexture = (rtexture-minSrc)/(maxSrc-minSrc);

            cv::resize(rtexture, rtexture, cv::Size(width, height));

            rtexture = 255*rtexture;
            rtexture.convertTo(rtexture_uchar, CV_8U);



            rtexture.copyTo(result(cv::Rect(0, 0, width, height)));
            frame.copyTo(result(cv::Rect(width, 0, width, height)));
        }

        vidOut << result;

//        std::cout << fi << std::endl;
    }

    min_x = *std::min_element(xoffs.begin(), xoffs.end());
    min_y = *std::min_element(yoffs.begin(), yoffs.end());
    max_x = *std::max_element(xoffs.begin(), xoffs.end());
    max_y = *std::max_element(yoffs.begin(), yoffs.end());

    free(h_xp);
    free(h_yp);
    return false;
}





bool VideoFitter::generate_texture(int subj_id, int imwidth, const std::string& out_dir_root, const float fov_data, const float tz)
{
    std::stringstream ss0;
    ss0 << out_dir_root << "/" << fov_data << "_" << tz;

    std::string out_dir = ss0.str();

    std::cout << out_dir_root << std::endl;
    if (!std::experimental::filesystem::exists(out_dir_root))
        std::experimental::filesystem::create_directory(out_dir_root);

    if (!std::experimental::filesystem::exists(out_dir))
        std::experimental::filesystem::create_directory(out_dir);
    
    cv::Mat frame;

    int Nframes = 100;

    int width = imwidth;
    int height = imwidth;

    float h_lambdas[3] = {-7.3627f, 51.1364f, 100.1784f};
    float h_Lintensity = 0.0005f;

    cv::Mat bgim = cv::imread("./data/bg2.png", cv::IMREAD_GRAYSCALE);
    bgim.convertTo(bgim, CV_32FC1);
    bgim /= 255.0f;

    cv::Mat bgim_tex = cv::imread("./data/bg2.png", cv::IMREAD_GRAYSCALE);
    bgim_tex = bgim_tex(cv::Rect(0,0,DIMX, DIMY));
    bgim_tex.convertTo(bgim_tex, CV_32FC1);
    bgim_tex /= 255.0f;

    Nframes = std::min<int>(Nframes, config::MAX_VID_FRAMES_TO_PROCESS);

    float *h_xp = (float*)malloc( config::NPTS*sizeof(float) );
    float *h_yp = (float*)malloc( config::NPTS*sizeof(float) );

    std::vector<int> xoffs, yoffs;
    std::default_random_engine generator(1907);
    std::uniform_real_distribution<float> distribution(-0.75f,0.75f);

    for (size_t fi=0; fi<Nframes; ++fi)
    {
        float u1 = distribution(generator);
        float u2 = distribution(generator);
        float u3 = distribution(generator);


        cams[0].update_camera(1.0f);
        float pose[6] = {0, 0, tz, u1, u2, 0};

        float exp_coeffs[config::K_EPSILON];

        for (size_t ei=0; ei<config::K_EPSILON; ++ei)
            exp_coeffs[ei] = 0.0f;

        ov.set_frame(0);
        cudaMemcpy(ov.epsilons, exp_coeffs, r.Kepsilon*sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(ov.taux, &pose[0], 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ov.u, &pose[0]+3, 3*sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(ov.lambda, h_lambdas, sizeof(float)*3, cudaMemcpyHostToDevice);
        cudaMemcpy(ov.Lintensity, &h_Lintensity, sizeof(float), cudaMemcpyHostToDevice);

        rc.set_u_ptr(ov.u);
        rc.process();

        r.compute_texture(handle, ov, o);
        bool is_face_ok = r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);
        if (!is_face_ok) {
            std::cout << "face problematic -- skipping " << std::endl;
        }


        //!print_vector(r.d_xp, config::NPTS, "dxp");
        //!print_vector(r.d_yp, config::NPTS, "");


        cudaMemcpy(h_xp, r.d_xp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_yp, r.d_yp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        int minx = (int)(*std::min_element(h_xp, h_xp+config::NPTS));
        int maxx = (int)(*std::max_element(h_xp, h_xp+config::NPTS));

        float facew = (float) (maxx-minx);

        if (facew > 120)
        {
            cams[0].update_camera(config::REF_FACE_SIZE/facew);
            r.compute_nonrigid_shape2(handle, ov, rc.R, cams[0]);

            cudaMemcpy(h_xp, r.d_xp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_yp, r.d_yp, config::NPTS*sizeof(float), cudaMemcpyDeviceToHost);
        }

        r.set_x0_short_y0_short(0, h_xp, h_yp, config::NPTS, false);

        xoffs.push_back((int)(*std::min_element(h_xp, h_xp+config::NPTS)/cams[0].resize_coef));
        xoffs.push_back((int)(*std::max_element(h_xp, h_xp+config::NPTS)/cams[0].resize_coef));
        yoffs.push_back((int)(*std::min_element(h_yp, h_yp+config::NPTS)/cams[0].resize_coef));
        yoffs.push_back((int)(*std::max_element(h_yp, h_yp+config::NPTS)/cams[0].resize_coef));

        ushort Nunique;
        cudaMemcpy(r.d_texIm, bgim_tex.data, sizeof(float)*DIMX*DIMY, cudaMemcpyHostToDevice);
        r.render(0, o, ov, rc.R, handle, &Nunique, d_cropped_face, d_buffer_face, false, false);

        int cur_width = std::round(((double) width)*cams[0].resize_coef);
        int cur_height = std::round(((double) height)*cams[0].resize_coef);

        cv::Mat rtexture = bgim.clone(); //cur_height, cur_width, CV_32F, cv::Scalar::all(0));
        cv::resize(rtexture, rtexture, cv::Size(cur_width, cur_height));


        int x_offset = r.x0_short[0];
        int y_offset = r.y0_short[0];
        cv::Mat rtexture_uchar;

        {
            cv::Mat rectMat(DIMX, DIMY, CV_32F);

            cudaMemcpy(rectMat.data, r.d_texIm, sizeof(float)*NTOTAL_PIXELS, cudaMemcpyDeviceToHost);
            cv::transpose(rectMat, rectMat);

            int rect_width = DIMX;
            int rect_height = DIMY;

            if (rtexture.rows - y_offset < DIMY) {
                rect_height = rtexture.rows - y_offset-1;
            }

            if (rtexture.cols - x_offset < DIMX) {
                rect_width = rtexture.cols - x_offset-1;
            }

            rectMat = rectMat(cv::Rect(0,0,rect_width, rect_height));

            rectMat.copyTo(rtexture(cv::Rect(x_offset, y_offset, rect_width, rect_height)));

            double minSrc, maxSrc;
            cv::minMaxLoc(rtexture, &minSrc, &maxSrc);
            rtexture = (rtexture-minSrc)/(maxSrc-minSrc);

            cv::resize(rtexture, rtexture, cv::Size(width, height));
//            cv::imshow("rtexture", rtexture);
//            cv::waitKey(0);
            rtexture *= 255;
            rtexture.convertTo(rtexture_uchar, CV_8U);

        }
        std::stringstream ss;
        ss << out_dir << "/id" << std::setfill('0') << std::setw(3) << subj_id << "_" << std::setw(3) << std::setfill('0') << fi << ".jpg";
        cv::imwrite(ss.str(), rtexture_uchar);

        std::cout << fi << std::endl;
    }

    min_x = *std::min_element(xoffs.begin(), xoffs.end());
    min_y = *std::min_element(yoffs.begin(), yoffs.end());
    max_x = *std::max_element(xoffs.begin(), xoffs.end());
    max_y = *std::max_element(yoffs.begin(), yoffs.end());

    free(h_xp);
    free(h_yp);
    /*
    */
    return false;

}






bool VideoFitter::learn_identity(const std::string& filepath, LandmarkData& ld, float *h_alphas, float *h_betas)
{

    std::vector<std::vector<float> > selected_frame_xps, selected_frame_yps;
    std::vector<std::vector<float> > selected_frame_xranges, selected_frame_yranges;
    std::vector<cv::Mat> selected_frames;

    fit_video_frames_landmarks_sparse(filepath, ld, selected_frame_xps, selected_frame_yps,
                                      selected_frame_xranges, selected_frame_yranges, selected_frames);

    int num_used_recs = 0;


    Camera &cam0 = cams[0];

    for (uint num_rec=0; num_rec<config::NTOT_RECONSTRS; ++num_rec)
    {
        if (config::PRINT_DEBUG)
            std::cout << num_rec << std::endl;
        cam0.update_camera(1.0f);
        size_t fstart = num_rec*config::NFRAMES;
        size_t fend = fstart + config::NFRAMES;

        if (fend > selected_frames.size())
            break;

        // To store the sliced vector
        vector<vector<float> > cxps(selected_frame_xps.begin()+fstart, selected_frame_xps.begin()+fend);
        vector<vector<float> > cyps(selected_frame_yps.begin()+fstart, selected_frame_yps.begin()+fend);

        vector<vector<float> > cxranges(selected_frame_xranges.begin()+fstart, selected_frame_xranges.begin()+fend);
        vector<vector<float> > cyranges(selected_frame_yranges.begin()+fstart, selected_frame_yranges.begin()+fend);

        vector<Mat> cframes(selected_frames.begin()+fstart, selected_frames.begin()+fend);

        float *h_X0_cur, *h_Y0_cur, *h_Z0_cur, *h_tex_mu_cur, *h_alphas_cur, *h_betas_cur;

        h_alphas_cur = (float*)malloc( ov.Kalpha*sizeof(float) );
        h_betas_cur = (float*)malloc( ov.Kbeta*sizeof(float) );

        h_X0_cur = (float*)malloc( config::NPTS*sizeof(float) );
        h_Y0_cur = (float*)malloc( config::NPTS*sizeof(float) );
        h_Z0_cur = (float*)malloc( config::NPTS*sizeof(float) );
        h_tex_mu_cur = (float*)malloc( config::NPTS*sizeof(float) );

        bool success = fit_multiframe(cxps, cyps, cxranges, cyranges, cframes, h_X0_cur, h_Y0_cur, h_Z0_cur, h_tex_mu_cur);

        if (success) {
            if (config::PRINT_DEBUG)
                std::cout << "Success" << std::endl;
            num_used_recs += 1;
            cudaMemcpy(h_alphas_cur, ov.alphas, sizeof(float)*ov.Kalpha, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_betas_cur, ov.betas,  sizeof(float)*ov.Kbeta, cudaMemcpyDeviceToHost);

            for (size_t ai=0; ai<ov.Kalpha; ++ai)
            {
                if (num_used_recs == 1)
                    h_alphas[ai] = h_alphas_cur[ai];
                else
                    h_alphas[ai] += h_alphas_cur[ai];
            }

            for (size_t bi=0; bi<ov.Kbeta; ++bi)
            {
                if (num_used_recs == 1)
                    h_betas[bi] = h_betas_cur[bi];
                else
                    h_betas[bi] += h_betas_cur[bi];
            }
        }

        free(h_tex_mu_cur);
        free(h_X0_cur);
        free(h_Y0_cur);
        free(h_Z0_cur);

        free(h_alphas_cur);
        free(h_betas_cur);
    }

    if (num_used_recs == 0)
        return false;

    float weight_ = ((float) 1./((float) num_used_recs)) ;

    for (size_t ai=0; ai<ov.Kalpha; ++ai)
        h_alphas[ai] *= weight_;

    for (size_t bi=0; bi<ov.Kbeta; ++bi)
        h_betas[bi] *= weight_;

    return true;
}






bool VideoFitter::fit_multiframe(const std::vector<std::vector<float> >& selected_frame_xps, const  std::vector<std::vector<float> >& selected_frame_yps,
                                 const std::vector<std::vector<float> >& selected_frame_xranges, const  std::vector<std::vector<float> >& selected_frame_yranges,
                                 const std::vector< cv::Mat >& selected_frames,
                                 float *h_X0, float *h_Y0, float *h_Z0, float *h_tex_mu, std::vector<std::string>* result_basepaths)
{
    // <!-- We'll probably need to change the below --> // we'll need to create one camera per frame, e.g., a vector of Camera's
    std::vector<float> face_sizes;
    for (size_t fi=0; fi<selected_frames.size(); ++fi)
    {
        float face_size = (float) compute_face_size(&selected_frame_xps[fi][0], &selected_frame_yps[fi][0]);
        face_sizes.push_back(face_size);
    }

    for (size_t i=0; i<cams.size(); ++i) {
        cams[i].update_camera(config::REF_FACE_SIZE/face_sizes[i]);
    }

    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);
    ov.set_frame(0);
    ov_linesearch.set_frame(0);

    cudaEvent_t     start, stop;
    // rendered expression basis

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );


    if (config::PRINT_DEBUG)
        std::cout << "BEFORE MULTI" << std::endl;
    // <!-- We'll probably need to change the below --> // we'll pass cams instead of cam
    bool success = fit_to_multi_images(cams, selected_frame_xps, selected_frame_yps,
                                       selected_frame_xranges, selected_frame_yranges, selected_frames,
                                       result_basepaths, r,  o, handleDn,
                                       handle, d_cropped_face, d_buffer_face, ov, ov_linesearch, ov_lb, ov_lb_linesearch, rc,
                                       rc_linesearch,  dc, s, s_lambda, d_tmp, search_dir_Lintensity, dg_Lintensity,
                                       h_X0, h_Y0, h_Z0, h_tex_mu);
    if (config::PRINT_DEBUG)
        std::cout << "AFTER MULTI" << std::endl;


    // get stop time, and display the timing results
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    if (config::PRINT_DEBUG)
        printf( "Time to generate:  %3.1f ms\n", elapsedTime );




    return success;

}




int VideoFitter::fit_video_frames_landmarks_sparse(const std::string& filepath,
                                                   LandmarkData& ld,
                                                   std::vector<std::vector<float> >& selected_frame_xps,
                                                   std::vector<std::vector<float> >& selected_frame_yps,
                                                   std::vector<std::vector<float> >& selected_frame_xranges,
                                                   std::vector<std::vector<float> >& selected_frame_yranges,
                                                   std::vector< cv::Mat >& selected_frames)
{
    Camera& cam0 = cams[0];
    cam0.update_camera(1.0f);

    cv::VideoCapture capture(filepath);
    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    FPS = capture.get(cv::CAP_PROP_FPS);
    cv::Mat frame;

    vector< vector<float> > all_xp_vec, all_yp_vec;
    vector< vector<float> > all_xrange, all_yrange;

    vector<float> face_sizes;
    vector<float> minxs, minys, maxxs, maxys;

    int Ntotframes = capture.get(cv::CAP_PROP_FRAME_COUNT);
    FPS = capture.get(cv::CAP_PROP_FPS);

    Ntotframes = std::min<int>(Ntotframes, ld.get_num_frames());

    int N_FRAMES_TO_CONSIDER = 800;
    int EVERY_N_FRAMES = 1;

    if (Ntotframes > N_FRAMES_TO_CONSIDER)
        EVERY_N_FRAMES = (int) (Ntotframes/N_FRAMES_TO_CONSIDER);

    int idx = 0;
    while (true)
    {
        idx++;
//        all_xp_vec.push_back(ld.);

        if ((idx-1) >= ld.xp_vecs.size())
            break;

        all_xp_vec.push_back(ld.get_xpvec(idx-1));
        all_yp_vec.push_back(ld.get_ypvec(idx-1));

        all_xrange.push_back(std::vector<float>());
        all_yrange.push_back(std::vector<float>());

        capture >> frame;

        if (frame.empty())
            break;

        if (cam0.cam_remap) {
            cv::remap(frame, frame, cam0.map1, cam0.map2, cv::INTER_LINEAR);
        }

        if (idx == 1) {
            min_x = frame.cols;
            max_x = 0;
            min_y = frame.rows;
            max_y = 0;
        }

        if (idx < FPS*config::SKIP_FIRST_N_SECS)
            continue;

        // PUTBACK
//        EVERY_N_FRAMES
//        if (idx % (int)std::round(FPS*config::EVERY_N_SECS) != 0)
        if (idx % EVERY_N_FRAMES != 0)
            continue;

        if (idx >= config::MAX_VID_FRAMES_TO_PROCESS)
            break;

        std::vector<float>& xcur = all_xp_vec[all_xp_vec.size()-1];
        std::vector<float>& ycur = all_yp_vec[all_yp_vec.size()-1];

        float face_size = compute_face_size(&xcur[0], &ycur[0]);

        if (face_size == -1.0f)
            continue;

        if (xcur.size() == 0)
            continue;

        int cur_xmin = (int) *std::min_element(xcur.begin(), xcur.end());
        int cur_xmax = (int) *std::max_element(xcur.begin(), xcur.end());

        int cur_ymin = (int) *std::min_element(ycur.begin(), ycur.end());
        int cur_ymax = (int) *std::max_element(ycur.begin(), ycur.end());

        if (cur_xmin <= 0 || cur_ymin <= 0 || cur_xmax >= frame.cols || cur_ymax >= frame.rows)
            continue;

        minxs.push_back(cur_xmin);
        minys.push_back(cur_ymin);

        maxxs.push_back(cur_xmax);
        maxys.push_back(cur_ymax);

        if (cur_xmin < min_x)
            min_x = cur_xmin;

        if (cur_xmax > max_x)
            max_x = cur_xmax;

        if (cur_ymin < min_y)
            min_y = cur_ymin;

        if (cur_ymax > max_y)
            max_y = cur_ymax;

        if (config::PRINT_DEBUG)
            std::cout << idx << std::endl;

        // !PUTBACK
        //        if (idx >= 6000)
        //            break;

        face_sizes.push_back(face_size);
        if (config::PRINT_DEBUG)
           std::cout << "face size is " << face_size << std::endl;
    }

    // The lines below compute the median face size
    size_t ni = face_sizes.size() / 2;
    nth_element(face_sizes.begin(), face_sizes.begin()+ni, face_sizes.end());
    float median_face = face_sizes[ni];

    nth_element(minxs.begin(), minxs.begin()+ni, minxs.end());
    nth_element(minys.begin(), minys.begin()+ni, minys.end());
    nth_element(maxxs.begin(), maxxs.begin()+ni, maxxs.end());
    nth_element(maxys.begin(), maxys.begin()+ni, maxys.end());

    float median_face_width = maxxs[ni]-minxs[ni];
    float median_face_height = maxys[ni]-minys[ni];

    min_x = minxs[ni]-median_face_width/1.5;
    max_x = maxxs[ni]+median_face_width/1.5;

    min_y = minys[ni]-median_face_height/1.5;
    max_y = maxys[ni]+median_face_height/1.5;

    //    RESIZE_COEF = 88.0f/median_face;
    //    RESIZE_COEF = 92.0f/median_face;
    //////////////////    RESIZE_COEF = 72.0f/median_face;

    cam0.update_camera(76.0f/median_face);

    std::vector<Camera> cams0;
    cams0.push_back(Camera(cam0));

    uint Nframes = idx;

    capture.release();
    capture.open(filepath);

    //	uint Nframes = 600; // capture.get(cv::CAP_PROP_FRAME_COUNT);


    const ushort Kalpha = config::K_ALPHA_L;
    const ushort Kbeta = config::K_BETA_L;
    const ushort Kepsilon = config::K_EPSILON_L;

    const bool use_identity = true;
    const bool use_texture = true;
    const bool use_expression = true;

    Renderer r(1, Kalpha, Kbeta, Kepsilon, use_identity, use_texture, use_expression);


    OptimizationVariables ov_lb(1, Kalpha, Kbeta, Kepsilon, use_identity, use_texture, use_expression, true);
    OptimizationVariables ov_lb_linesearch(1, Kalpha, Kbeta, Kepsilon, use_identity, use_texture, use_expression, true);

    Logbarrier_Initializer li_init(&cams0, &ov_lb, handleDn, 1.00f, use_identity, use_texture, use_expression, r, true);

    ov_lb.set_frame(0);
    ov_lb_linesearch.set_frame(0);

    RotationComputer rc(ov_lb.u);
    RotationComputer rc_linesearch(ov_lb_linesearch.u);

    Solver s(handleDn, ov_lb.Ktotal);

    cudaEvent_t     start, stop;
    // rendered expression basis

    float xp[NLANDMARKS_51], yp[NLANDMARKS_51];

    std::vector<float> rolls;
    std::vector<float> yaws;
    std::vector<float> pitches;

    yaws.reserve(Nframes);
    rolls.reserve(Nframes);
    pitches.reserve(Nframes);


    uint num_angles = (int) (ANGLE_MAX-ANGLE_MIN)/ANGLE_STEP + 1;


    std::vector< std::vector<int> > angle_idx_vs_frame;
    angle_idx_vs_frame.reserve(num_angles*num_angles);

    for (uint i=0; i<(num_angles*num_angles); ++i) {
        angle_idx_vs_frame.push_back(std::vector<int>());
    }

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );


    for( uint fi=0; fi<Nframes; fi++)
    {
        if (all_xp_vec.size() <= fi || all_xp_vec[fi].size() == 0) {
            rolls.push_back(NAN);
            yaws.push_back(NAN);
            pitches.push_back(NAN);
            continue;
        }

        for (uint ii=0; ii<NLANDMARKS_51; ++ii)
        {
            xp[ii] = (float) cams0[0].resize_coef*all_xp_vec[fi][ii];
            yp[ii] = (float) cams0[0].resize_coef*all_yp_vec[fi][ii];
        }


        /*
       cv::Mat dummy(1800, 800, CV_32FC1, cv::Scalar::all(255));
       for (int i=0; i<NLANDMARKS_51; ++i)
       {
       cv::Point2f ptOrig(xp[i], yp[i]);
       cv::circle(dummy, ptOrig, 3, cv::Scalar(0,0,0), cv::FILLED, 8, 0);
       }
     */

        bool is_bb_OK = check_if_bb_OK(xp, yp);

        float face_size = compute_face_size(xp, yp);

        if (config::PRINT_DEBUG)
            std::cout << fi << "\t => " << is_bb_OK<< std::endl;

        ////////////////////////////////////////////////////////////////////////////////////
        ov_lb.reset();
        ov_lb_linesearch.reset();

        cudaMemset(ov_lb.slack, 0, sizeof(float));
        cudaMemset(ov_lb_linesearch.slack, 0, sizeof(float));

        ov_lb.reset_tau_logbarrier();
        ov_lb_linesearch.reset_tau_logbarrier();

        li_init.set_landmarks_from_host(0, xp, yp);
        li_init.initialize_with_orthographic_t(handleDn, handle, 0, xp, yp, face_size, &ov_lb);
        li_init.set_minimal_slack(handle, &ov_lb);

        li_init.fit_model(handleDn, handle, &ov_lb, &ov_lb_linesearch);

        float yaw(NAN), pitch(NAN), roll(NAN);

        if (li_init.fit_success && xp[0] != 0.0f && is_bb_OK)
        {
            int yaw_idx, pitch_idx, roll_idx;
            li_init.rc.compute_euler_angles(yaw, pitch, roll);
            li_init.rc.compute_angle_idx(yaw, pitch, roll, yaw_idx, pitch_idx, roll_idx);

            if (roll_idx != -1 && pitch_idx != -1 && yaw_idx != -1)
            {
                int angle_idx = num_angles*roll_idx + pitch_idx;
                angle_idx_vs_frame[angle_idx].push_back(fi);
            }
        }

        /*
       std::cout << "yaw " << RAD2DEG(yaw) << " - pitch " << RAD2DEG(pitch) << " roll - " << RAD2DEG(roll) << std::endl;
       cv::imshow("dummy", dummy);
       cv::waitKey(0);
     */

        rolls.push_back(roll);
        yaws.push_back(yaw);
        pitches.push_back(pitch);
    }

    std::vector<int> yaw_centroids;

    const int YAW_MAX = 60;
    const int YAW_MIN = -60;
    const int YAW_INCREMENT = 10;

    for (int c=YAW_MIN; c<=YAW_MAX; c+= YAW_INCREMENT)
        yaw_centroids.push_back((c)+YAW_INCREMENT/2);


    std::vector<int> pitch_centroids;

    const int PITCH_MAX = 60;
    const int PITCH_MIN = -60;
    const int PITCH_INCREMENT = 10;

    for (int c=PITCH_MIN; c<=PITCH_MAX; c+= PITCH_INCREMENT)
        pitch_centroids.push_back((c)+PITCH_INCREMENT/2);

    using std::vector;

    // Create vector to store frame ids, which will later be shuffled
    vector<vector<vector<int> > > vector_angles;
    for (size_t yi=0; yi<yaw_centroids.size(); ++yi) {
        vector<vector<int> > tmp;
        for (size_t pi=0; pi<pitch_centroids.size(); ++pi) {
            tmp.push_back(vector<int>());
        }
        vector_angles.push_back(tmp);
    }

    for (size_t fi=0; fi<Nframes; ++fi) {
        if (isnan(yaws[fi]))
            continue;

        //        std::cout << "yaw/pitch " << yaws[fi] << '\t' << pitches[fi] << std::endl;

        std::vector<double> pitch_dist, yaw_dist;
        for (size_t pi=0; pi<pitch_centroids.size(); ++pi)
            pitch_dist.push_back(fabs(pitch_centroids[pi]-RAD2DEG(pitches[fi])));

        for (size_t yi=0; yi<yaw_centroids.size(); ++yi)
            yaw_dist.push_back(fabs(yaw_centroids[yi]-RAD2DEG(yaws[fi])));

        size_t pitch_idx = std::distance(pitch_dist.begin(), std::min_element(pitch_dist.begin(), pitch_dist.end()));
        size_t yaw_idx = std::distance(yaw_dist.begin(), std::min_element(yaw_dist.begin(), yaw_dist.end()));

        vector_angles[yaw_idx][pitch_idx].push_back(fi);
    }

    std::default_random_engine e_seed(1907);

    std::vector<size_t> selected_frame_idx;

    // pick frames random subsets of angles. Limit number of angle bin to 14 (see below)
    for (size_t yi=0; yi<yaw_centroids.size(); ++yi) {
        for (size_t pi=0; pi<pitch_centroids.size(); ++pi) {
            // @@@
            std::shuffle(vector_angles[yi][pi].begin(), vector_angles[yi][pi].end(), e_seed);

            for (size_t ci=0; ci<(size_t)std::min<int>(vector_angles[yi][pi].size(), config::NFRAMES_PER_ANGLE_BIN); ++ci)
                selected_frame_idx.push_back(vector_angles[yi][pi][ci]);
        }
    }


    std::set<int> selected_frame_idx_set(selected_frame_idx.begin(), selected_frame_idx.end());


    vector<int> tmpidx;

    vector<cv::Mat> sorted_frames;
    vector<vector<float> > sorted_frame_xps, sorted_frame_yps;
    vector<vector<float> > sorted_frame_xranges, sorted_frame_yranges;

    cam0.update_camera(1.0f);
    for( uint t=0; t<Nframes; ++t)
    {
        capture >> frame;
        if(frame.empty())
            break;

        if (cam0.cam_remap) {
            cv::remap(frame, frame, cam0.map1, cam0.map2, cv::INTER_LINEAR);
        }

        const bool is_in = selected_frame_idx_set.find(t) != selected_frame_idx_set.end();

        if (!is_in)
            continue;

        sorted_frames.push_back(frame.clone());
        sorted_frame_xps.push_back(all_xp_vec[t]);
        sorted_frame_yps.push_back(all_yp_vec[t]);
        sorted_frame_xranges.push_back(all_xrange[t]);
        sorted_frame_yranges.push_back(all_yrange[t]);
    }



    std::vector<int> ivec;

    for (size_t num_perms=1; num_perms<=config::NPERMS; ++num_perms) {
        std::vector<int> _ivec(sorted_frames.size());
        std::iota(_ivec.begin(), _ivec.end(), 0); // ivec will become: [0..99..]
        // @@@
        std::shuffle(_ivec.begin(), _ivec.end(), e_seed);
        ivec.insert(ivec.end(), _ivec.begin(), _ivec.end());
    }

    vector<cv::Mat> tmp_frames;
    vector<vector<float> > tmp_frame_xps, tmp_frame_yps;
    vector<vector<float> > tmp_frame_xranges, tmp_frame_yranges;

    if (config::PRINT_DEBUG)
    {
        std::cout << "selected frames are: ";
        for (auto pp : ivec)
            std::cout << pp << " ";
        std::cout << std::endl;
    }


    for (size_t ri : ivec)
    {
        tmp_frames.push_back(sorted_frames[ri].clone());
        tmp_frame_xps.push_back(sorted_frame_xps[ri]);
        tmp_frame_yps.push_back(sorted_frame_yps[ri]);
        tmp_frame_xranges.push_back(sorted_frame_xranges[ri]);
        tmp_frame_yranges.push_back(sorted_frame_yranges[ri]);

        if (tmp_frames.size() == (size_t) config::NFRAMES)
        {
            std::vector<Camera> cams;
            for (size_t ti=0; ti<tmp_frames.size(); ++ti) {
                cams.push_back(Camera(cam0));
            }

            bool fit_success = fit_to_multi_images_landmarks_only(
                        tmp_frame_xps, tmp_frame_yps,
                        tmp_frame_xranges, tmp_frame_yranges,
                        r, handleDn, handle, rc, rc_linesearch, cams);

            if (fit_success)
            {
                for (size_t ci=0; ci<tmp_frames.size(); ++ci) {
                    selected_frames.push_back(tmp_frames[ci].clone());
                    selected_frame_xps.push_back(tmp_frame_xps[ci]);
                    selected_frame_yps.push_back(tmp_frame_yps[ci]);
                    selected_frame_xranges.push_back(tmp_frame_xranges[ci]);
                    selected_frame_yranges.push_back(tmp_frame_yranges[ci]);
                }
            }

            tmp_frames.clear();
            tmp_frame_xps.clear();
            tmp_frame_yps.clear();
            tmp_frame_xranges.clear();
            tmp_frame_yranges.clear();
        }

        // prevent overloading memory with frames
        if (selected_frames.size() >= config::NMAX_FRAMES) {
            break;
        }

        // prevent overloading memory with frames
        if (selected_frames.size() >= config::NTOT_RECONSTRS*config::NFRAMES) {
            break;
        }
    }

    if (config::PRINT_DEBUG)
    {
        std::cout << "We selected " << selected_frames.size() << " frames" << std::endl;
        std::cout << "We selected " << selected_frames.size() << " frames" << std::endl;
        std::cout << "We selected " << selected_frames.size() << " frames" << std::endl;
        std::cout << "We selected " << selected_frames.size() << " frames" << std::endl;
    }
    /*
    writeArrFile<float>(&yaws[0], "yaws.dat", Nframes, 1, false);
    writeArrFile<float>(&pitches[0], "pitches.dat", Nframes, 1, false);
    */


    // get stop time, and display the timing results
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    capture.release();

    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    //	printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    if (config::PRINT_DEBUG)
    {
        printf( "Processing speed:  %.2f fps\n", elapsedTime/Nframes );
        std::cout << "Cleaned everything " << std::endl;
    }
    return 0;
}


VideoFitter::~VideoFitter()
{
    cusolverDnDestroy(handleDn);
    cublasDestroy(handle);

    cudaFree( search_dir_Lintensity );
    cudaFree( dg_Lintensity );
    cudaFree( d_tmp );

    cudaFree( d_cropped_face );
    cudaFree( d_buffer_face);
}





void VideoOutput::add_exp_coeffs(const size_t frame_id, const vector<float>& cur_exp_coefs)
{
    if (exp_coefs.find(frame_id) == exp_coefs.end()) {
        exp_coefs.insert(std::make_pair(frame_id, vector<vector<float>>()));
    }

    exp_coefs[frame_id].push_back(cur_exp_coefs);

    abs_frame_ids.insert(frame_id);
}


std::vector<float> VideoOutput::compute_avg_exp_coeffs(const size_t frame_id, bool ignore_first)
{
    if (exp_coefs.find(frame_id) == exp_coefs.end())
        return std::vector<float>(Kepsilon, NAN);

    std::vector<float> avgs(Kepsilon, 0);

    int K = exp_coefs[frame_id].size();

    int k0 = 0;
    if (ignore_first)
        k0 = 1;

    for (int k=k0; k<K; ++k) {
        for (int ei=0; ei<Kepsilon; ++ei)
            avgs[ei] += exp_coefs[frame_id][k][ei]/((float) (K-k0));
    }

    return avgs;
}





std::vector<float> VideoOutput::combine_exp_coeffs(const size_t frame_id)
{
    if (exp_coefs.find(frame_id) == exp_coefs.end())
        return std::vector<float>(Kepsilon*config::NRES_COEFS, NAN);

    if (exp_coefs.find(frame_id) == exp_coefs.end())
        return std::vector<float>(Kepsilon*config::NRES_COEFS, NAN);

    std::vector<float> exps; //(Kepsilon*config::NRES_COEFS, 0);

    for (int k=0; k<config::NRES_COEFS; ++k) {
        for (int ei=0; ei<Kepsilon; ++ei) {
            if (k >= exp_coefs[frame_id].size()) {
                exps.push_back(NAN);
            } else {
                exps.push_back(exp_coefs[frame_id][k][ei]);
            }
        }
    }

    return exps;
}







void VideoOutput::add_pose(const size_t frame_id, const vector<float>& cur_exp_coefs)
{
    if (poses.find(frame_id) == poses.end()) {
        poses.insert(std::make_pair(frame_id, vector<vector<float>>()));
    }

    poses[frame_id].push_back(cur_exp_coefs);
}





void VideoOutput::add_illum(const size_t frame_id, const vector<float>& cur_illum_coefs)
{
    if (illum_coeffs.find(frame_id) == illum_coeffs.end()) {
        illum_coeffs.insert(std::make_pair(frame_id, cur_illum_coefs));
    }
    else
        return;
}



std::vector<float> VideoOutput::get_illum(const size_t frame_id)
{
    if (illum_coeffs.find(frame_id) == illum_coeffs.end())
        return std::vector<float>(4, NAN);
   return illum_coeffs[frame_id];
}

std::vector<float> VideoOutput::compute_avg_illum_last_k_frames(const int cur_frame_id, const int K)
{
    int t0 = std::max<int>(0, cur_frame_id-K);

    std::vector<float> out_illum(4, 0);
    size_t num_frames = 0;
    for (size_t t=t0; t<cur_frame_id; ++t) {
        std::vector<float> cur_illum = get_illum(t);

        if (!isnanf(cur_illum[0])) {
            num_frames += 1;
            out_illum[0] += cur_illum[0];
            out_illum[1] += cur_illum[1];
            out_illum[2] += cur_illum[2];
            out_illum[3] += cur_illum[3];
        }
    }

    if (num_frames > 0)
    {
        out_illum[0] /= num_frames;
        out_illum[1] /= num_frames;
        out_illum[2] /= num_frames;
        out_illum[3] /= num_frames;
    }

    return out_illum;
}

std::vector<float> VideoOutput::compute_avg_pose(const size_t frame_id)
{
    if (poses.find(frame_id) == poses.end())
        return std::vector<float>(6, NAN);

    std::vector<float> avgs(6, 0);

    int K = poses[frame_id].size();

    for (int k=0; k<K; ++k) {
        for (int ei=0; ei<6; ++ei)
            avgs[ei] += poses[frame_id][k][ei]/((float) K);
    }

    return avgs;
}

void VideoOutput::save_expressions(const std::string& filepath, bool ignore_first)
{
    int MAX_FRAMEID = *std::max_element(abs_frame_ids.begin(), abs_frame_ids.end());

    using std::vector;
    vector<vector<float> > all_exps;

    all_exps.reserve(MAX_FRAMEID);
    for (int t=0; t<MAX_FRAMEID; ++t)
        all_exps.push_back(compute_avg_exp_coeffs(t, ignore_first));

    write_2d_vector<float>(filepath, all_exps);

}


void VideoOutput::save_expressions_all(const std::string& filepath)
{
    int MAX_FRAMEID = *std::max_element(abs_frame_ids.begin(), abs_frame_ids.end());

    using std::vector;
    vector<vector<float> > all_exps;

    all_exps.reserve(MAX_FRAMEID);
    for (int t=0; t<=MAX_FRAMEID; ++t)
        all_exps.push_back(combine_exp_coeffs(t));

    write_2d_vector<float>(filepath, all_exps);

}



void VideoOutput::save_poses(const std::string& filepath, OptimizationVariables* ov, RotationComputer* rc)
{
    int MAX_FRAMEID = *std::max_element(abs_frame_ids.begin(), abs_frame_ids.end());

    using std::vector;
    vector<vector<float> > all_poses;

    all_poses.reserve(MAX_FRAMEID);
    for (int t=0; t<MAX_FRAMEID; ++t)
    {
        vector<float> cpose = compute_avg_pose(t);

        if (ov != NULL)
        {
            float yaw, pitch, roll;

            ov->set_frame(t);

            cudaMemcpy(ov->u, &cpose[0]+3, 3*sizeof(float), cudaMemcpyHostToDevice);
            rc->set_u_ptr(ov->u);
            rc->process();
            rc->compute_euler_angles(yaw, pitch, roll);

            cpose.push_back(yaw);
            cpose.push_back(pitch);
            cpose.push_back(roll);
        }
        all_poses.push_back(cpose);
    }

    write_2d_vector<float>(filepath, all_poses);
}


void VideoOutput::save_illums(const std::string& filepath)
{
    int MAX_FRAMEID = *std::max_element(abs_frame_ids.begin(), abs_frame_ids.end());

    using std::vector;
    vector<vector<float> > all_illums;

    all_illums.reserve(MAX_FRAMEID);
    for (int t=0; t<MAX_FRAMEID; ++t)
    {
        vector<float> cillums = get_illum(t);
        all_illums.push_back(cillums);
    }

    write_2d_vector<float>(filepath, all_illums);
}



/*
 *


def gaussian_kernel(width = 15, sigma = 1.):
    assert width == np.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = np.linspace(-radius,  radius,  width)
    x = np.float32(x)
    sigma = np.float32(sigma)
    filterx = x*x / (2 * sigma * sigma)
    filterx = np.exp(-1 * filterx)
    assert filterx.sum()>0,  'something very wrong if gaussian kernel sums to zero!'
    filterx /= filterx.sum()
    return filterx

f = gaussian_kernel()
print(f)
*/

