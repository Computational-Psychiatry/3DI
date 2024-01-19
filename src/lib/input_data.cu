#include "input_data.h"
#include "constants.h"
#include "preprocessing.h"
#include "funcs.h"

void InputData::add_data(const cv::Mat& frame, const std::vector<float>& xp, const std::vector<float> &yp, size_t fi, float face_size)
{
    if (frames.size() == T)
    {
        /*
        frames.clear();
        xp_origs.clear();
        yp_origs.clear();
        abs_frame_ids.clear();
        face_sizes.clear();
        */
        frames.pop_front();
        xp_origs.pop_front();
        yp_origs.pop_front();
        abs_frame_ids.pop_front();
        face_sizes.pop_front();
    }

    frames.push_back(frame);
    xp_origs.push_back(xp);
    yp_origs.push_back(yp);
    abs_frame_ids.push_back(fi);
    face_sizes.push_back(face_size);
}


void InputData::get_resized_landmarks(size_t rel_frame_id, const float resize_coef, float* xp, float *yp)
{
    // @@@ the landmarks will probably be already resized in the data structure
    for (int i=0; i<NLANDMARKS_51; ++i)
    {
        xp[i] = resize_coef*(xp_origs[rel_frame_id][i]);
        yp[i] = resize_coef*(yp_origs[rel_frame_id][i]);
    }
}


void InputData::get_resized_frame(size_t rel_frame_id, const float resize_coef, cv::Mat& frame_dst)
{
    cv::cvtColor(frames[rel_frame_id], frame_dst, cv::COLOR_BGR2GRAY);

    frame_dst.convertTo(frame_dst, CV_32FC1);
    frame_dst = frame_dst/255.0f;

    // @@@ probably needs to be done for all cams
    cv::resize(frame_dst, frame_dst, cv::Size(), resize_coef, resize_coef);
    cv::copyMakeBorder(frame_dst, frame_dst, 0, DIMY, 0, DIMX, cv::BORDER_CONSTANT, 0);
}


void InputData::clear()
{
    frames.clear();
    xp_origs.clear();
    yp_origs.clear();
    abs_frame_ids.clear();
}


LandmarkData::LandmarkData()
{
    LandmarkData::check_CUDA(LANDMARK_DETECTOR_BACKEND, LANDMARK_DETECTOR_TARGET);
}

LandmarkData::LandmarkData(const std::string& landmarks_path)
{
    LandmarkData::check_CUDA(LANDMARK_DETECTOR_BACKEND, LANDMARK_DETECTOR_TARGET);

    init_from_txtfile(landmarks_path);
}


LandmarkData::LandmarkData(const std::string &video_path, const std::string &faces_path, const std::string& landmarks_path)
{
    LandmarkData::check_CUDA(LANDMARK_DETECTOR_BACKEND, LANDMARK_DETECTOR_TARGET);

    vector<vector<float> > face_rects;
    if (std::experimental::filesystem::exists(faces_path))
        face_rects = read2DVectorFromFile_unknown_size<float>(faces_path);
    else
        face_rects = detect_faces(video_path, faces_path);

    vector<vector<float> > all_lmks;
    if (std::experimental::filesystem::exists(landmarks_path))
        all_lmks = read2DVectorFromFile_unknown_size<float>(landmarks_path);
    else
        all_lmks = detect_landmarks(video_path, face_rects, landmarks_path);

    fill_xpypvec(all_lmks);
}


void LandmarkData::init_from_txtfile(const std::string &landmarks_path)
{
    vector<vector<float> > all_lmks = read2DVectorFromFile_unknown_size<float>(landmarks_path);
    fill_xpypvec(all_lmks);

}

void LandmarkData::fill_xpypvec(vector<vector<float> > &all_lmks)
{
    size_t T = all_lmks.size();
    for (size_t t=0; t<T; ++t)
    {
        vector<float> xp_vec, yp_vec;

        for (size_t i=0; i<NLANDMARKS_51; ++i)
        {
            xp_vec.push_back(all_lmks[t][2*i]);
            yp_vec.push_back(all_lmks[t][2*i+1]);
        }

        xp_vecs.push_back(xp_vec);
        yp_vecs.push_back(yp_vec);
    }
}

int LandmarkData::get_face_size(size_t t)
{
    vector<float> xp_vec = get_xpvec(t);
    vector<float> yp_vec = get_ypvec(t);

    //    cv::waitKey(0);
    int cur_xmin = (int) *std::min_element(xp_vec.begin(), xp_vec.end());
    int cur_xmax = (int) *std::max_element(xp_vec.begin(), xp_vec.end());

    int cur_ymin = (int) *std::min_element(yp_vec.begin(), yp_vec.end());
    int cur_ymax = (int) *std::max_element(yp_vec.begin(), yp_vec.end());

    int face_width = cur_xmax-cur_xmin;
    int face_height = cur_ymax-cur_ymin;
    return (float) std::max<int>(face_width, face_height);
}


bool LandmarkData::check_CUDA(cv::dnn::Backend& LANDMARK_DETECTOR_BACKEND, cv::dnn::Target& LANDMARK_DETECTOR_TARGET)
{
    const std::string caffeConfigFile = config::FACE_DETECTOR_DPATH;
    const std::string caffeWeightFile = config::FACE_DETECTOR_MPATH;
    cv::dnn::Net landmark_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_MPATH);

    landmark_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    landmark_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::Mat im_cropped(256, 256, CV_32FC3, cv::Scalar::all(1) );
    cv::Mat net_input = cv::dnn::blobFromImage(im_cropped);
    landmark_net.setInput(net_input);

    //cv::Mat netOut = landmark_net.forward().clone();
    try {
        cv::Mat netOut = landmark_net.forward().clone();
    } catch (cv::Exception) {
        std::cout << "OpenCV does not seem to have been install with CUDA support. If you did install with CUDA support, you may have compiled it with the wrong CUDA_ARCH_BIN parameter. " << std::endl <<
                     "Consider re-compiling OpenCV, and during compilation make sure that the CUDA_ARCH_BIN is the correct one for your device. (Our github repository contains a wiki page with compilation instructions.  )" << std::endl <<
                     "Without CUDA support, landmark detection will be very slow." << std::endl;
        LANDMARK_DETECTOR_BACKEND = cv::dnn::DNN_BACKEND_OPENCV;
        LANDMARK_DETECTOR_TARGET = cv::dnn::DNN_TARGET_CPU;
        return false;
    }
    return true;
}

vector<vector<float> > LandmarkData::detect_faces(const std::string& filepath, const std::string& rects_filepath)
{
    const std::string caffeConfigFile = config::FACE_DETECTOR_DPATH;
    const std::string caffeWeightFile = config::FACE_DETECTOR_MPATH;

    std::string framework = "caffe";

    cv::dnn::Net detection_net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    detection_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    detection_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::VideoWriter video_out;

    cv::VideoCapture capture(filepath);

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    cv::Mat frame;

    vector<float> face_sizes;
    int idx = 0;
    cv::Rect ROI(-1, -1, -1, -1);

    vector<vector<float> > face_rects;
    while (true) {
        idx++;

        std::cout << "Processing frame #" << idx << '\r' << std::flush;
        vector<float> xp_vec, yp_vec;
        vector<float> xrange, yrange;

        capture >> frame;

        if (frame.empty())
            break;

        if (idx < 0)
            continue;

        double face_confidence;
        cv::Rect face_rect = detect_face_opencv(detection_net, framework, frame, &ROI, &face_confidence, true);
        face_rects.push_back(vector<float>({(float)face_rect.x, (float)face_rect.y, (float)face_rect.width, (float)face_rect.height}));

        if (idx >= config::MAX_VID_FRAMES_TO_PROCESS)
            break;
    }

    write_2d_vector<float>(rects_filepath, face_rects);

    return face_rects;
}


vector<vector<float> > LandmarkData::detect_landmarks(const std::string &video_filepath, const vector<vector<float> > &face_rects, const std::string &landmarks_filepath)
{
    vector<vector<float> > all_lmks;

    const std::string caffeConfigFile = config::FACE_DETECTOR_DPATH;
    const std::string caffeWeightFile = config::FACE_DETECTOR_MPATH;

    std::string device = "CPU";
    std::string framework = "caffe";

    cv::dnn::Net detection_net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

    cv::dnn::Net landmark_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_MPATH);
    landmark_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    landmark_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::dnn::Net leye_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_LEYE_MPATH);
    leye_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    leye_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::dnn::Net reye_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_REYE_MPATH);
    reye_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    reye_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::dnn::Net mouth_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_MOUTH_MPATH);
    mouth_net.setPreferableBackend(LANDMARK_DETECTOR_BACKEND);
    mouth_net.setPreferableTarget(LANDMARK_DETECTOR_TARGET);

    cv::dnn::Net correction_net = cv::dnn::readNetFromTensorflow(config::LANDMARK_CORRECTION_MPATH);

    cv::VideoCapture capture(video_filepath);

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    cv::Mat frame;


    int Nframes = std::min<int>(config::MAX_VID_FRAMES_TO_PROCESS, (int) capture.get(cv::CAP_PROP_FRAME_COUNT));
    int idx = 0;
    cv::Rect ROI(-1, -1, -1, -1);
    while (true) {
        idx++;

        std::cout << "Processing frame #" << idx << "/" << Nframes << '\r' << std::flush;

        vector<float> xp_vec, yp_vec;
        vector<float> xrange, yrange;

        if (config::PRINT_DEBUG) {
            if (idx % config::PRINT_EVERY_N_FRAMES == 0)
                std::cout << "Processing frame# " << idx << std::endl;
        }

        capture >> frame;

        if (frame.empty())
            break;

        if (idx >= config::MAX_VID_FRAMES_TO_PROCESS)
            break;

        float face_size;

        double face_confidence(0.99);

        if (idx-1 >= face_rects.size()) {
            std::cout << "WARNING: Looks like there are not enough face rectangles during landmark detection; breaking" << std::endl;
        }

        cv::Rect face_rect(face_rects[idx-1][0], face_rects[idx-1][1], face_rects[idx-1][2], face_rects[idx-1][3]);

        try {
            if (face_rect.width > 10) {
                detect_landmarks_opencv(face_rect, face_confidence, landmark_net, leye_net, reye_net, mouth_net, correction_net, frame,
                                        xp_vec, yp_vec, face_size, xrange, yrange, config::USE_LOCAL_MODELS, false);
            }
        } catch (std::exception& e)
        {
            std::cout << "Problem with landmark detection at frame " << idx << std::endl;
        }

        vector<float> lmks_combined;
        for (size_t i=0; i<NLANDMARKS_51; ++i)
        {
            if (xp_vec.size() == 51 && yp_vec.size() == 51)
            {
                lmks_combined.push_back(xp_vec[i]);
                lmks_combined.push_back(yp_vec[i]);
            }
            else
            {
                lmks_combined.push_back(0);
                lmks_combined.push_back(0);
            }
        }

        all_lmks.push_back(lmks_combined);

        if (face_size == -1.0f)
            continue;

        if (idx > config::MAX_VID_FRAMES_TO_PROCESS)
            break;
    }

    std::cout << std::endl;

    write_2d_vector<float>(landmarks_filepath, all_lmks);

    return all_lmks;
}




