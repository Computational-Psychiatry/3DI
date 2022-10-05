#include "input_data.h"
#include "constants.h"

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
