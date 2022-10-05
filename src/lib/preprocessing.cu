/*
 * funcs.cu
 *
 *  Created on: Aug 8, 2020
 *      Author: root
 */

#include "preprocessing.h"
#include "config.h"
#include "funcs.h"
#include <deque>
#include <numeric>

cv::Point2f transform_pt(float px, float py, const cv::Point2f* center,
                       float* scale, float* resolution, bool invert)
{
    float h = 200.0f * (*scale);

    float trans_data[9] = {(*resolution)/h, 0.0f, (*resolution)*(-center->x/h + 0.5f),
                           0.0f, (*resolution)/h, (*resolution)*(-center->y/h + 0.5f),
                           0.0f,0.0f,1.0f};
    cv::Mat trans(3,3, CV_32FC1, trans_data);

    float col_data[3] = {px, py, 1.0f};
    cv::Mat col(3,1, CV_32FC1, col_data);

    cv::Mat ptTrans_mat;
    if (invert)
        ptTrans_mat = trans.inv()*col;
    else
        ptTrans_mat = trans*col;

    cv::Point2f ptTrans(ptTrans_mat.at<float>(0,0), ptTrans_mat.at<float>(1,0));

    return ptTrans;
}


cv::Rect detect_face_opencv(Net& detection_net, const std::string& framework, cv::Mat &frame, Rect *prev_d, double* face_confidence, bool multiscale_start)
{
    cv::Rect d;

    if (multiscale_start) {
        if (prev_d->x == -1 || prev_d->y == -1) {
            d = detectFaceOpenCVDNN_multiscale(detection_net, frame, framework, face_confidence);
        } else {
            d = detectFaceOpenCVDNN(detection_net, frame, framework, face_confidence, prev_d);

            if (*face_confidence < 0.5) {
                d = detectFaceOpenCVDNN_multiscale(detection_net, frame, framework, face_confidence);
            }
        }
    } else {
        d = detectFaceOpenCVDNN(detection_net, frame, framework, face_confidence);
    }

    if (prev_d != NULL && *face_confidence > 0.5) {
        int face_size = (int)(d.width+d.height)/2.0;
        prev_d->x = std::max<int>(0, (int)d.x-2*(int)face_size);
        prev_d->y = std::max<int>(0, (int)d.y-2*(int)face_size);
        prev_d->width =  std::min<int>(frame.cols-prev_d->x, face_size*5);
        prev_d->height = std::min<int>(frame.rows-prev_d->y, face_size*5);
    }

    return d;
}


void detect_landmarks_opencv(const cv::Rect& d, double face_confidence, Net& landmark_net, Net& leye_net, Net& reye_net, Net& mouth_net, Net& correction_net,
                             cv::Mat &frame, std::vector<float>& xp_vec, std::vector<float>& yp_vec, float &bbox_size,
                             std::vector<float>& xrange, std::vector<float>& yrange, bool use_local_models, bool plot,
                         vector<vector<double> >* xs, vector<vector<double> >* ys)
{
    if (config::SINGLE_LANDMARKS)
    {
        detect_landmarks_opencv_single(d, face_confidence, landmark_net, leye_net, reye_net, mouth_net, correction_net,
                             frame, xp_vec, yp_vec, bbox_size, xrange, yrange, use_local_models, plot);
        
    }
    else
    {
        int denom = 0;
        std::vector<double> perturbsx({3, 5, -3, -5, 2, 4, -8, 6, 1, 7,-5, 8, -9, 3, 5});
        std::vector<double> perturbsy({1, 4, -2, -4, 5,3 -7,-4, 8, 0, -5, 4, -9, 3, 0});
        std::vector<double> perturbssize({4, 3, -2, -1, 7, 0, 3 -3, 6, 7, -4, -7, 6, 3, -9});

        for (size_t i=0; i<config::NMULTICOMBS; ++i)
        {
            vector<float> xcur, ycur, xrcur, yrcur;

            double fs = (d.width+d.height)/2.0;
            
            int xnew = d.x + fs*perturbsx[i] / 100.0;
            int ynew = d.y + fs*perturbsy[i] / 100.0;
            int widthnew = d.width + d.width*perturbssize[i]/100.0f;
            int heightnew = d.height + d.height*perturbssize[i]/100.0f;

            xnew = std::max<int>(xnew, 0);
            ynew = std::max<int>(ynew, 0);
            widthnew = std::min<int>(frame.cols-xnew-1, widthnew);
            heightnew = std::min<int>(frame.rows-ynew-1, heightnew);

            cv::Rect dcur(xnew, ynew, widthnew, heightnew);

            detect_landmarks_opencv_single(dcur, face_confidence, landmark_net, leye_net, reye_net, mouth_net, correction_net,
                             frame, xcur, ycur, bbox_size, xrcur, yrcur, use_local_models, false);

            if (xs != NULL && ys != NULL)
            {
                for (size_t j=0; j<NLANDMARKS_51; ++j)
                {
                    (*xs)[j].push_back(xcur[j]);
                    (*ys)[j].push_back(ycur[j]);
                }
            }

            if (i == 0)
            {
                
                xp_vec.insert(xp_vec.end(), xcur.begin(), xcur.end()); 
                yp_vec.insert(yp_vec.end(), ycur.begin(), ycur.end()); 
                xrange.insert(xrange.end(), xrcur.begin(), xrcur.end()); 
                yrange.insert(yrange.end(), yrcur.begin(), yrcur.end()); 

                if (xp_vec.size() == 0)
                    return;
            }
            else
            {
                for (size_t j=0; j<xcur.size(); ++j)
                {
                    xp_vec[j] += xcur[j];
                    yp_vec[j] += ycur[j];
                    xrange[j] += xrcur[j];
                    yrange[j] += yrcur[j];
                }
            }

            if (xcur.size() == NLANDMARKS_51)
                denom++;
        }

        for (size_t j=0; j<xp_vec.size(); ++j)
        {
            xp_vec[j] /= denom;
            yp_vec[j] /= denom;
            xrange[j] /= denom;
            yrange[j] /= denom;
        }



        if (plot)
        {
            for (uint i=0; i<xp_vec.size(); ++i)
            {

                cv::Point2f ptOrig(xp_vec[i]-1, yp_vec[i]-1);
                circle(frame, ptOrig, 2.5, cv::Scalar(0,0,255), cv::FILLED, 8, 0);
            }

            std::stringstream ss;
            ss << face_confidence;
            cv::rectangle(frame, cv::Point2f(d.x, d.y), cv::Point(d.x+d.width, d.y+d.height), cv::Scalar(0, 255, 0),2, 4);
            cv::putText(frame,ss.str(),  cv::Point2f(d.x, d.y-20), cv::FONT_HERSHEY_PLAIN, 1.75, cv::Scalar(0,0,255), 2);
            cv::imshow("frame", frame);
            cv::waitKey(0);
        }


        /*
        for (size_t j=0; j<NLANDMARKS_51; ++j)
        {
            xstd->push_back(compute_std(xs[j]));
            ystd->push_back(compute_std(ys[j]));
        }
        */
    }
}




void detect_landmarks_opencv_single(const cv::Rect& d, double face_confidence, Net& landmark_net, Net& leye_net, Net& reye_net, Net& mouth_net, Net& correction_net,
                             cv::Mat &frame, std::vector<float>& xp_vec, std::vector<float>& yp_vec, float &bbox_size,
                             std::vector<float>& xrange, std::vector<float>& yrange, bool use_local_models, bool plot)
{
    xp_vec.clear();
    yp_vec.clear();
    xrange.clear();
    yrange.clear();

    //    cv::GaussianBlur( frame, frame, cv::Size( 3, 3 ), 0, 0 );

    float resolution = 64.0f;

    cv::Point2f ptCenter((d.x+d.x+d.width)/2.0f, (d.y+d.y+d.height)/2.0f);
    float scale = (d.width+d.height)/195.0f;


    float res_orig = 256.0f;

    // upper left point
    cv::Point ul = transform_pt(1, 1, &ptCenter, &scale, &res_orig, true);
    cv::Point br = transform_pt(256.0f, 256.0f, &ptCenter, &scale, &res_orig, true);

    uint ht = frame.rows;
    uint wd = frame.cols;

    int newX[2] = {std::max<int>(1, -ul.x+1), std::min<int>(br.x, wd)-ul.x};
    int newY[2] = {std::max<int>(1, -ul.y+1), std::min<int>(br.y, ht)-ul.y};

    int oldX[2] = {std::max<int>(1, ul.x+1), std::min<int>(br.x, wd)};
    int oldY[2] = {std::max<int>(1, ul.y + 1), std::min<int>(br.y, ht)};

    bbox_size = -1.0f;

    /*
    // we comment these out for now -- looks like we are employing enough try/catch blocks for now
    if (newX[0] <0 || newX[1] < 0)
        return;

    if (newY[0] < 0 || newY[1] < 0)
        return;

    if (newX[0] > (int) wd || newX[1] > (int) wd)
        return;

    if (newY[0] > (int) ht || newY[1] > (int) ht)
        return;
    */

    cv::Mat im_cropped((int) (br.y-ul.y), (int) (br.x - ul.x), CV_8UC3, cv::Scalar::all(0) );

    cv::Mat tmpFace(frame(cv::Rect(oldX[0], oldY[0], oldX[1]-oldX[0], oldY[1]-oldY[0])));

    tmpFace.copyTo(im_cropped(cv::Rect(newX[0], newY[0], newX[1]-newX[0], newY[1]-newY[0])));

    //    cv::GaussianBlur( tmpFace, tmpFace, cv::Size( 3, 3 ), 0, 0 );

    int64 t0 = cv::getTickCount();
    cv::resize(im_cropped, im_cropped, cv::Size(256.0f, 256.0f));

    im_cropped.convertTo(im_cropped, CV_32FC3);
    cv::cvtColor(im_cropped, im_cropped, cv::COLOR_BGR2RGB);
    im_cropped = im_cropped/255.0f;

    cv::Mat net_input = cv::dnn::blobFromImage(im_cropped);
    landmark_net.setInput(net_input);
    cv::Mat netOut = landmark_net.forward().clone();

    cv::Mat im_cropped_flipped;
    cv::flip(im_cropped, im_cropped_flipped, 1);

    /*
    cv::imshow("c", im_cropped);
    cv::imshow("cf", im_cropped_flipped);
    cv::waitKey(0);
    */

    bool do_flip = true;

    if (do_flip)
    {
        cv::Mat net_input_flipped = cv::dnn::blobFromImage(im_cropped_flipped);
        landmark_net.setInput(net_input_flipped);
        cv::Mat netOut_flipped = landmark_net.forward().clone();
        heatmaps_to_landmarks(&netOut, &netOut_flipped, xp_vec, yp_vec, &ptCenter, NLANDMARKS_68, &scale, &resolution, do_flip, true);
    }
    else
    {
        heatmaps_to_landmarks(&netOut, NULL, xp_vec, yp_vec, &ptCenter, NLANDMARKS_68, &scale, &resolution, do_flip, true);
    }




    //    cv::waitKey(0);
    int cur_xmin = (int) *std::min_element(xp_vec.begin(), xp_vec.end());
    int cur_xmax = (int) *std::max_element(xp_vec.begin(), xp_vec.end());

    int cur_ymin = (int) *std::min_element(yp_vec.begin(), yp_vec.end());
    int cur_ymax = (int) *std::max_element(yp_vec.begin(), yp_vec.end());

    int face_width = cur_xmax-cur_xmin;
    int face_height = cur_ymax-cur_ymin;
    int face_size = (float) std::max<int>(face_width, face_height);

    bbox_size = face_size; // sqrt(face_width*face_width+face_height*face_height);

    cv::Mat part_tmp;

    using std::pair;
    using std::vector;

    std::vector<float> xp_vec_new(xp_vec), yp_vec_new(yp_vec);

    std::vector<float> min_xs(xp_vec), max_xs(xp_vec), min_ys(yp_vec), max_ys(yp_vec);

    std::vector<double> tx_rates{0.04, -0.04, 0.04, -0.04, 0.02, -0.02, 0.01, 0.03};
    std::vector<double> ty_rates{-0.01, 0.03, 0.04, -0.06, 0.01, -0.01, 0.025, -0.03};

    for (uint tau=0; tau<tx_rates.size(); ++tau) {
        pair<vector<float>, vector<float>> cur_local_pts = compute_landmarks_wlocal_models(frame, face_size, xp_vec, yp_vec, part_tmp,
                                                                                           leye_net, reye_net, mouth_net, tx_rates[tau], ty_rates[tau]);
        vector<float> &xcur = std::get<0>(cur_local_pts);
        vector<float> &ycur = std::get<1>(cur_local_pts);

        for (uint i=0; i<68; ++i) {
            xp_vec_new[i] += xcur[i];
            yp_vec_new[i] += ycur[i];


            max_xs[i] = std::max<float>(xcur[i], max_xs[i]);
            max_ys[i] = std::max<float>(ycur[i], max_ys[i]);
            min_xs[i] = std::min<float>(xcur[i], min_xs[i]);
            min_ys[i] = std::min<float>(ycur[i], min_ys[i]);
        }
    }

    for (uint i=17; i<68; ++i) {
        xrange.push_back((max_xs[i]-min_xs[i])/face_size);
        yrange.push_back((max_ys[i]-min_ys[i])/face_size);
    }

    if (use_local_models)
    {
        for (uint i=0; i<68; ++i) {
            xp_vec[i] = xp_vec_new[i]/(tx_rates.size()+1);
            yp_vec[i] = yp_vec_new[i]/(tx_rates.size()+1);
        }
    }

    /*
     *
     *     std::vector<float> min_xs(xp_vec), max_xs(xp_vec), min_ys(yp_vec), max_ys(yp_vec);

    for (uint tau=0; tau<tx_rates.size(); ++tau) {
        pair<vector<float>, vector<float>> cur_local_pts = compute_landmarks_wlocal_models(frame, face_size, xp_vec, yp_vec, part_tmp,
                                                                                           leye_net, reye_net, mouth_net, tx_rates[tau], ty_rates[tau]);
        vector<float> &xcur = std::get<0>(cur_local_pts);
        vector<float> &ycur = std::get<1>(cur_local_pts);

        for (uint i=0; i<68; ++i) {
            max_xs[i] = std::max<float>(xcur[i], max_xs[i]);
            max_ys[i] = std::max<float>(ycur[i], max_ys[i]);
            min_xs[i] = std::min<float>(xcur[i], min_xs[i]);
            min_ys[i] = std::min<float>(ycur[i], min_ys[i]);
        }
    }*/


    bool apply_correction = false;
//    bool apply_correction = true;



    if (apply_correction)
    {

        float xmean = std::accumulate(xp_vec.begin(), xp_vec.end(), 0.0f)/xp_vec.size();
        float ymean = std::accumulate(yp_vec.begin(), yp_vec.end(), 0.0f)/yp_vec.size();

        float landmarks_mean = (xmean+ymean)/2.0f;

        std::vector<float> landmarks_zeromean;

        for (uint i=0; i<68; ++i) {
            landmarks_zeromean.push_back(xp_vec[i]-xmean);
            landmarks_zeromean.push_back(yp_vec[i]-ymean);
        }

        float landmarks_std = compute_std<float>(landmarks_zeromean);

        std::vector<float> landmarks_tmp;

        for (uint i=0; i<xp_vec.size(); ++i) {
            landmarks_tmp.push_back((xp_vec[i]-xmean)/landmarks_std);
            landmarks_tmp.push_back((yp_vec[i]-ymean)/landmarks_std);
        }

        cv::Mat input_mat(1, NLANDMARKS_68*2, CV_32FC1, landmarks_tmp.data());

        correction_net.setInput(cv::dnn::blobFromImage(input_mat));
        cv::Mat cnet_Out = correction_net.forward();

        float *cnet_data = (float*) cnet_Out.data;
        /*
        for (uint i=0; i<NLANDMARKS_68*2; ++i) {
            landmarks_tmp[i] += cnet_data[i];
        }
        for (uint i=0; i<xp_vec.size(); ++i)
        {
            cv::Point2f ptOrig(xp_vec[i], yp_vec[i]);
            circle(frame, ptOrig, 3, cv::Scalar(0,0,255), cv::FILLED, 8, 0);
        }
    */
        for (uint i=0; i<NLANDMARKS_68; ++i) {
            xp_vec[i] = (landmarks_tmp[2*i]+cnet_data[2*i])*landmarks_std+xmean;
            yp_vec[i] = (landmarks_tmp[2*i+1]+cnet_data[2*i+1])*landmarks_std+ymean;
        }
    }


    xp_vec.erase(xp_vec.begin(), xp_vec.begin()+17);
    yp_vec.erase(yp_vec.begin(), yp_vec.begin()+17);

    /*
    xrange.clear();
    yrange.clear();

    for (uint i=17; i<68; ++i) {
        xrange.push_back((max_xs[i]-min_xs[i])/bbox_size);
        yrange.push_back((max_ys[i]-min_ys[i])/bbox_size);
    }

    xp_vec.erase(xp_vec.begin(), xp_vec.begin()+17);
    yp_vec.erase(yp_vec.begin(), yp_vec.begin()+17);
*/
    /*
    for (uint i=0; i<xp_vec.size(); ++i)
    {
        cv::Point2f ptOrig(xp_vec[i], yp_vec[i]);
        circle(frame, ptOrig, 3, cv::Scalar(255,255,255), cv::FILLED, 8, 0);
    }
    */

    if (plot)
    {
        cv::Mat frame_clone = frame.clone();
        for (uint i=0; i<xp_vec.size(); ++i)
        {

            cv::Point2f ptOrig(xp_vec[i], yp_vec[i]);
            circle(frame_clone, ptOrig, 2, cv::Scalar(0,0,255), cv::FILLED, 8, 0);
        }

        std::stringstream ss;
        ss << face_confidence;
        cv::rectangle(frame_clone, cv::Point(d.x, d.y), cv::Point(d.x+d.width, d.y+d.height), cv::Scalar(0, 255, 0),2, 4);
        //! cv::imshow("frame_clone", frame_clone);
        //! cv::waitKey(1);
        //! cv::putText(frame_clone,ss.str(),  cv::Point(d.x, d.y-20), cv::FONT_HERSHEY_PLAIN, 1.75, cv::Scalar(0,0,255), 2);
    }
    /*
    */
    /*
//    cv::imshow("part", part);
    for (uint i=0; i<xp_vec.size(); ++i)
    {
        cv::Point2f ptOrig(xp_vec[i], yp_vec[i]);
        circle(frame, ptOrig, 3, cv::Scalar(0,0,255), cv::FILLED, 8, 0);
    }
    cv::imshow("frame", frame);

    cv::waitKey(0);
    */

    /*
    int64 t1 = cv::getTickCount();
    double secs = (t1-t0)/cv::getTickFrequency();

    std::cout << secs << std::endl;
    */
}


void heatmaps_to_landmarks(cv::Mat* netOut, cv::Mat* netOut_flipped,
                           std::vector<float>& xp_vec, std::vector<float>& yp_vec,
                           cv::Point2f* ptCenter, uint num_landmarks,
                           float* scale, float* resolution, bool do_flip, bool do_transformation)
{
    float *data_ptr = (float*) netOut->data;
    float *data_ptr_flipped;

    if (do_flip)
        data_ptr_flipped = (float*) netOut_flipped->data;

    int pairs[] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20,
                   19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36,
                   41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65};

    for (uint i=0; i<num_landmarks; ++i)
    {
        float data_local[64*64];
        memcpy(data_local, data_ptr+64*64*i, 64*64*sizeof(float));

        cv::Mat curMap(64, 64, CV_32FC1,  data_local);
        cv::Mat curMap_total = curMap;

        if (do_flip)
        {
            float data_flipped_local[64*64];
            memcpy(data_flipped_local, data_ptr_flipped+64*64*pairs[i], 64*64*sizeof(float));
            cv::Mat curMap_flipped(64, 64, CV_32FC1, data_flipped_local);
            cv::flip(curMap_flipped, curMap_flipped, 1);
            curMap_total = curMap_total + curMap_flipped;
        }

        cv::Point maxLoc;

        double minSrc, maxSrc;
        cv::minMaxLoc(curMap_total, &minSrc, &maxSrc, 0, &maxLoc);
        curMap_total = (curMap_total-minSrc)/(maxSrc-minSrc);
        //        cv::imshow("curmap", curMap_total);
        //        cv::waitKey(0);

        maxLoc.x += 1.0f;
        maxLoc.y += 1.0f;

        float px = maxLoc.x-1.0f;
        float py = maxLoc.y-1.0f;
        float diffx = curMap_total.at<float>(py, px+1) - curMap_total.at<float>(py, px-1);
        float diffy = curMap_total.at<float>(py+1, px) - curMap_total.at<float>(py-1, px);

        px += 1.0f;
        py += 1.0f;

        if (diffx > 0)
            px += 0.25f;
        else
            px -= 0.25f;

        if (diffy > 0)
            py += 0.25f;
        else
            py -= 0.25f;

        px -= 0.5f;
        py -= 0.5f;

        // we skip the points that correspond to the jaw, hence we start from 17
        if (do_transformation)
        {
            cv::Point2f ptOrig = transform_pt(px, py, ptCenter, scale, resolution, true);
            xp_vec.push_back(ptOrig.x);
            yp_vec.push_back(ptOrig.y);
        }
        else
        {
            xp_vec.push_back(px);
            yp_vec.push_back(py);
        }
    }
}


std::pair<std::vector<float>, std::vector<float>> compute_landmarks_wlocal_models(cv::Mat& frame, int face_size, std::vector<float>& xp_vec0, std::vector<float>& yp_vec0, cv::Mat& part,
                                                                                  Net& leye_net, Net& reye_net, Net& mouth_net, double tx_rate, double ty_rate)
{
    double part_rescale_rate_leye, part_rescale_rate_reye, part_rescale_rate_mouth;
    int part_top_leye, part_left_leye, part_top_reye, part_left_reye, part_top_mouth, part_left_mouth;

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    crop_part(PART::LEYE, frame, face_size, xp_vec0, yp_vec0, tx_rate, ty_rate, part_rescale_rate_leye, part_top_leye, part_left_leye, part);
    part.convertTo(part, CV_32FC3);
    part = part/255.0f;

    cv::Mat leye_input = cv::dnn::blobFromImage(part, 1.0f, cv::Size(), cv::Scalar(), true);
    leye_net.setInput(leye_input);
    cv::Mat leye_netOut = leye_net.forward().clone();

    std::vector<float> xp_vec_leye, yp_vec_leye;
    heatmaps_to_landmarks(&leye_netOut, NULL, xp_vec_leye, yp_vec_leye, NULL, 11, NULL, NULL, false, false);
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    crop_part(PART::REYE, frame, face_size, xp_vec0, yp_vec0, tx_rate, ty_rate, part_rescale_rate_reye, part_top_reye, part_left_reye, part);
    part.convertTo(part, CV_32FC3);
    part = part/255.0f;

    cv::Mat reye_input = cv::dnn::blobFromImage(part, 1.0f, cv::Size(), cv::Scalar(), true);
    reye_net.setInput(reye_input);
    cv::Mat reye_netOut = reye_net.forward().clone();

    std::vector<float> xp_vec_reye, yp_vec_reye;
    heatmaps_to_landmarks(&reye_netOut, NULL, xp_vec_reye, yp_vec_reye, NULL, 11, NULL, NULL, false, false);
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    crop_part(PART::MOUTH, frame, face_size, xp_vec0, yp_vec0, tx_rate, ty_rate, part_rescale_rate_mouth, part_top_mouth, part_left_mouth, part);
    part.convertTo(part, CV_32FC3);
    part = part/255.0f;

    cv::Mat mouth_input = cv::dnn::blobFromImage(part, 1.0f, cv::Size(), cv::Scalar(), true);
    mouth_net.setInput(mouth_input);
    cv::Mat mouth_netOut = mouth_net.forward().clone();

    std::vector<float> xp_vec_mouth, yp_vec_mouth;
    heatmaps_to_landmarks(&mouth_netOut, NULL, xp_vec_mouth, yp_vec_mouth, NULL, 20, NULL, NULL, false, false);

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint lidx_src[11] = {17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41};
    uint ridx_src[11] = {22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47};
    uint midx_src[20] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67};


    std::vector<float> xp_local(xp_vec0), yp_local(yp_vec0);

    for (uint i=0; i<11; i++) {

        xp_local[lidx_src[i]] = part_rescale_rate_leye*xp_vec_leye[i] + part_left_leye;
        yp_local[lidx_src[i]] = part_rescale_rate_leye*yp_vec_leye[i] + part_top_leye;
        xp_local[ridx_src[i]] = part_rescale_rate_reye*xp_vec_reye[i] + part_left_reye;
        yp_local[ridx_src[i]] = part_rescale_rate_reye*yp_vec_reye[i] + part_top_reye;
    }

    for (uint i=0; i<20; i++) {
        xp_local[midx_src[i]] = part_rescale_rate_mouth*xp_vec_mouth[i] + part_left_mouth;
        yp_local[midx_src[i]] = part_rescale_rate_mouth*yp_vec_mouth[i] + part_top_mouth;
    }

    return std::pair<std::vector<float>, std::vector<float>>(xp_local, yp_local);
}


cv::Rect detectFaceOpenCVDNN_multiscale(Net detection_net, Mat &frame, string framework, double* confidence_ptr)
{
    std::vector<double> confidences;
    std::vector<cv::Rect> rects;

    std::vector<int> step_sizes({600});

    for (int &step_size : step_sizes)
    {
        for (int offx=0; offx<frame.cols; offx += step_size/2)
        {
            for (int offy=0; offy<frame.rows; offy += step_size/2)
            {
                int x0=offx;
                int y0=offy;
                int xf=std::min<int>(frame.cols, offx+step_size);
                int yf=std::min<int>(frame.rows, offy+step_size);

                double face_confidence_cur;
                cv::Rect curROI(x0, y0, (xf-x0), (yf-y0));

                cv::Rect curd = detectFaceOpenCVDNN(detection_net, frame, framework, &face_confidence_cur, &curROI);

                confidences.push_back(face_confidence_cur);
                rects.push_back(curd);
            }
        }
    }

    int maximizer = std::max_element(confidences.begin(), confidences.end()) - confidences.begin();

    *confidence_ptr = confidences[maximizer];

    return cv::Rect(rects[maximizer]); //detectFaceOpenCVDNN(detection_net, frame, framework, &face_confidence, prev_d);
}



cv::Rect detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN_orig, string framework, double* confidence_ptr, cv::Rect* ROI)
{
    cv::Mat frameOpenCVDNN = frameOpenCVDNN_orig.clone();

    if (ROI != NULL) {
        frameOpenCVDNN = frameOpenCVDNN(*ROI);
    }

    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    cv::Mat inputBlob;
    if (framework == "caffe")
        inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
    else
        inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());


    cv::Rect rect_return(-1, -1, -1, -1);
    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            //            cv::rectangle(copyFrameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
            if (rect_return.x == -1) {
                if (ROI != NULL)
                    rect_return = cv::Rect(x1+ROI->x, y1+ROI->y, x2-x1, y2-y1);
                else
                    rect_return = cv::Rect(x1, y1, x2-x1, y2-y1);

                if (confidence_ptr != NULL)
                    *confidence_ptr = confidence;
            }
        }
    }

    if (rect_return.x != -1)
        return rect_return;

    cv::Mat frameOpenCVDNN_gray = frameOpenCVDNN.clone();

    cv::cvtColor(frameOpenCVDNN_gray, frameOpenCVDNN_gray, cv::COLOR_RGB2GRAY);
    cv::cvtColor(frameOpenCVDNN_gray, frameOpenCVDNN_gray, cv::COLOR_GRAY2RGB);


    cv::Mat inputBlob_gray;
    if (framework == "caffe")
        inputBlob_gray = cv::dnn::blobFromImage(frameOpenCVDNN_gray, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
    else
        inputBlob_gray = cv::dnn::blobFromImage(frameOpenCVDNN_gray, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);

    net.setInput(inputBlob_gray, "data");
    detection = net.forward("detection_out");

    cv::Mat detectionMat_gray(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());


    for(int i = 0; i < detectionMat_gray.rows; i++)
    {
        float confidence = detectionMat_gray.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat_gray.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat_gray.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat_gray.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat_gray.at<float>(i, 6) * frameHeight);

            if (rect_return.x == -1) {
                if (ROI != NULL)
                    rect_return = cv::Rect(x1+ROI->x, y1+ROI->y, x2-x1, y2-y1);
                else
                    rect_return = cv::Rect(x1, y1, x2-x1, y2-y1);

                if (confidence_ptr != NULL)
                    *confidence_ptr = confidence;
            }

        }
    }

    return rect_return;
}


cv::Point transform_pt(float px, float py, const cv::Point2f& center,
                       float scale,
                       float resolution,
                       bool invert)
{
    float h = 200.0f * scale;

    float trans_data[9] = {resolution/h, 0.0f, resolution*(-center.x/h + 0.5f),
                           0.0f, resolution/h, resolution*(-center.y/h + 0.5f),
                           0.0f,0.0f,1.0f};
    cv::Mat trans(3,3, CV_32FC1, trans_data);

    float col_data[3] = {px, py, 1.0f};
    cv::Mat col(3,1, CV_32FC1, col_data);

    cv::Mat ptTrans_mat;
    if (invert)
        ptTrans_mat = trans.inv()*col;
    else
        ptTrans_mat = trans*col;

    cv::Point ptTrans((int)ptTrans_mat.at<float>(0,0), (int)ptTrans_mat.at<float>(1,0));

    return ptTrans;
}


void crop_part(int part_id, cv::Mat& im, double face_size,
               std::vector<float>& xp_vec, std::vector<float>& yp_vec,
               double tx_rate, double ty_rate, double& rescale_rate,
               int& top, int& left, cv::Mat& part)
{
    double xmean(0), ymean(0);
    uint leye_landmarks[2] = {36, 39};
    uint reye_landmarks[2] = {42, 45};
    uint mouth_landmarks[20] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67};

    double part_size = 0.0;

    int num_cidx(-1);
    if (part_id == PART::LEYE) {
        for (uint i=0; i<2; i++) {
            xmean += xp_vec[leye_landmarks[i]];
            ymean += yp_vec[leye_landmarks[i]];
        }
        num_cidx = 2;
        part_size = face_size/1.2;
    } else if (part_id == PART::REYE) {
        for (uint i=0; i<2; i++) {
            xmean += xp_vec[reye_landmarks[i]];
            ymean += yp_vec[reye_landmarks[i]];
        }
        num_cidx = 2;
        part_size = face_size/1.2;
    } else if (part_id == PART::MOUTH) {
        for (uint i=0; i<20; i++) {
            xmean += xp_vec[mouth_landmarks[i]];
            ymean += yp_vec[mouth_landmarks[i]];
        }
        num_cidx = 20;
        part_size = face_size/1.35;
    }

    xmean /= num_cidx;
    ymean /= num_cidx;

    cv::Point2f partCenter(xmean, ymean);

    uint part_size_int = round(part_size);

    top = round(face_size*ty_rate + partCenter.y-part_size/2.0);
    left = round(face_size*tx_rate + partCenter.x-part_size/2.0);

    if (left < 0)
        left = std::max<int>(0, left);
    if (top < 0)
        top = std::max<int>(0, top);

    cv::Rect part_rect(left, top, part_size_int, part_size_int);

    part = im(part_rect).clone();

    float image_shiftx = 32*(1-1.15); //part.cols/2.0-32.0;
    float image_shifty = 32*(1-1.15);; //part.rows/2.0-32.0;

    float M_data[6] = {1.15f, 0.0f, image_shiftx, 0.0f, 1.15f, image_shifty};

    cv::Mat M(2,3, CV_32FC1, M_data);

    rescale_rate = part.cols/64.0;
    cv::resize(part, part, cv::Size(64, 64));
    cv::warpAffine(part, part, M, cv::Size(64,64));
}




