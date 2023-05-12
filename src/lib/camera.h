/*
 * camera.h
 *
 *  Created on: Aug 11, 2020
 *      Author: root
 */
#ifndef CAMERA_H_
#define CAMERA_H_

#include "constants.h"
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

struct Camera
{
    float *phix;
    float *phiy;
    float *cx;
    float *cy;

    float h_phix;
    float h_phiy;
    float h_cx;
    float h_cy;

    float orig_phix;
    float orig_phiy;
    float orig_cx;
    float orig_cy;

    float resize_coef;

    // if we want to undistort too, we use this
    bool cam_remap;

    bool initialized = false;

    // Those are matrices that are used for un-distorting images
    cv::Mat map1, map2;

    std::string calibration_path;


    Camera()
    {
        phix = NULL;
        phiy = NULL;
        cx = NULL;
        cy = NULL;
        resize_coef = 1.0;
    }

    Camera(const Camera& cam0)
    {
        if (cam0.cam_remap) {
            init(cam0.calibration_path);
            resize_coef = cam0.resize_coef;
            calibration_path = cam0.calibration_path;
        } else {
            init(cam0.h_phix, cam0.h_phiy, cam0.h_cx, cam0.h_cy, cam0.cam_remap);
            resize_coef = cam0.resize_coef;
        }
    }

    Camera(float _phix,
           float _phiy,
           float _cx,
           float _cy,
           bool _cam_remap = false) : h_phix(_phix), h_phiy(_phiy), h_cx(_cx), h_cy(_cy), cam_remap(_cam_remap)
    {
        init(_phix, _phiy, _cx, _cy, _cam_remap);
    }

    Camera(const std::string& calibration_path_) : calibration_path(calibration_path_)
    {
        init(calibration_path);
    }

    void init(const std::string& cameramodel_path)
    {
        calibration_path = cameramodel_path;
        initialized = true;
        cam_remap = false; // this is off now
        cv::Size imageSize;

        cv::FileStorage file(cameramodel_path, cv::FileStorage::READ);
        cv::Mat camMat_preUndistort, camMat, distCoeffs;

        file["cameraMatrix_preUndistort"] >> camMat_preUndistort;
        file["cameraMatrix_postUndistort"] >> camMat;
        file["distCoeffs"] >> distCoeffs;
        file["imageSize"] >> imageSize;

        cv::initUndistortRectifyMap(
                    camMat_preUndistort, distCoeffs, cv::Mat(),
                    camMat, imageSize,
                    CV_16SC2, map1, map2);

        h_phix = (float) camMat.at<double>(0,0);
        h_phiy = (float) camMat.at<double>(1,1);
        h_cx = (float) camMat.at<double>(0,2);
        h_cy = (float) camMat.at<double>(1,2);

        HANDLE_ERROR( cudaMalloc( (void**)&phix, sizeof(float)) );
        HANDLE_ERROR( cudaMalloc( (void**)&phiy, sizeof(float)) );

        HANDLE_ERROR( cudaMalloc( (void**)&cx, sizeof(float)) );
        HANDLE_ERROR( cudaMalloc( (void**)&cy, sizeof(float)) );

        HANDLE_ERROR( cudaMemcpy( phix, &h_phix,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( phiy, &h_phiy,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( cx, &h_cx,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( cy, &h_cy,  sizeof(float), cudaMemcpyHostToDevice) );

        orig_phix = h_phix;
        orig_phiy = h_phiy;
        orig_cx = h_cx;
        orig_cy = h_cy;

        cv::initUndistortRectifyMap(
                    camMat_preUndistort, distCoeffs, cv::Mat(),
                    camMat, imageSize,
                    CV_16SC2, map1, map2);


    }

    void init(float _phix,
              float _phiy,
              float _cx,
              float _cy,
              bool _cam_remap = false)
    {
        initialized = true;

        h_phix = _phix;
        h_phiy = _phiy;
        h_cx = _cx;
        h_cy = _cy;
        cam_remap = _cam_remap;

        HANDLE_ERROR( cudaMalloc( (void**)&phix, sizeof(float)) );
        HANDLE_ERROR( cudaMalloc( (void**)&phiy, sizeof(float)) );

        HANDLE_ERROR( cudaMalloc( (void**)&cx, sizeof(float)) );
        HANDLE_ERROR( cudaMalloc( (void**)&cy, sizeof(float)) );

        HANDLE_ERROR( cudaMemcpy( phix, &_phix,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( phiy, &_phiy,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( cx, &_cx,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( cy, &_cy,  sizeof(float), cudaMemcpyHostToDevice) );

        orig_phix = h_phix;
        orig_phiy = h_phiy;
        orig_cx = h_cx;
        orig_cy = h_cy;
    }



    void update_camera(float _resize_coef)
    {
        resize_coef = _resize_coef;
        h_phix = orig_phix*resize_coef;
        h_phiy = orig_phiy*resize_coef;
        h_cx = orig_cx*resize_coef;
        h_cy = orig_cy*resize_coef;

        HANDLE_ERROR( cudaMemcpy( phix, &h_phix,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( phiy, &h_phiy,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( cx, &h_cx,  sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy( cy, &h_cy,  sizeof(float), cudaMemcpyHostToDevice) );
    }

    void calibrateFromFrames(const std::vector<cv::Mat>& images, const std::string& out_path, const cv::Size& board_size=cv::Size(6,9))
    {

        // Creating vector to store vectors of 3D points for each checkerboard image
        std::vector<std::vector<cv::Point3f> > objpoints;

        // Creating vector to store vectors of 2D points for each checkerboard image
        std::vector<std::vector<cv::Point2f> > imgpoints;

        // Defining the world coordinates for 3D points
        std::vector<cv::Point3f> objp;

        for(int i=0; i<board_size.height; i++)
        {
            for(int j=0; j<board_size.width; j++)
                objp.push_back(cv::Point3f(j,i,0));
        }

        cv::Size imageSize;

        cv::Mat frame, gray;
        // vector to store the pixel coordinates of detected checker board corners
        std::vector<cv::Point2f> corner_pts;
        bool success;

        // Looping over all the images in the directory
        for(uint i=0; i<images.size(); i++)
        {
            frame = images[i].clone();
            cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

            imageSize = frame.size();

            // Finding checker board corners
            // If desired number of corners are found in the image then success = true
            success = cv::findChessboardCorners(gray, board_size, corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

            /*
              * If desired number of corner are detected,
              * we refine the pixel coordinates and display
              * them on the images of checker board
            */
            if (success)
            {
                cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

                // refining pixel coordinates for given 2d points.
                cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);

                // Displaying the detected corner points on the checker board
                cv::drawChessboardCorners(frame, board_size, corner_pts, success);

                objpoints.push_back(objp);
                imgpoints.push_back(corner_pts);
            }

            std::cout << success << std::endl;

            cv::imshow("Image",frame);
            cv::waitKey(1);
        }

        cv::destroyAllWindows();

        cv::Mat cameraMatrix,distCoeffs,R,T;

        /*
         * Performing camera calibration by
         * passing the value of known 3D points (objpoints)
         * and corresponding pixel coordinates of the
         * detected corners (imgpoints)
        */
        cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);

        std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
        std::cout << "distCoeffs : " << distCoeffs << std::endl;

        cv::Mat view, rview, map1, map2;
        float undistort_alpha = 0.5;

        // Declare what you need
        cv::FileStorage file(out_path, cv::FileStorage::WRITE);

        // Write to file!
        file << "cameraMatrix_preUndistort" << cameraMatrix;
        file << "distCoeffs" << distCoeffs;
        file << "imageSize" << imageSize;

        if (false)
        {
            cv::Mat newCamMat;
            cv::fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,
                                                                    cv::Matx33d::eye(), newCamMat, 1);
            cv::fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Matx33d::eye(), newCamMat, imageSize,
                                                 CV_16SC2, map1, map2);
        }
        else
        {
            cv::Mat newCm = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, undistort_alpha, imageSize, 0);
            cv::initUndistortRectifyMap(
                        cameraMatrix, distCoeffs, cv::Mat(),
                        newCm, imageSize,
                        CV_16SC2, map1, map2);

            file << "undistort_alpha" << undistort_alpha;
            file << "cameraMatrix_postUndistort" << newCm;
        }

        file.release();

        for(size_t i = 0; i < images.size(); i++ )
        {
            view = images[i].clone();
            if(view.empty())
                continue;
            remap(view, rview, map1, map2, cv::INTER_LINEAR);
            imshow("Image View", rview);
            cv::waitKey(0);
        }
    }


    void undistort(cv::Mat& frame, cv::Mat& uframe)
    {
        remap(frame, uframe, map1, map2, cv::INTER_LINEAR);
    }


    ~Camera()
    {
        if (cx != NULL)
            HANDLE_ERROR( cudaFree( cx ));
        if (cy != NULL)
            HANDLE_ERROR( cudaFree( cy));
        if (phix != NULL)
            HANDLE_ERROR( cudaFree( phix ));
        if (phiy != NULL)
            HANDLE_ERROR( cudaFree( phiy ));
    }
};


struct MassUndistorter
{
    Camera cam;
    MassUndistorter(const std::string& calibration_path) : cam(calibration_path) {}

    void undistort(const std::string& input_video_path, const std::string& output_video_path)
    {
        cv::VideoCapture vidIn(input_video_path);
        cv::VideoWriter vidOut;

        cv::Mat frame;

        int Nframes = vidIn.get(cv::CAP_PROP_FRAME_COUNT);
        double FPS = vidIn.get(cv::CAP_PROP_FPS);

        int width = vidIn.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = vidIn.get(cv::CAP_PROP_FRAME_HEIGHT);
        vidOut.open(output_video_path, cv::VideoWriter::fourcc('X','V','I','D'), FPS, cv::Size(width, height), true);

        for (size_t fi=0; fi<Nframes; ++fi)
        {
            vidIn >> frame;
            cam.undistort(frame, frame);
            vidOut << frame;
        }
    }
};



#endif /* CAMERA_H_ */
