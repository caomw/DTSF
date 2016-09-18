/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

/**
* The origninal file is from ORB-SLAM2.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include <mutex>

namespace ORB_SLAM2
{
    using namespace std;

class MapPoint;

class Frame
{
public:
    Frame();

        Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
              cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &fast);

//        Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
//              cv::Mat &K, cv::Mat &distCoef, const float &bf);

        void AddKeyPoints();

        void RefreshKeyPoints();

    // Set the camera pose.
    void SetPose(const cv::Mat &Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        std::unique_lock<std::mutex> lock1(mMutexTcw);
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        std::unique_lock<std::mutex> lock1(mMutexTcw);
        return mRwc.clone();
    }

        inline cv::Mat GetRotation()
        {
            std::unique_lock<std::mutex> lock1(mMutexTcw);
            return mTcw.rowRange(0,3).colRange(0,3).clone();
        }

        inline cv::Mat GetInversePose()
        {
            cv::Mat Twc(cv::Mat::eye(4,4,CV_32FC1));
            {
                std::unique_lock<std::mutex> lock1(mMutexTcw);
                mRwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
                mOw.copyTo(Twc.col(3).rowRange(0,3));
            }
            return Twc;
        }

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

    // Backprojects a keypoint (if stereo/depth info available) into left camera coordinates.
    cv::Mat UnprojectStereoInCamera(const int &i);

        static cv::Mat UnprojectStereoInCamera(const float &u, const float &v,
                                            const float &z);

        void AddSteroMatch(const float &u,const float &v, const float &ur,const float &depth,const int &live);

        void AddSteroMatch(const float &u,const float &v, const float &ur,const float &depth,
                           const int &live,const int &refIdxL,bool isFix);

public:

    //
    const static int mnPyrLevel{2};

    cv::Mat mImgPyrL[mnPyrLevel],mImgPyrR[mnPyrLevel];

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

        static float fastTh;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int N;

    // Number of stereo matched keypoints
    int mNMatched;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;
        std::vector<int> mvLives;
        std::vector<int> mvRefMatchedIdx;
        std::vector<bool> mvbIsFixed,mvbIsValid;



    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;


    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    bool mbIsKeyFrame;


    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;


//private:

    // Undistort keypoints given OpenCV distortion parameters.
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();
        std::mutex mMutexTcw;

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc
};

}// namespace ORB_SLAM

#endif // FRAME_H
