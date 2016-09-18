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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>


#include"Frame.h"
#include "System.h"
#include "types_stereo_direct_six_dof_expmap.h"
#include <mutex>
#include "MapPoint.h"
#include "Thirdparty/libviso2/src/viso_stereo.h"
#include <thread>

namespace ORB_SLAM2
{
class MapPoint;
class System;

class Tracking
{  

public:

    Tracking(System *pSys, const string &strSettingPath,/*Map* pMap,*/ const int sensor);

        ~Tracking();
    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

//protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

        void OptimizeKeyFrameDepth();

        void CreatNewKeyFrame();

        void TrackWithViso();

        void TrackWithVisoMutiFrames();

        bool NeedNewKeyFrame2();
//
    bool NeedNewKeyFrame();


        cv::Mat mImGray;

        g2o::Pattern<1> mPattern;

        viso2::VisualOdometryStereo* mpViso;

        int mDims[3];

        std::vector<cv::Mat> mvIncrePoses;


    // System
    System* mpSystem;


    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

        float mFastTh;

        Frame *mpLastFrame,*mpCurrentFrame;
        Frame *mpRefFrame;
        std::list<Frame*> mlpLastFrames,mlpRefFrames;

        std::list<cv::Mat> mlOptimizedPoses;
        std::list<MapPoint*> mlpMapPoints;

    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

        bool mbStartOpti, mbFinish;

        inline bool IsStartOpti()
        {
            std::unique_lock<std::mutex> lock1(mMutexStartOpti);
            return mbStartOpti;
        }

        inline void SetStartOpti(bool isStart)
        {
            std::unique_lock<std::mutex> lock1(mMutexStartOpti);
            mbStartOpti = isStart;
        }

        inline bool IsFinish()
        {
            std::unique_lock<std::mutex> lock1(mMutexStartOpti);
            return mbFinish;
        }

        inline void SetFinish(bool isFinish)
        {
            std::unique_lock<std::mutex> lock1(mMutexStartOpti);
            mbFinish = isFinish;
        }

        std::thread* mptOptiKeyFrame;
        std::mutex mMutexStartOpti;
        std::mutex mMutexLastFrames;
        std::mutex mMutexOptimizedPoses;
//    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
