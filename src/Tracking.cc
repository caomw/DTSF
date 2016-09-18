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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>


#include"Converter.h"

#include"Optimizer.h"

#include<iostream>

#include<mutex>
using namespace std;

namespace ORB_SLAM2
{

    Tracking::Tracking(System *pSys, const string &strSettingPath, const int sensor):
            mState(NO_IMAGES_YET), mSensor(sensor),
            mpSystem(pSys),
            mnLastRelocFrameId(0),mbStartOpti(false),
            mbFinish(false)
    {
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4,1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3!=0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if(fps==0)
            fps=30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;
        mFastTh = fSettings["FastTh"];


        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if(DistCoef.rows==5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;
        cout << "- fast threshold: " << mFastTh << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if(mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;


        mPattern.mPatternSize = 1;
        mPattern.mBorder = 3;

        mPattern.mPatternPoints(0,0) = 0;
        mPattern.mPatternPoints(0,1) = 0;


        viso2::VisualOdometryStereo::parameters param;
        param.calib.f  = fx; // focal length in pixels
        param.calib.cu = cx; // principal point (u-coordinate) in pixels
        param.calib.cv = cy; // principal point (v-coordinate) in pixels
        param.base     = mbf/fx; // baseline in meters
        mpViso = new viso2::VisualOdometryStereo(param);

//        mptOptiKeyFrame = new std::thread(&Tracking::OptimizeKeyFrameDepth,this);
//        mptOptiKeyFrame->join();


//        mPattern.mPatternSize = 8;
//        mPattern.mBorder = 3;
//
//        mPattern.mPatternPoints(0,0) = 0;
//        mPattern.mPatternPoints(0,1) = 0;
//
//        mPattern.mPatternPoints(1,0) = 0;
//        mPattern.mPatternPoints(1,1) = -2;
//
//        mPattern.mPatternPoints(2,0) = -1;
//        mPattern.mPatternPoints(2,1) = -1;
//
//        mPattern.mPatternPoints(3,0) = -2;
//        mPattern.mPatternPoints(3,1) = 0;
//
//        mPattern.mPatternPoints(4,0) = -1;
//        mPattern.mPatternPoints(4,1) = 1;
//
//        mPattern.mPatternPoints(5,0) = 0;
//        mPattern.mPatternPoints(5,1) = 2;
//
//        mPattern.mPatternPoints(6,0) = 1;
//        mPattern.mPatternPoints(6,1) = 1;
//
//        mPattern.mPatternPoints(7,0) = 2;
//        mPattern.mPatternPoints(7,1) = 0;

//        mPattern.mPatternSize = 16;
//        mPattern.mBorder = 3;
//        int idx = 0;
//        for(int i = -2;i<2;i++)
//            for(int j = -2;j<2;j++)
//            {
//                mPattern.mPatternPoints(idx,0) = i;
//                mPattern.mPatternPoints(idx++,1) = j;
//
//            }
//        cv::Mat iniPose(cv::Mat::eye(4,4,CV_32FC1));

    }

    Tracking::~Tracking()
    {

        SetFinish(true);
//        delete mptOptiKeyFrame;
        for(auto pF:mlpLastFrames)
        {
            delete pF;
        }
        for(auto pF:mlpRefFrames)
        {
            delete pF;
        }
        for(auto mp:mlpMapPoints)
        {
            delete mp;
        }
        mlpRefFrames.clear();
        delete mpViso;
    }
    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
    {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;

        if(mImGray.channels()==3)
        {
            if(mbRGB)
            {
                cvtColor(mImGray,mImGray,CV_RGB2GRAY);
                cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
            }
            else
            {
                cvtColor(mImGray,mImGray,CV_BGR2GRAY);
                cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
            }
        }
        else if(mImGray.channels()==4)
        {
            if(mbRGB)
            {
                cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
                cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
            }
            else
            {
                cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
                cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
            }
        }

//        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mK,mDistCoef,mbf,mThDepth);
//        if(mState==NO_IMAGES_YET)
//        {
//            //这里有一个深海巨坑
//            //直接用new取创建一个frame类指针，就无法计算立体匹配结果
//            //这里是为什么？
//            mpCurrentFrame = new Frame(mImGray,imGrayRight,timestamp,mK,mDistCoef,mbf);
////            mpCurrentFrame = new Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mK,mDistCoef,mbf,mThDepth);
////            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mK,mDistCoef,mbf,mThDepth);
////            mpCurrentFrame = &mCurrentFrame;
//        }
//        else
        {
            mpCurrentFrame = new Frame(mImGray,imGrayRight,timestamp,mK,mDistCoef,mbf,mFastTh);
        }

//        TrackWithViso();
//        Track();
        TrackWithVisoMutiFrames();

        return mpCurrentFrame->mTcw.clone();
    }


    void Tracking::TrackWithVisoMutiFrames()
    {
        if(mState==NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }


        if(mState==NOT_INITIALIZED)
        {

            mpCurrentFrame->SetPose(cv::Mat::eye(4,4,CV_32F));

            mvIncrePoses.push_back(mpCurrentFrame->mTcw);
            mpLastFrame = mpCurrentFrame;
            mlpLastFrames.push_back(mpCurrentFrame);
            mpRefFrame = mpCurrentFrame;
            mpCurrentFrame->mbIsKeyFrame = true;


            cv::Mat imgL,imgR;

            mpCurrentFrame->mImgPyrL[0].copyTo(imgL);
            mpCurrentFrame->mImgPyrR[0].copyTo(imgR);
            mDims[0] = imgL.cols;
            mDims[1] = imgL.rows;
            mDims[2] = imgL.cols;
            mpViso->process(imgL.data,imgR.data,mDims);

            for(int i = 0;i < mpRefFrame->N;i++)
            {
                if(mpRefFrame->mvDepth[i]>0)
                {
                    MapPoint* mp = new MapPoint(mpRefFrame->mvDepth[i],mpRefFrame,i);
                    mlpMapPoints.push_back(mp);
//                    mpRefFrame->mvpMapPoints[i] = mp;
                }
            }
//            cv::imshow("aa",mLastFrame.mImgPyrL[1]);
//            cv::waitKey();
            mState = OK;
            return;
//            if(mState!=OK)
//                return;
        }
        else
        {
            // System is initialized. Track Frame.
            cv::Mat imgL,imgR;
            mpCurrentFrame->mImgPyrL[0].copyTo(imgL);
            mpCurrentFrame->mImgPyrR[0].copyTo(imgR);
            if(mpViso->process(imgL.data,imgR.data,mDims))
            {
                cv::Mat cvPoseIncre(cv::Mat::eye(4,4,CV_32F));
                viso2::Matrix poseIncre = mpViso->getMotion();
                for(int i = 0;i<3;i++)
                    for(int j = 0;j<4;j++)
                    {
                        cvPoseIncre.at<float>(i,j) = poseIncre.val[i][j];
                    }
                mvIncrePoses.push_back(cvPoseIncre*mvIncrePoses.back());
//                cout << "cvPoseIncre: " << cvPoseIncre <<endl;
                mpCurrentFrame->SetPose(cvPoseIncre*mpLastFrame->mTcw);
//                cout << mpCurrentFrame->GetInversePose() << endl;
                mlpLastFrames.push_back(mpCurrentFrame);
//                delete mpLastFrame;
                mpLastFrame = mpCurrentFrame;
                if(NeedNewKeyFrame2())
                {
                    mpRefFrame = mpCurrentFrame;
                    while(IsStartOpti())
                        usleep(30);
                    auto incre = mvIncrePoses.rbegin();
                    auto frameIter = mlpLastFrames.rbegin();
                    cv::Mat &T =mlpLastFrames.front()->mTcw;
                    for(;incre!=mvIncrePoses.rend();incre++,frameIter++)
                    {
                        (*frameIter)->SetPose((*incre)*T);
                    }
                    mvIncrePoses.clear();
                    mvIncrePoses.push_back(cv::Mat::eye(4,4,CV_32FC1));
                    SetStartOpti(true);


                    /*mlpLastFrames.front()->RefreshKeyPoints();
//                    int trackedFeatures = Optimizer::PoseEstimationDirectDepthPyr(mpRefFrame,mpCurrentFrame,&mPattern);
//                    cout << "pose before: " << mpCurrentFrame->mTcw << endl;
//                    int trackedFeatures = Optimizer::PoseEstimationDirectDepth2(mpRefFrame,mpCurrentFrame,&mPattern);
//                    int trackedFeatures = Optimizer::PoseEstimationDirectDepthAffine(mpRefFrame,mpCurrentFrame,&mPattern);
                    int trackedFeatures = Optimizer::OptimizeKeyFrameDepth(mlpLastFrames,&mPattern);
//                    int trackedFeatures = Optimizer::OptimizeKeyFrameDepthWithAffine(mlpLastFrames,&mPattern);
                    while(mlpLastFrames.size() > 1)
                    {
                        Frame* pFrame = mlpLastFrames.front();
                        mlpLastFrames.pop_front();
                        mlOptimizedPoses.push_back(pFrame->GetInversePose());
                        delete pFrame;
                    }
//                    cout << "trackedFeatures: " << trackedFeatures << endl;
//                    delete mpRefFrame;
                    cout << "pose after: " << mpCurrentFrame->GetInversePose() << endl;
                    mpRefFrame = mpCurrentFrame;*/
                }
//                if(!mpLastFrame->mbIsKeyFrame)
//                    delete mpLastFrame;
//                mlpLastFrames.push_back(mpCurrentFrame);
//                mpLastFrame = mpCurrentFrame;
            }
        }

    }


    void Tracking::OptimizeKeyFrameDepth()
    {
        while(1)
        {
            if(IsStartOpti())
            {

                std::list<Frame*> lastFrames;
                while(mlpLastFrames.size() > 1)
                {
                    Frame* pFrame = mlpLastFrames.front();
                    mlpLastFrames.pop_front();
                    lastFrames.push_back(pFrame);
                }
                lastFrames.push_back(mlpLastFrames.front());
                cout << "lastFrames.front()->RefreshKeyPoints()" << endl;
                lastFrames.front()->RefreshKeyPoints();
//                    int trackedFeatures = Optimizer::PoseEstimationDirectDepthPyr(mpRefFrame,mpCurrentFrame,&mPattern);
//                    cout << "pose before: " << mpCurrentFrame->mTcw << endl;
//                    int trackedFeatures = Optimizer::PoseEstimationDirectDepth2(mpRefFrame,mpCurrentFrame,&mPattern);
//                    int trackedFeatures = Optimizer::PoseEstimationDirectDepthAffine(mpRefFrame,mpCurrentFrame,&mPattern);
//                cout << "OptimizeKeyFrameDepth()" << endl;
                int trackedFeatures = Optimizer::OptimizeKeyFrameDepth(lastFrames,&mPattern);
//                    int trackedFeatures = Optimizer::OptimizeKeyFrameDepthWithAffine(mlpLastFrames,&mPattern);

//                    cout << "trackedFeatures: " << trackedFeatures << endl;
//                    delete mpRefFrame;
                cout << "pose after: " << lastFrames.back()->GetInversePose() << endl;
                while(lastFrames.size() > 1)
                {
                    Frame* pFrame = lastFrames.front();
                    lastFrames.pop_front();
                    {
                        std::unique_lock<std::mutex> lock1(mMutexOptimizedPoses);
                        mlOptimizedPoses.push_back(pFrame->GetInversePose());
                    }
                    delete pFrame;
                }
                SetStartOpti(false);
            }
            else
                usleep(20);

            if(IsFinish())
            {
                break;
            }

        }

    }


    bool Tracking::NeedNewKeyFrame2()
    {
        const cv::Mat &increT = mvIncrePoses.back();
        double linearDist = cv::norm(increT.rowRange(0,3).col(3));
        const cv::Mat &R = increT.rowRange(0,3).colRange(0,3);
        cv::Mat rvec;
        cv::Rodrigues(R,rvec);
        double rotationDist = fabs(min(cv::norm(rvec),2*M_PI-cv::norm(rvec)));
        std::cout << "linearDist: " << linearDist << std::endl;
        std::cout << "rotationDist: " << rotationDist << std::endl;
        return linearDist > 3 || rotationDist > 0.3
            /*|| static_cast<float>(mpCurrentFrame->mNMatched)/static_cast<float>(mpRefFrame->mNMatched)<0.6*/;
    }

    bool Tracking::NeedNewKeyFrame()
    {
        cv::Mat curPos = mpCurrentFrame->GetCameraCenter();
        cv::Mat refPos = mpRefFrame->GetCameraCenter();
        double linearDist = cv::norm(curPos,refPos);
        cv::Mat R = mpRefFrame->GetRotation()*mpCurrentFrame->GetRotationInverse();
        cv::Mat rvec;
        cv::Rodrigues(R,rvec);
        double rotationDist = fabs(min(cv::norm(rvec),2*M_PI-cv::norm(rvec)));
        std::cout << "linearDist: " << linearDist << std::endl;
        std::cout << "rotationDist: " << rotationDist << std::endl;
        return linearDist > 3 || rotationDist > 0.3
            /*|| static_cast<float>(mpCurrentFrame->mNMatched)/static_cast<float>(mpRefFrame->mNMatched)<0.6*/;
    }

    void Tracking::TrackWithViso()
    {
        if(mState==NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }


        if(mState==NOT_INITIALIZED)
        {

            mpCurrentFrame->SetPose(cv::Mat::eye(4,4,CV_32F));

            mpLastFrame = mpCurrentFrame;
            mlpLastFrames.push_back(mpCurrentFrame);
            mpRefFrame = mpCurrentFrame;
            mpCurrentFrame->mbIsKeyFrame = true;


            cv::Mat imgL,imgR;

            mpCurrentFrame->mImgPyrL[0].copyTo(imgL);
            mpCurrentFrame->mImgPyrR[0].copyTo(imgR);
            mDims[0] = imgL.cols;
            mDims[1] = imgL.rows;
            mDims[2] = imgL.cols;
            mpViso->process(imgL.data,imgR.data,mDims);

            for(int i = 0;i < mpRefFrame->N;i++)
            {
                if(mpRefFrame->mvDepth[i]>0)
                {
                    MapPoint* mp = new MapPoint(mpRefFrame->mvDepth[i],mpRefFrame,i);
                    mlpMapPoints.push_back(mp);
//                    mpRefFrame->mvpMapPoints[i] = mp;
                }
            }
//            cv::imshow("aa",mLastFrame.mImgPyrL[1]);
//            cv::waitKey();
            mState = OK;
            return;
//            if(mState!=OK)
//                return;
        }
        else
        {
            // System is initialized. Track Frame.
            cv::Mat imgL,imgR;
            mpCurrentFrame->mImgPyrL[0].copyTo(imgL);
            mpCurrentFrame->mImgPyrR[0].copyTo(imgR);
            if(mpViso->process(imgL.data,imgR.data,mDims))
            {
                cv::Mat cvPoseIncre(cv::Mat::eye(4,4,CV_32F));
                viso2::Matrix poseIncre = mpViso->getMotion();
                for(int i = 0;i<3;i++)
                    for(int j = 0;j<4;j++)
                    {
                        cvPoseIncre.at<float>(i,j) = poseIncre.val[i][j];
                    }
//                cout << "cvPoseIncre: " << cvPoseIncre <<endl;
                mpCurrentFrame->SetPose(cvPoseIncre*mpLastFrame->mTcw);
                cout << mpCurrentFrame->GetInversePose() << endl;
                if(NeedNewKeyFrame())
                {
//                    mpRefFrame->AddKeyPoints();
                    mpRefFrame->RefreshKeyPoints();
//
//                    int trackedFeatures = Optimizer::PoseEstimationDirectDepthPyr(mpRefFrame,mpCurrentFrame,&mPattern);
//                    cout << "pose before: " << mpCurrentFrame->mTcw << endl;
                    int trackedFeatures = Optimizer::PoseEstimationDirectDepth2(mpRefFrame,mpCurrentFrame,&mPattern);
//                    int trackedFeatures = Optimizer::PoseEstimationDirectDepthAffine(mpRefFrame,mpCurrentFrame,&mPattern);

                    cout << "trackedFeatures: " << trackedFeatures << endl;
                    delete mpRefFrame;
                    cout << "pose after: " << mpCurrentFrame->GetInversePose() << endl;
                    mpRefFrame = mpCurrentFrame;
                    mpCurrentFrame->mbIsKeyFrame = true;
                }
                if(!mpLastFrame->mbIsKeyFrame)
                    delete mpLastFrame;
                mlpLastFrames.push_back(mpCurrentFrame);
                mpLastFrame = mpCurrentFrame;
            }
        }

    }

    void Tracking::Track()
    {
        if(mState==NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }


        if(mState==NOT_INITIALIZED)
        {

            mpCurrentFrame->SetPose(cv::Mat::eye(4,4,CV_32F));

            mpLastFrame = mpCurrentFrame;

            mpRefFrame = mpCurrentFrame;
            mpCurrentFrame->mbIsKeyFrame = true;


            cv::Mat imgL,imgR;
            mDims[0] = imgL.cols;
            mDims[1] = imgL.rows;
            mDims[2] = imgL.cols;
            mpCurrentFrame->mImgPyrL[0].copyTo(imgL);
            mpCurrentFrame->mImgPyrR[0].copyTo(imgR);
            mpViso->process(imgL.data,imgR.data,mDims);

            for(int i = 0;i < mpRefFrame->N;i++)
            {
                if(mpRefFrame->mvDepth[i]>0)
                {
                    MapPoint* mp = new MapPoint(mpRefFrame->mvDepth[i],mpRefFrame,i);
                    mlpMapPoints.push_back(mp);
//                    mpRefFrame->mvpMapPoints[i] = mp;
                }
            }
//            cv::imshow("aa",mLastFrame.mImgPyrL[1]);
//            cv::waitKey();
            mState = OK;
            return;
//            if(mState!=OK)
//                return;
        }
        else
        {
            // System is initialized. Track Frame.

//            cv::imshow("aa",mCurrentFrame.mImgPyrL[1]);
//            cv::waitKey();
//            mVelocity = mMotionModel.PredictedVelocity();
            mpCurrentFrame->SetPose(/*mVelocity**/mpLastFrame->mTcw);
//            DirectTrackWithMotionModel();
//            int trackedFeatures = Optimizer::PoseEstimationDirectDepthPyr(mpRefFrame,mpCurrentFrame,&mPattern);
//            int trackedFeatures = Optimizer::PoseEstimationDirectDepth2(mpLastFrame,mpCurrentFrame,&mPattern);
//            int trackedFeatures = Optimizer::PoseEstimationDirectDepthAffine(mpLastFrame,mpCurrentFrame,&mPattern);
//            int trackedFeatures = Optimizer::poseEstimationDirect(mpLastFrame,mpCurrentFrame,&mPattern);
//            cout << mpCurrentFrame->mTcw << endl;
            cout << mpCurrentFrame->GetInversePose() << endl;
//            mLastFrame = Frame(mCurrentFrame);
//            mMotionModel.UpdateCameraPose(mpCurrentFrame->mTcw);

//            if(NeedNewKeyFrame())
//            {
//                int i = mpCurrentFrame->mvDepth.size();
//                mpCurrentFrame->AddKeyPoints();
//
//                for(;i<mpCurrentFrame->N;i++)
//                {
//                    if(mpCurrentFrame->mvDepth[i]>0)
//                    {
//                        MapPoint* mp = new MapPoint(mpCurrentFrame->mvDepth[i],mpCurrentFrame,i);
//                        mlpMapPoints.push_back(mp);
//                    }
//                }
//                mpCurrentFrame->mbIsKeyFrame = true;
//                mpRefFrame = mpCurrentFrame;
////                cout << "insert new key frame" << endl;
////                CreatNewKeyFrame();
////                int trackedFeatures = Optimizer::PoseEstimationMultiFrames(mlpLastFrames,&mPattern);
////                Optimizer::OptimizeKeyFrameDepth(mlpLastFrames,&mPattern);
////                Optimizer::OptimizeKeyFrameDepthWithAffine(mlpLastFrames,&mPattern);
////                cout << "multiple optimization" << endl;
//
//            }

//            mpCurrentFrame->RefreshKeyPoints(mpORBextractorLeft, mpORBextractorRight);
            mpCurrentFrame->RefreshKeyPoints();

            {
//                if(!mpLastFrame->mbIsKeyFrame)
                delete mpLastFrame;
//                if(mpCurrentFrame->mnId%3 == 0)
//                {
//                    int i = mpCurrentFrame->mvDepth.size();
//                    mpCurrentFrame->AddKeyPoints(mpORBextractorLeft,mpORBextractorRight);
//                    for(;i<mpCurrentFrame->N;i++)
//                    {
//                        if(mpCurrentFrame->mvDepth[i]>0)
//                        {
//                            MapPoint* mp = new MapPoint(mpCurrentFrame->mvDepth[i],mpCurrentFrame,i);
//                            mlpMapPoints.push_back(mp);
//                        }
//                    }
//                }

//                mlpLastFrames.push_back(mpLastFrame);
                mpLastFrame = mpCurrentFrame;

            }

        }


    }




} //namespace ORB_SLAM
