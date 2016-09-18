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

#include "Frame.h"
#include "Converter.h"
#include <thread>
#include "Thirdparty/ELAS/src/elas.h"

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy,Frame::fastTh;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;


Frame::Frame()
{}



    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
                 cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &fast)
            :mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf) ,N(0),mNMatched(0),mbIsKeyFrame(false)
    {
        // Frame ID
        mnId=nNextId++;

        imLeft.copyTo(mImgPyrL[0]);
        imRight.copyTo(mImgPyrR[0]);

        if(mbInitialComputations)
        {
            ComputeImageBounds(imLeft);

            fx = K.at<float>(0,0);
            fy = K.at<float>(1,1);
            cx = K.at<float>(0,2);
            cy = K.at<float>(1,2);
            invfx = 1.0f/fx;
            invfy = 1.0f/fy;
            fastTh = fast;

            mbInitialComputations=false;
        }
        mb = mbf/fx;

        for(int i = 1;i<mnPyrLevel;i++)
        {
            cv::resize(mImgPyrL[i-1],mImgPyrL[i],cv::Size(),.5,.5);
            cv::resize(mImgPyrR[i-1],mImgPyrR[i],cv::Size(),.5,.5);
        }


    }

//    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, cv::Mat &K, cv::Mat &distCoef, const float &bf)
//            :mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), N(0),mNMatched(0),mbIsKeyFrame(false)
//    {
//        // Frame ID
//        mnId = nNextId++;
//
//        imLeft.copyTo(mImgPyrL[0]);
//        imRight.copyTo(mImgPyrR[0]);
//
//
//        for(int i = 1;i<mnPyrLevel;i++)
//        {
//            cv::resize(mImgPyrL[i-1],mImgPyrL[i],cv::Size(),.5,.5);
//            cv::resize(mImgPyrR[i-1],mImgPyrR[i],cv::Size(),.5,.5);
//        }
//
//
//        if(mbInitialComputations)
//        {
//            ComputeImageBounds(imLeft);
//
//            fx = K.at<float>(0,0);
//            fy = K.at<float>(1,1);
//            cx = K.at<float>(0,2);
//            cy = K.at<float>(1,2);
//            invfx = 1.0f/fx;
//            invfy = 1.0f/fy;
//
//            mbInitialComputations=false;
//        }
//
//        mb = mbf/fx;
//
//
//        const int width = imLeft.cols,
//                height = imLeft.rows;
//        const int32_t dims[3] = {width,height,width};
//        Elas::parameters param;
//        param.postprocess_only_left = false;
//        Elas elas(param);
//        cv::Mat D1(imLeft.rows,imLeft.cols,CV_32FC1),D2(imLeft.rows,imLeft.cols,CV_32FC1);
//        float *D1_data = (float*)D1.data,
//                *D2_data = (float*)D2.data;
//        elas.process(imLeft.data,imRight.data,D1_data,D2_data,dims);
//
//
//
//        FAST(imLeft,mvKeys,fastTh);
//
//        UndistortKeyPoints();
//
//        N = mvKeys.size();
//        mvLives.resize(N,0);
//        mvbIsFixed.resize(N,false);
//        mvDepth.resize(N,-1);
//        mvuRight.resize(N,-1);
//        mvbIsValid.resize(N,true);
//        mvpMapPoints.resize(N, nullptr);
//
//
//        for(int i = 0,imax = mvKeys.size();i<imax;i++)
//        {
//
//            const cv::KeyPoint &kp = mvKeys[i];
//            const float u = kp.pt.x,
//                    v = kp.pt.y;
//
//            const float &disparity = D1.at<float>(floorf(v),floorf(u));
//            if(disparity<0) continue;
//            const float ur = u - disparity;
//            mvDepth[i] = mbf/disparity;
//            mvuRight[i] = ur;
//            mNMatched++;
//        }
//    }

    void Frame::AddSteroMatch(const float &u,const float &v, const float &ur,const float &depth,const int &live)
    {
        cv::KeyPoint a,b;
        a.pt.x = u;
        a.pt.y = v;
        b.pt.x = ur;
        b.pt.y = v;
        mvKeys.push_back(a);
        mvKeysUn.push_back(a);
        mvKeysRight.push_back(b);
        mvuRight.push_back(ur);
        mvDepth.push_back(depth);
        mvLives.push_back(live);
        mvbIsFixed.push_back(false);

        ++mNMatched;
        ++N;
    }

    void Frame::AddSteroMatch(const float &u,const float &v, const float &ur,const float &depth,
                              const int &live,const int &refIdxL,bool isFix)
    {
        cv::KeyPoint a,b;
        a.pt.x = u;
        a.pt.y = v;
        b.pt.x = ur;
        b.pt.y = v;
        mvKeys.push_back(a);
        mvKeysUn.push_back(a);
        mvKeysRight.push_back(b);
        mvuRight.push_back(ur);
        mvDepth.push_back(depth);
        mvLives.push_back(live);
        mvbIsFixed.push_back(isFix);
        mvRefMatchedIdx.push_back(refIdxL);
        mvbIsValid.push_back(true);
        ++mNMatched;
        ++N;
    }

    void Frame::RefreshKeyPoints()
    {
        mvLives.clear();
        mvuRight.clear();
        mvDepth.clear();
        mvpMapPoints.clear();

        const cv::Mat &imLeft = mImgPyrL[0];
        const cv::Mat &imRight = mImgPyrR[0];
        imLeft.copyTo(mImgPyrL[0]);
        imRight.copyTo(mImgPyrR[0]);

        const int width = imLeft.cols,
                height = imLeft.rows;
        const int32_t dims[3] = {width,height,width};
        Elas::parameters param;
        param.postprocess_only_left = false;
        Elas elas(param);
        cv::Mat D1(imLeft.rows,imLeft.cols,CV_32FC1),D2(imLeft.rows,imLeft.cols,CV_32FC1);
        float *D1_data = (float*)D1.data,
                *D2_data = (float*)D2.data;
        elas.process(imLeft.data,imRight.data,D1_data,D2_data,dims);

        FAST(imLeft,mvKeys,fastTh);

        UndistortKeyPoints();

        N = mvKeys.size();
        mvLives.resize(N,0);
        mvbIsFixed.resize(N,false);
        mvDepth.resize(N,-1);
        mvuRight.resize(N,-1);
        mvbIsValid.resize(N,true);
        mvpMapPoints.resize(N, nullptr);


        for(int i = 0,imax = mvKeys.size();i<imax;i++)
        {

            const cv::KeyPoint &kp = mvKeys[i];
            const float u = kp.pt.x,
                    v = kp.pt.y;

            const float &disparity = D1.at<float>(floorf(v),floorf(u));
            if(disparity<0) continue;
            const float ur = u - disparity;
            mvDepth[i] = mbf/disparity;
            mvuRight[i] = ur;
            mNMatched++;
        }
    }


    void Frame::AddKeyPoints()
    {
        const cv::Mat &imLeft = mImgPyrL[0];
        const cv::Mat &imRight = mImgPyrR[0];
        const int width = imLeft.cols,
                height = imLeft.rows;
        const int32_t dims[3] = {width,height,width};
        Elas::parameters param;
        param.postprocess_only_left = false;
        Elas elas(param);
        cv::Mat D1(height,width,CV_32FC1),D2(height,width,CV_32FC1);
        float *D1_data = (float*)D1.data,
                *D2_data = (float*)D2.data;
        elas.process(imLeft.data,imRight.data,D1_data,D2_data,dims);

        std::vector<cv::KeyPoint> kpCandidates;
        FAST(imLeft, kpCandidates, fastTh);
        cv::Mat candidatePoints(kpCandidates.size(),2,CV_32FC1);
        std::vector<bool> vIsRemoved(kpCandidates.size(),false);

        if(mvKeys.size()>0)
        {
            for(int i = 0,imax = kpCandidates.size();i<imax;i++)
            {
                candidatePoints.at<float>(i,0) = kpCandidates[i].pt.x;
                candidatePoints.at<float>(i,1) = kpCandidates[i].pt.y;
            }
            cv::flann::KDTreeIndexParams indexParams(2);
            cv::flann::Index kdtree(candidatePoints, indexParams);

            cv::flann::SearchParams params(128);
            const float radius = 2;
            for(int i = 0,imax = mvKeys.size();i<imax;i++)
            {
                std::vector<float> query;
                const cv::KeyPoint &kp = mvKeys[i];
                query.push_back(kp.pt.x);
                query.push_back(kp.pt.y);
                std::vector<int> indices;//找到点的索引
                std::vector<float> dists;
                kdtree.radiusSearch(query, indices, dists, radius, 10, params);
                for(int j = 0,jmax = indices.size();j<jmax;j++)
                {
                    if(indices[j] == 0)
                        break;
                    vIsRemoved[indices[j]] = true;
                }
            }
        }


        for(int i = 0,imax = vIsRemoved.size();i<imax;i++)
        {
            if(vIsRemoved[i]) continue;
            const cv::KeyPoint &kp = kpCandidates[i];
            const float u = kp.pt.x,
                    v = kp.pt.y;

            const float &disparity = D1.at<float>(floorf(v),floorf(u));
            if(disparity<0) continue;

            const float ur = u - disparity;
            mvKeys.push_back(kpCandidates[i]);
            mvDepth.push_back(mbf/disparity);
            mvuRight.push_back(ur);
            mvLives.push_back(0);
            mvbIsFixed.push_back(false);
            mvbIsValid.push_back(true);
            mvpMapPoints.push_back(nullptr);
            mNMatched++;
        }
        UndistortKeyPoints();
        N = mvKeys.size();

    }




void Frame::SetPose(const cv::Mat &Tcw)
{


    {
        std::unique_lock<std::mutex> lock1(mMutexTcw);
        mTcw = Tcw.clone();
    }

//    mRcw = mTcw.rowRange(0,3).colRange(0,3);
//    mRwc = mRcw.t();
//    mtcw = mTcw.rowRange(0,3).col(3);
//    mOw = -mRcw.t()*mtcw;

    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{
    std::unique_lock<std::mutex> lock1(mMutexTcw);
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}



void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}



cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

    cv::Mat Frame::UnprojectStereoInCamera(const int &i)
    {
        const float z = mvDepth[i];
        if(z>0)
        {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u-cx)*z*invfx;
            const float y = (v-cy)*z*invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
            return x3Dc;
        }
        else
            return cv::Mat();
    }

    cv::Mat Frame::UnprojectStereoInCamera(const float &u, const float &v,
                                        const float &z)
    {

        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return x3Dc;
    }
} //namespace ORB_SLAM
