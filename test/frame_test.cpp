//
// Created by zsk on 16-7-23.
//

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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include <System.h>
#include <fstream>
using namespace std;


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

void TrackImages(vector<string> &vstrImageLeft, vector<string> &vstrImageRight,
                 vector<double> &vTimestamps, ORB_SLAM2::Tracking *pTracker,
                 ofstream &out);
int main(int argc, char **argv)
{

    if(argc != 3)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[2]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    ofstream out("position_out.txt");
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], ORB_SLAM2::System::STEREO, false);

    ORB_SLAM2::Tracking *pTracker = SLAM.mpTracker;
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    std::thread t1(&TrackImages,ref(vstrImageLeft),ref(vstrImageRight),ref(vTimestamps),pTracker,ref(out));
    std::thread t2(&ORB_SLAM2::Tracking::OptimizeKeyFrameDepth,pTracker);
    t1.join();
    t2.join();
//    pTracker->SetFinish(true);

    return 0;
}

void TrackImages(vector<string> &vstrImageLeft, vector<string> &vstrImageRight,
                 vector<double> &vTimestamps, ORB_SLAM2::Tracking *pTracker,
                ofstream &out)
{
    // Main loop
    cv::Mat imLeft, imRight;
    int nImages = vstrImageLeft.size();
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return;
        }
        cout << "-----------------------" << endl;
        cout << "img idx:" << ni << endl;
        pTracker->GrabImageStereo(imLeft, imRight, tframe);

//        const cv::Mat &OW = pTracker->mpCurrentFrame->mOw;
//        out << ni << " " << OW.at<float>(0) << " " << OW.at<float>(1) << " " << OW.at<float>(2) << endl;
//        cv::Mat Tcw = pTracker->mpCurrentFrame->GetInversePose();
//
//        out /*<< ni << " " */<< Tcw.at<float>(0,0) << " " << Tcw.at<float>(0,1) << " " << Tcw.at<float>(0,2) << " " << Tcw.at<float>(0,3)
//                << " " << Tcw.at<float>(1,0) << " " << Tcw.at<float>(1,1) << " " << Tcw.at<float>(1,2) << " " << Tcw.at<float>(1,3)
//                << " " << Tcw.at<float>(2,0) << " " << Tcw.at<float>(2,1) << " " << Tcw.at<float>(2,2) << " " << Tcw.at<float>(2,3) << endl;

        {
            unique_lock<mutex> lock1(pTracker->mMutexOptimizedPoses);
            while(!pTracker->mlOptimizedPoses.empty())
            {

                const cv::Mat& Tcw = pTracker->mlOptimizedPoses.front();
                out /*<< ni << " " */<< Tcw.at<float>(0,0) << " " << Tcw.at<float>(0,1) << " " << Tcw.at<float>(0,2) << " " << Tcw.at<float>(0,3)
                                     << " " << Tcw.at<float>(1,0) << " " << Tcw.at<float>(1,1) << " " << Tcw.at<float>(1,2) << " " << Tcw.at<float>(1,3)
                                     << " " << Tcw.at<float>(2,0) << " " << Tcw.at<float>(2,1) << " " << Tcw.at<float>(2,2) << " " << Tcw.at<float>(2,3) << endl;
                pTracker->mlOptimizedPoses.pop_front();
            }
        }

    }
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}
