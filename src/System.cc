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



#include "System.h"
#include "Converter.h"
#include <thread>


namespace ORB_SLAM2
{

    System::System(const string &strSettingsFile, const eSensor sensor,
                       const bool bUseViewer):
        mSensor(sensor)
    {

        cout << "Input sensor was set to: ";


        //Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }


        //(it will live in the main thread of execution, the one that called this constructor)
        mpTracker = new Tracking(this, strSettingsFile,mSensor);


    }

    System::~System()
    {

        delete mpTracker;
    }


} //namespace ORB_SLAM
