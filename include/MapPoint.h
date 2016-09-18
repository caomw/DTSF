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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"Frame.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{


class Frame;


class MapPoint
{
public:

    MapPoint(const float &d, Frame* pFrame, const int &idxF);

        void AddObservation(Frame* pFrame, const int &idx);



public:

        float md;

        int mnExtractFrameId;

     // Keyframes observing the point and associated index in keyframe
//     std::map<KeyFrame*,size_t> mObservations;
    std::map<Frame*,size_t> mObservations;

};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
