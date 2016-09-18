//
// Created by zsk on 16-7-30.
//

#ifndef ORB_SLAM2_MATHUTILIS_HPP
#define ORB_SLAM2_MATHUTILIS_HPP
#include <opencv2/opencv.hpp>

class MathUtilis {

public:
    static void DrawTrackedFeatures(const cv::Mat &img1,const cv::Mat &img2,float u1,float v1,float u2, float v2);

    static void DrawQuasiFeatures(const cv::Mat &img1,const cv::Mat &img2,
                                  const cv::Mat &img3,const cv::Mat &img4,
                                  float u1,float v1,float u2, float v2,
                                  float u3,float v3,float u4, float v4);

    static cv::Mat TriangulateQuadMatch(const cv::Mat &Tcw,const float &fx,const float &fy,
                                        const float &cx,const float &cy,const float &baseline,
                                        const float &kp1x,const float &kp1y,const float &kp2x,
                                        const float &kp3x, const float &kp3y,const float &kp4x);

};


#endif //ORB_SLAM2_MATHUTILIS_HPP
