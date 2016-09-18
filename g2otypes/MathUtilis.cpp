//
// Created by zsk on 16-7-30.
//

#include "MathUtilis.h"

void MathUtilis::DrawTrackedFeatures(const cv::Mat &img1,const cv::Mat &img2,float u1,float v1,float u2, float v2)
{
    cv::Mat I1,I2;
    img1.copyTo(I1);
    img2.copyTo(I2);
    cv::circle(I1,cv::Point(u1,v1),3,cv::Scalar::all(0),-1);
    cv::imshow("img1",I1);
    cv::waitKey(33);

    cv::circle(I2,cv::Point(u2,v2),3,cv::Scalar::all(0),-1);
    cv::imshow("img2",I2);
    cv::waitKey();
}

void MathUtilis::DrawQuasiFeatures(const cv::Mat &img1,const cv::Mat &img2,
                              const cv::Mat &img3,const cv::Mat &img4,
                              float u1,float v1,float u2, float v2,
                              float u3,float v3,float u4, float v4)
{
    cv::Mat I1,I2,I3,I4;
    cv::cvtColor(img1,I1,CV_GRAY2BGR);
    cv::cvtColor(img2,I2,CV_GRAY2BGR);
    cv::cvtColor(img3,I3,CV_GRAY2BGR);
    cv::cvtColor(img4,I4,CV_GRAY2BGR);
//    img1.convertTo(I1,CV_8UC3);
//    img2.convertTo(I2,CV_8UC3);
//    img3.convertTo(I3,CV_8UC3);
//    img4.convertTo(I4,CV_8UC3);

    cv::circle(I1,cv::Point(u1,v1),3,cv::Scalar(255,0,0),-1);
    cv::imshow("img1",I1);
    cv::waitKey(33);

    cv::circle(I2,cv::Point(u2,v2),3,cv::Scalar(255,0,0),-1);
    cv::imshow("img2",I2);
    cv::waitKey(33);

    cv::circle(I3,cv::Point(u3,v3),3,cv::Scalar(255,0,0),-1);
    cv::imshow("img3",I3);
    cv::waitKey(33);

    cv::circle(I4,cv::Point(u4,v4),3,cv::Scalar(255,0,0),-1);
    cv::imshow("img4",I4);
    cv::waitKey();
}

cv::Mat MathUtilis::TriangulateQuadMatch(const cv::Mat &Tcw,const float &fx,const float &fy,
                                             const float &cx,const float &cy,const float &baseline,
                                             const float &kp1x,const float &kp1y,const float &kp2x,
                                             const float &kp3x, const float &kp3y,const float &kp4x)
{
    assert(Tcw.rows>2 && Tcw.cols>3 && "Tcw is not a pose matrix");

    cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1x-cx)/fx,
            (kp1y-cy)/fy, 1.0 );
    cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2x-cx)/fx,
            (kp1y-cy)/fy, 1.0 );
    cv::Mat xn3 = (cv::Mat_<float>(3,1) << (kp3x-cx)/fx,
            (kp3y-cy)/fy, 1.0 );
    cv::Mat xn4 = (cv::Mat_<float>(3,1) << (kp4x-cx)/fx,
            (kp3y-cy)/fy, 1.0 );
    cv::Mat P1L = cv::Mat::zeros(3,4,CV_32FC1);
    cv::Mat P1R(3,4,CV_32FC1), P2L(3,4,CV_32FC1),P2R(3,4,CV_32FC1);
    P1L.colRange(0,3) = cv::Mat::eye(3,3,CV_32FC1);
    P1L.copyTo(P1R);
    P1L.copyTo(P1R);
    P1R.at<float>(0,3) = -baseline;
    Tcw.rowRange(0,3).colRange(0,4).copyTo(P2L);
    P2L.copyTo(P2R);
    P2R.at<float>(0,3) -= baseline;
    cv::Mat A(8,4,CV_32FC1);
    A.row(0) = xn1.at<float>(0)*P1L.row(2)-P1L.row(0);
    A.row(1) = xn1.at<float>(1)*P1L.row(2)-P1L.row(1);
    A.row(2) = xn2.at<float>(0)*P1R.row(2)-P1R.row(0);
    A.row(3) = xn2.at<float>(1)*P1R.row(2)-P1R.row(1);
    A.row(4) = xn3.at<float>(0)*P2L.row(2)-P2L.row(0);
    A.row(5) = xn3.at<float>(1)*P2L.row(2)-P2L.row(1);
    A.row(6) = xn4.at<float>(0)*P2R.row(2)-P2R.row(0);
    A.row(7) = xn4.at<float>(1)*P2R.row(2)-P2R.row(1);
    cv::Mat w,u,vt;

    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    cv::Mat x3D = vt.row(3).t();
    return x3D.rowRange(0,3)/x3D.at<float>(3);
}