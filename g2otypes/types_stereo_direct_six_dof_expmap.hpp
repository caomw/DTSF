//
// Created by zsk on 16-7-26.
//


template <int D>
bool EdgeDirectBase<D>::IsOutofBorder(int rows,int cols,int u,int v)
{
    return (u - mPattern->mBorder<0 || ( u+mPattern->mBorder ) >cols ||
            ( v - mPattern->mBorder ) <0 || ( v+mPattern->mBorder ) >rows);
}

template <int D>
bool EdgeDirectBase<D>::IsOutofBorder(int rows,int cols,int u,int v, int ur)
{
    return (u - mPattern->mBorder<0 || ( u+mPattern->mBorder ) >cols ||
            ( v - mPattern->mBorder ) <0 || ( v+mPattern->mBorder ) >rows
            || ur - mPattern->mBorder < 0 || ur+mPattern->mBorder >= cols);
}


//u_ref,v_ref当前层参考特征点的坐标
//ftrPos在参考左相机相机系下的特征坐标
template <int D>
void EdgeDirectBase<D>::precomputeJacobianLeft(const cv::Mat& ref_img, /*const cv::Mat &ftrPos,*/const Eigen::Vector3d &ftrPos,
                                               const float &u_ref, const float &v_ref, const float &focal_length,
                                               const int &level)
{

    const int u_ref_i = floorf(u_ref);
    const int v_ref_i = floorf(v_ref);

    if(IsOutofBorder(ref_img.rows,ref_img.cols,
                     static_cast<int>(u_ref),
                     static_cast<int>(v_ref)))
    {
//            std::cout << "edge is invalid" << std::endl;
        mbRefOutBorder[level] = true;
        return;
    }
    // evaluate projection jacobian
    const int stride = ref_img.cols;
    Matrix<double,2,6> frame_jac;
    jacobian_xyz2uv(ftrPos, frame_jac);

    // compute bilateral interpolation weights for reference image
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    size_t pixel_counter = 0;
    double* cache_ptr = ref_patch_cache_.data();
    const PatternMatrix & patternPoints = mPattern->mPatternPoints;
    for(int i = 0;i < mPattern->mPatternSize;i++,pixel_counter++,cache_ptr++)
    {
        const int ix = patternPoints(i,0),
                iy = patternPoints(i,1);
        uint8_t* ref_img_ptr = (uint8_t*) ref_img.data + (v_ref_i+iy)*stride + (u_ref_i+ix);

        *cache_ptr = w_ref_tl*ref_img_ptr[0] + w_ref_tr*ref_img_ptr[1] + w_ref_bl*ref_img_ptr[stride] + w_ref_br*ref_img_ptr[stride+1];

        // we use the inverse compositional: thereby we can take the gradient always at the same position
        // get gradient of warped image (~gradient at warped position)
        float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1] + w_ref_tr*ref_img_ptr[2] + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                           -(w_ref_tl*ref_img_ptr[-1] + w_ref_tr*ref_img_ptr[0] + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
        float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride] + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                           -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0] + w_ref_br*ref_img_ptr[1]));

        // cache the jacobian
        jacobian_cache_.col(pixel_counter) =
                (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length / (1<<level));
    }
    mbRefOutBorder[level] = false;
    have_ref_patch_cache_ = true;
}


//u_ref,v_ref当前层参考特征点的坐标
//ftrPos在参考左相机相机系下的特征坐标
template <int D>
void EdgeDirectBase<D>::precomputeJacobianRight(const cv::Mat& ref_img, /*const cv::Mat &ftrPos,*/const Eigen::Vector3d &ftrPos,
                                                const float &u_ref, const float &v_ref,
                                                const float &focal_length,const float &baseline,
                                                const int &level)
{

    if(mbRefOutBorder[level])
        return;

    const int u_ref_i = floorf(u_ref);
    const int v_ref_i = floorf(v_ref);
    if(IsOutofBorder(ref_img.rows,ref_img.cols,
                     static_cast<int>(u_ref),
                     static_cast<int>(v_ref)))
    {
//            std::cout << "edge is invalid" << std::endl;
        mbRefOutBorder[level] = true;
        return;
    }
    // evaluate projection jacobian
    const int stride = ref_img.cols;
    Matrix<double,2,6> frame_jac;
    jacobian_xyz2uv(ftrPos, baseline, frame_jac);
    // compute bilateral interpolation weights for reference image
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    size_t pixel_counter = 0;
    double* cache_ptr = ref_patch_cache_right_.data();
//        double* cache_ptr = ref_patch_cache_right_.data();
    const PatternMatrix & patternPoints = mPattern->mPatternPoints;
    for(int i = 0;i<mPattern->mPatternSize;i++,pixel_counter++,cache_ptr++)
    {
        const int ix = patternPoints(i,0),
                iy = patternPoints(i,1);
        uint8_t* ref_img_ptr = (uint8_t*) ref_img.data + (v_ref_i+iy)*stride + (u_ref_i+ix);

        *cache_ptr = w_ref_tl*ref_img_ptr[0] + w_ref_tr*ref_img_ptr[1] + w_ref_bl*ref_img_ptr[stride] + w_ref_br*ref_img_ptr[stride+1];

        // we use the inverse compositional: thereby we can take the gradient always at the same position
        // get gradient of warped image (~gradient at warped position)
        float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1] + w_ref_tr*ref_img_ptr[2] + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                           -(w_ref_tl*ref_img_ptr[-1] + w_ref_tr*ref_img_ptr[0] + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
        float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride] + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                           -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0] + w_ref_br*ref_img_ptr[1]));

        // cache the jacobian
        jacobian_cache_right_.col(pixel_counter) =
                (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length / (1<<level));
    }
    mbRefOutBorder[level] = false;
    have_ref_patch_cache_right_ = true;

}





//判断参考patch是否越界，越界的话误差置为0
//每一步都需要判断预测的点是否越界，越界的话误差置为0
//这里与计算jacobai无关系，jacoian无论都存在能计算的值
template <int D>
void EdgeSE3StereoProjectDirect<D>::computeError()
{

    if(mbRefOutBorder[mLevel])
    {
        _error.setZero();
        return;
    }

    const cv::Mat &curImg = mCurFrame->mImgPyrL[mLevel];
    const cv::Mat &curImgRight = mCurFrame->mImgPyrR[mLevel];

    const InverseDirectVertexSE3Expmap* v0 = static_cast<const InverseDirectVertexSE3Expmap*> ( _vertices[0] );


    Eigen::Vector3d xCurLocal = v0->estimate().map ( mPtLeft );


//        std::cout <<xCurLocal<<std::endl;
    //save the predicted location of tracked feature
    mvPredictUVUr[mLevel][0] = (xCurLocal[0]*mRefFrame->fx/xCurLocal[2] + mRefFrame->cx)*mFactorInv;
    mvPredictUVUr[mLevel][1] = (xCurLocal[1]*mRefFrame->fy/xCurLocal[2] + mRefFrame->cy)*mFactorInv;
    mvPredictUVUr[mLevel][2] = mvPredictUVUr[mLevel][0] - mRefFrame->mbf/xCurLocal[2]*mFactorInv;


    const float u_cur = mvPredictUVUr[mLevel][0];
    const float v_cur = mvPredictUVUr[mLevel][1];
    const float ur_cur = mvPredictUVUr[mLevel][2];


    const int u_cur_i = floorf(u_cur);
    const int v_cur_i = floorf(v_cur);
    const int ur_cur_i = floorf(ur_cur);



//        if ( u_cur_i - mPattern->mBorder<0 || ( u_cur_i+mPattern->mBorder ) >curImg.cols ||
//                ( v_cur_i - mPattern->mBorder ) <0 || ( v_cur_i+mPattern->mBorder ) >curImg.rows
//                ||ur_cur_i - mPattern->mBorder < 0 || ur_cur_i+mPattern->mBorder >= curImg.cols )
    if ( u_cur_i < curLevelOrbEdgeThreshold || u_cur_i >=(curImg.cols - curLevelOrbEdgeThreshold) ||
         v_cur_i < curLevelOrbEdgeThreshold || v_cur_i >=(curImg.rows - curLevelOrbEdgeThreshold)
         ||ur_cur_i < 0 || ur_cur_i >= (curImg.cols - curLevelOrbEdgeThreshold) )
    {
        _error.setZero();
        mIsVisible[mLevel] = false;
    }
    else
    {

        const int patchSize = mPattern->mPatternSize;
        for(int i = 0;i<patchSize;i++)
        {
            const int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            _error[i] = getPixelValue(curImg,u_cur+ix,v_cur+iy) - ref_patch_cache_[i];
            _error[i+patchSize] = getPixelValue(curImgRight,ur_cur+ix,v_cur+iy) - ref_patch_cache_right_[i];

            if(useInformation)
            {
                _information(i,i) = Weight(_error[i]);
                _information(i+patchSize,i+patchSize) = Weight(_error[i+patchSize]);
            }

        }
//            std::cout << _error.transpose() << std::endl;
        mIsVisible[mLevel] = true;
    }

}

template <int D>
void EdgeSE3StereoProjectDirect<D>::linearizeOplus()
{


    //jacobian已经算好了，不用再进行计算
//        if(have_ref_patch_cache_right_ && have_ref_patch_cache_)
//            return;
    if(!have_ref_patch_cache_)
        precomputeJacobianLeft(mRefFrame->mImgPyrL[mLevel],mPtLeft,mMeasUVUr[0]*mFactorInv,
                               mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mLevel);
    if(!have_ref_patch_cache_right_)
        precomputeJacobianRight(mRefFrame->mImgPyrR[mLevel],mPtLeft,mMeasUVUr[2]*mFactorInv,
                                mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mRefFrame->mb,mLevel);
    if(mbRefOutBorder[mLevel] || !mIsVisible[mLevel])
    {
        _jacobianOplusXi.setZero();
    }
    else
    {
        for(int i = 0;i<mPattern->mPatternSize;i++)
        {
            _jacobianOplusXi.row(i) = jacobian_cache_.col(i);
            _jacobianOplusXi.row(i+mPattern->mPatternSize) = jacobian_cache_right_.col(i);
        }
    }
}

template <int D>
bool EdgeSE3StereoProjectDirect<D>::read( std::istream& in )
{
    return true;
}

template <int D>
bool EdgeSE3StereoProjectDirect<D>::write( std::ostream& out ) const
{
    return true;
}


//u,v是k-1特征点在底层图像的坐标，depth为对应深度
//plk为特征对应3d在k时刻左相机系下的坐标
//R为k-1时刻到k时刻的旋转矩阵
//JLeft和JRight分别为左右相机对深度
template <int D>
void EdgeDirectBase<D>::computeJacobianDepth(const float &u,const float &v, const float &depth,
                                                              const float &fu,const float &fv,const float cx,
                                                              const float cy,const float &baseline,
                                                              const Eigen::Vector3d &plk, const Eigen::Matrix3d &R,
                                                              Eigen::Vector2d &JLeft,Eigen::Vector2d &JRight)
{
    float normlisedU = (u-cx)/fu;
    float normlisedV = (v-cy)/fv;
    float a1 = R(0,0)*normlisedU+R(0,1)*normlisedV+R(0,2);
    float a2 = R(1,0)*normlisedU+R(1,1)*normlisedV+R(1,2);
    float a3 = R(2,0)*normlisedU+R(2,1)*normlisedV+R(2,2);
    const float xlk = plk[0];
    const float ylk = plk[1];
    const float zlk = plk[2];
    JLeft[0] = fu*(a1-xlk*a3/zlk)/zlk;
    JLeft[1] = fv*(a2-ylk*a3/zlk)/zlk;
    JRight[0] = fu*(a1-(xlk-baseline)*a3/zlk)/zlk;
    JRight[1] = JLeft[1];
}

template <int D>
void EdgeSE3StereoDepthProjectDirect<D>::DecreaseLevel()
{
    --mLevel;
    mFactorInv = 1.f/(1<<mLevel);
    curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
    assert(mLevel>=0 && "level should be greater than zero");
    have_ref_patch_cache_ = false;
    have_ref_patch_cache_right_ = false;
    precomputeJacobianLeft(mRefFrame->mImgPyrL[mLevel],mPtLeft,mMeasUVUr[0]*mFactorInv,
                           mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mLevel);
    precomputeJacobianRight(mRefFrame->mImgPyrR[mLevel],mPtLeft,mMeasUVUr[2]*mFactorInv,
                            mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mRefFrame->mb,mLevel);
}

template <int D>
void EdgeSE3StereoDepthProjectDirect<D>::computeError()
{

    if(mbRefOutBorder[mLevel])
    {
        _error.setZero();
        return;
    }

    const cv::Mat &curImg = mCurFrame->mImgPyrL[mLevel];
    const cv::Mat &curImgRight = mCurFrame->mImgPyrR[mLevel];
    const DirectVertexDepth* v0 = static_cast<const DirectVertexDepth*> ( _vertices[0] );
    const InverseDirectVertexSE3Expmap* v1 = static_cast<const InverseDirectVertexSE3Expmap*> ( _vertices[1] );

    const double depth = v0->estimate();

    cv::Mat pos = ORB_SLAM2::Frame::UnprojectStereoInCamera(mMeasUVUr[0],mMeasUVUr[1],depth);
    for(int i = 0;i<3;i++)
        mPtLeft[i] = pos.at<float>(i);

    Eigen::Vector3d xCurLocal = v1->estimate().map ( mPtLeft );
//        std::cout <<xCurLocal<<std::endl;
    //save the predicted location of tracked feature
    mvPredictUVUr[mLevel][0] = (xCurLocal[0]*mRefFrame->fx/xCurLocal[2] + mRefFrame->cx)*mFactorInv;
    mvPredictUVUr[mLevel][1] = (xCurLocal[1]*mRefFrame->fy/xCurLocal[2] + mRefFrame->cy)*mFactorInv;
    mvPredictUVUr[mLevel][2] = mvPredictUVUr[mLevel][0] - mRefFrame->mbf/xCurLocal[2]*mFactorInv;


    const float u_cur = mvPredictUVUr[mLevel][0];
    const float v_cur = mvPredictUVUr[mLevel][1];
    const float ur_cur = mvPredictUVUr[mLevel][2];


    const int u_cur_i = floorf(u_cur);
    const int v_cur_i = floorf(v_cur);
    const int ur_cur_i = floorf(ur_cur);


//        if ( u_cur_i - mPattern->mBorder<0 || ( u_cur_i+mPattern->mBorder ) >curImg.cols ||
//             ( v_cur_i - mPattern->mBorder ) <0 || ( v_cur_i+mPattern->mBorder ) >curImg.rows
//             ||ur_cur_i - mPattern->mBorder < 0 || ur_cur_i+mPattern->mBorder >= curImg.cols )
    if ( u_cur_i < curLevelOrbEdgeThreshold || u_cur_i >=(curImg.cols - curLevelOrbEdgeThreshold) ||
         v_cur_i < curLevelOrbEdgeThreshold || v_cur_i >=(curImg.rows - curLevelOrbEdgeThreshold)
         ||ur_cur_i < 0 || ur_cur_i >= (curImg.cols - curLevelOrbEdgeThreshold) )
    {
        _error.setZero();
        mIsVisible[mLevel] = false;
    }
    else
    {

        const int patchSize = mPattern->mPatternSize;
        for(int i = 0;i<patchSize;i++)
        {
            const int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            _error[i] = getPixelValue(curImg,u_cur+ix,v_cur+iy) - ref_patch_cache_[i];
            _error[i+patchSize] = getPixelValue(curImgRight,ur_cur+ix,v_cur+iy) - ref_patch_cache_right_[i];
            if(useInformation)
            {
                _information(i,i) = Weight(_error[i]);
                _information(i+patchSize,i+patchSize) = Weight(_error[i+patchSize]);
            }
        }
//            std::cout << _error.transpose() << std::endl;
        mIsVisible[mLevel] = true;
    }

}

template <int D>
void EdgeSE3StereoDepthProjectDirect<D>::linearizeOplus()
{


    //jacobian已经算好了，不用再进行计算
//        if(have_ref_patch_cache_right_ && have_ref_patch_cache_)
//            return;



    if(!have_ref_patch_cache_)
        precomputeJacobianLeft(mRefFrame->mImgPyrL[mLevel],mPtLeft,mMeasUVUr[0]*mFactorInv,
                               mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mLevel);
    if(!have_ref_patch_cache_right_)
        precomputeJacobianRight(mRefFrame->mImgPyrR[mLevel],mPtLeft,mMeasUVUr[2]*mFactorInv,
                                mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mRefFrame->mb,mLevel);
    if(mbRefOutBorder[mLevel] || !mIsVisible[mLevel])
    {
        _jacobianOplusXi.setZero();
        _jacobianOplusXj.setZero();
    }
    else
    {
        const cv::Mat& curImg = mCurFrame->mImgPyrL[mLevel];
        const cv::Mat& curImgRight = mCurFrame->mImgPyrR[mLevel];
//            const DirectVertexDepth* v0 = static_cast<const DirectVertexDepth*> ( _vertices[0] );
        const InverseDirectVertexSE3Expmap* v1 = static_cast<const InverseDirectVertexSE3Expmap*> ( _vertices[1] );
//            double d = v0->estimate();
        const SE3Quat &Tref = v1->estimate();

        Eigen::Matrix3d R = Tref.rotation().toRotationMatrix();
        Eigen::Vector3d plk = R*mPtLeft + Tref.translation();
        const float u_cur = mvPredictUVUr[mLevel][0];
        const float v_cur = mvPredictUVUr[mLevel][1];
        const float ur_cur = mvPredictUVUr[mLevel][2];
        for(int i = 0;i<mPattern->mPatternSize;i++)
        {
            _jacobianOplusXj.row(i) = jacobian_cache_.col(i);
            _jacobianOplusXj.row(i+mPattern->mPatternSize) = jacobian_cache_right_.col(i);

            const int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            const float u_offset = u_cur+ix,
                    v_offset = v_cur+iy,
                    ur_offset = ur_cur+ix;
//            float IuvL = getPixelValue(curImg,u_offset,v_offset);
//            double dxL = (getPixelValue(curImg,u_offset+1,v_offset) - IuvL);
//            double dyL = (getPixelValue(curImg,u_offset,v_offset+1) - IuvL);
            double dxL = 0.5f*(getPixelValue(curImg,u_offset+1,v_offset) - getPixelValue(curImg,u_offset-1,v_offset));
            double dyL = 0.5f*(getPixelValue(curImg,u_offset,v_offset+1) - getPixelValue(curImg,u_offset,v_offset-1));
            Eigen::Vector2d JdepthLeft,JdepthRight;
            computeJacobianDepth(u_offset, v_offset, mPtLeft[2], mRefFrame->fx*mFactorInv, mRefFrame->fy*mFactorInv,
                                 mRefFrame->cx*mFactorInv,mRefFrame->cy*mFactorInv, mRefFrame->mb,
                                 plk, R,
                                 JdepthLeft, JdepthRight);
            _jacobianOplusXi(i) = dxL*JdepthLeft[0] + dyL*JdepthLeft[1];

//            float IuvR = getPixelValue(curImgRight,ur_offset,v_offset);
//            double dxR = (getPixelValue(curImgRight,ur_offset+1,v_offset) - IuvR);
//            double dyR = (getPixelValue(curImgRight,ur_offset,v_offset+1) - IuvR);
            //TODO：more efficient?
            double dxR = 0.5f*(getPixelValue(curImgRight,ur_offset+1,v_offset) - getPixelValue(curImgRight,ur_offset-1,v_offset));
            double dyR = 0.5f*(getPixelValue(curImgRight,ur_offset,v_offset+1) - getPixelValue(curImgRight,ur_offset,v_offset-1));
            _jacobianOplusXi(i+mPattern->mPatternSize) = dxR*JdepthRight[0] + dyR*JdepthRight[1];
        }
    }
}


template <int D>
void EdgeSE3StereoDirectInverseDepthWithAffine<D>::DecreaseLevel()
{
    --mLevel;
    mFactorInv = 1.f/(1<<mLevel);
    curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
    assert(mLevel>=0 && "level should be greater than zero");
    have_ref_patch_cache_ = false;
    have_ref_patch_cache_right_ = false;
    precomputeJacobianLeft(mRefFrame->mImgPyrL[mLevel],mPtLeft,mMeasUVUr[0]*mFactorInv,
                           mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mLevel);
    precomputeJacobianRight(mRefFrame->mImgPyrR[mLevel],mPtLeft,mMeasUVUr[2]*mFactorInv,
                            mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mRefFrame->mb,mLevel);
}

template <int D>
void EdgeSE3StereoDirectInverseDepthWithAffine<D>::computeError()
{

    if(mbRefOutBorder[mLevel])
    {
        _error.setZero();
        return;
    }

    const cv::Mat &curImg = mCurFrame->mImgPyrL[mLevel];
    const cv::Mat &curImgRight = mCurFrame->mImgPyrR[mLevel];
    const DirectVertexDepth* v0 = static_cast<const DirectVertexDepth*> ( _vertices[0] );
    const InverseDirectVertexSE3Expmap* v1 = static_cast<const InverseDirectVertexSE3Expmap*> ( _vertices[1] );
    const IntensityAffine* v2 = static_cast<const IntensityAffine*> ( _vertices[2] );

    const double depth = v0->estimate();
//    const double depthInv = v0->estimate();
//    const double depth = 1/depthInv;
    const double affine_a = v2->estimate()[0];
    const double affine_b = v2->estimate()[1];

    cv::Mat pos = ORB_SLAM2::Frame::UnprojectStereoInCamera(mMeasUVUr[0],mMeasUVUr[1],depth);
    for(int i = 0;i<3;i++)
        mPtLeft[i] = pos.at<float>(i);

    Eigen::Vector3d xCurLocal = v1->estimate().map ( mPtLeft );
//        std::cout <<xCurLocal<<std::endl;
    //save the predicted location of tracked feature
    mvPredictUVUr[mLevel][0] = (xCurLocal[0]*mRefFrame->fx/xCurLocal[2] + mRefFrame->cx)*mFactorInv;
    mvPredictUVUr[mLevel][1] = (xCurLocal[1]*mRefFrame->fy/xCurLocal[2] + mRefFrame->cy)*mFactorInv;
    mvPredictUVUr[mLevel][2] = mvPredictUVUr[mLevel][0] - mRefFrame->mbf/xCurLocal[2]*mFactorInv;


    const float u_cur = mvPredictUVUr[mLevel][0];
    const float v_cur = mvPredictUVUr[mLevel][1];
    const float ur_cur = mvPredictUVUr[mLevel][2];


    const int u_cur_i = floorf(u_cur);
    const int v_cur_i = floorf(v_cur);
    const int ur_cur_i = floorf(ur_cur);


//        if ( u_cur_i - mPattern->mBorder<0 || ( u_cur_i+mPattern->mBorder ) >curImg.cols ||
//             ( v_cur_i - mPattern->mBorder ) <0 || ( v_cur_i+mPattern->mBorder ) >curImg.rows
//             ||ur_cur_i - mPattern->mBorder < 0 || ur_cur_i+mPattern->mBorder >= curImg.cols )
    if ( u_cur_i < curLevelOrbEdgeThreshold || u_cur_i >=(curImg.cols - curLevelOrbEdgeThreshold) ||
         v_cur_i < curLevelOrbEdgeThreshold || v_cur_i >=(curImg.rows - curLevelOrbEdgeThreshold)
         ||ur_cur_i < 0 || ur_cur_i >= (curImg.cols - curLevelOrbEdgeThreshold) )
    {
        _error.setZero();
        mIsVisible[mLevel] = false;
    }
    else
    {

        const int patchSize = mPattern->mPatternSize;
        for(int i = 0;i<patchSize;i++)
        {
            const int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            _error[i] = affine_a*getPixelValue(curImg,u_cur+ix,v_cur+iy)+affine_b - ref_patch_cache_[i];
            _error[i+patchSize] = affine_a*getPixelValue(curImgRight,ur_cur+ix,v_cur+iy)+affine_b - ref_patch_cache_right_[i];
            if(useInformation)
            {
                _information(i,i) = Weight(_error[i]);
                _information(i+patchSize,i+patchSize) = Weight(_error[i+patchSize]);
            }
        }
//            std::cout << _error.transpose() << std::endl;
        mIsVisible[mLevel] = true;
    }

}

template <int D>
void EdgeSE3StereoDirectInverseDepthWithAffine<D>::linearizeOplus()
{

    if(!have_ref_patch_cache_)
        precomputeJacobianLeft(mRefFrame->mImgPyrL[mLevel],mPtLeft,mMeasUVUr[0]*mFactorInv,
                               mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mLevel);
    if(!have_ref_patch_cache_right_)
        precomputeJacobianRight(mRefFrame->mImgPyrR[mLevel],mPtLeft,mMeasUVUr[2]*mFactorInv,
                                mMeasUVUr[1]*mFactorInv,mRefFrame->fx,mRefFrame->mb,mLevel);
    if(mbRefOutBorder[mLevel] || !mIsVisible[mLevel])
    {
        _jacobianOplus[0].setZero();
        _jacobianOplus[1].setZero();
        _jacobianOplus[2].setZero();
    }
    else
    {
        const cv::Mat& curImg = mCurFrame->mImgPyrL[mLevel];
        const cv::Mat& curImgRight = mCurFrame->mImgPyrR[mLevel];
//            const DirectVertexInverseDepthWithAffine* v0 = static_cast<const DirectVertexInverseDepthWithAffine*> ( _vertices[0] );
        const InverseDirectVertexSE3Expmap* v1 = static_cast<const InverseDirectVertexSE3Expmap*> ( _vertices[1] );
//            double d = v0->estimate();
        const SE3Quat &Tref = v1->estimate();

        Eigen::Matrix3d R = Tref.rotation().toRotationMatrix();
        Eigen::Vector3d plk = R*mPtLeft + Tref.translation();
        const float u_cur = mvPredictUVUr[mLevel][0];
        const float v_cur = mvPredictUVUr[mLevel][1];
        const float ur_cur = mvPredictUVUr[mLevel][2];
        const DirectVertexDepth* v0 = static_cast<const DirectVertexDepth*> ( _vertices[0] );

        const double depth = v0->estimate();
//        const double depthinv = v0->estimate();
//        const double depth = 1/depthinv;
//        double Ddepthinv = -depth*depth;
        for(int i = 0;i<mPattern->mPatternSize;i++)
        {
            _jacobianOplus[1].row(i) = jacobian_cache_.col(i);
            _jacobianOplus[1].row(i+mPattern->mPatternSize) = jacobian_cache_right_.col(i);

            const int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            const float u_offset = u_cur+ix,
                    v_offset = v_cur+iy,
                    ur_offset = ur_cur+ix;
            float IuvL = getPixelValue(curImg,u_offset,v_offset);
//            double dxL = (getPixelValue(curImg,u_offset+1,v_offset) - IuvL);
//            double dyL = (getPixelValue(curImg,u_offset,v_offset+1) - IuvL);
            double dxL = 0.5f*(getPixelValue(curImg,u_offset+1,v_offset) - getPixelValue(curImg,u_offset-1,v_offset));
            double dyL = 0.5f*(getPixelValue(curImg,u_offset,v_offset+1) - getPixelValue(curImg,u_offset,v_offset-1));
            Eigen::Vector2d JdepthLeft,JdepthRight;

            computeJacobianDepth(u_offset, v_offset, depth, mRefFrame->fx*mFactorInv, mRefFrame->fy*mFactorInv,
                                 mRefFrame->cx*mFactorInv,mRefFrame->cy*mFactorInv, mRefFrame->mb,
                                 plk, R,
                                 JdepthLeft, JdepthRight);
            _jacobianOplus[0](i,0) = (dxL*JdepthLeft[0] + dyL*JdepthLeft[1])/**Ddepthinv*/;


            float IuvR = getPixelValue(curImgRight,ur_offset,v_offset);
//            double dxR = (getPixelValue(curImgRight,ur_offset+1,v_offset) - IuvR);
//            double dyR = (getPixelValue(curImgRight,ur_offset,v_offset+1) - IuvR);
            double dxR = 0.5f*(getPixelValue(curImgRight,ur_offset+1,v_offset) - getPixelValue(curImgRight,ur_offset-1,v_offset));
            double dyR = 0.5f*(getPixelValue(curImgRight,ur_offset,v_offset+1) - getPixelValue(curImgRight,ur_offset,v_offset-1));
            _jacobianOplus[0](i+mPattern->mPatternSize,0) = (dxR*JdepthRight[0] + dyR*JdepthRight[1])/**Ddepthinv*/;

            _jacobianOplus[2](i,0) = IuvL;
            _jacobianOplus[2](i,1) = 1;

            _jacobianOplus[2](i+mPattern->mPatternSize,0) = IuvR;
            _jacobianOplus[2](i+mPattern->mPatternSize,1) = 1;
        }
    }
}

template<int D>
void EdgeMultiSE3StereoDirectInvDepth<D>::UpdateRefPatch()
{

    const cv::Mat &refImg = mRefFrame->mImgPyrL[mLevel];
    const cv::Mat &refImgRight = mRefFrame->mImgPyrR[mLevel];
    const float u_ref = mMeasUVUr[0]*mFactorInv;
    const float v_ref = mMeasUVUr[1]*mFactorInv;
    const float ur_ref = mMeasUVUr[2]*mFactorInv;

    if(IsOutofBorder(refImg.rows,refImg.cols,
                     static_cast<int>(u_ref),
                     static_cast<int>(v_ref),
                     static_cast<int>(ur_ref)))
    {
//            std::cout << "edge is invalid" << std::endl;
        mbRefOutBorder[mLevel] = true;
        return;
    }

    for(int i = 0; i< D;i++)
    {
        const int ix = mPattern->mPatternPoints(i,0),
                iy = mPattern->mPatternPoints(i,1);
        const float u_offset = u_ref+ix,
                v_offset = v_ref+iy,
                ur_offset = ur_ref+ix;
        ref_patch_cache_[i] = getPixelValue(refImg,u_offset,v_offset);
        ref_patch_cache_right_[i] = getPixelValue(refImgRight,ur_offset,v_offset);
    }
}

template <int D>
void EdgeMultiSE3StereoDirectInvDepth<D>::DecreaseLevel()
{
    --mLevel;
    mFactorInv = 1.f/(1<<mLevel);
    curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
    assert(mLevel>=0 && "level should be greater than zero");
    UpdateRefPatch();
}

template<int D>
void EdgeMultiSE3StereoDirectInvDepth<D>::computeError()
{
    if(mbRefOutBorder[mLevel])
    {
        _error.setZero();
        return;
    }

    const cv::Mat &curImg = mCurFrame->mImgPyrL[mLevel];
    const cv::Mat &curImgRight = mCurFrame->mImgPyrR[mLevel];

    const VertexSE3ExpmapRight* v0 = static_cast<const VertexSE3ExpmapRight*> ( _vertices[0] );
    const VertexSE3ExpmapRight* v1 = static_cast<const VertexSE3ExpmapRight*> ( _vertices[1] );
    const DirectVertexDepth* v2 = static_cast<const DirectVertexDepth*> ( _vertices[2] );

    const SE3Quat& Ti = v0->estimate();
    const SE3Quat& Tj = v1->estimate();
    const double depth = /*1/*/v2->estimate();
    SE3Quat Tij = Tj*Ti.inverse();
    cv::Mat pos = ORB_SLAM2::Frame::UnprojectStereoInCamera(mMeasUVUr[0],mMeasUVUr[1],depth);
    for(int i = 0;i<3;i++)
        mPtLeft[i] = pos.at<float>(i);

    Eigen::Vector3d xCurLocal = Tij.map ( mPtLeft );
//        std::cout <<xCurLocal<<std::endl;
    //save the predicted location of tracked feature
    mvPredictUVUr[mLevel][0] = (xCurLocal[0]*mRefFrame->fx/xCurLocal[2] + mRefFrame->cx)*mFactorInv;
    mvPredictUVUr[mLevel][1] = (xCurLocal[1]*mRefFrame->fy/xCurLocal[2] + mRefFrame->cy)*mFactorInv;
    mvPredictUVUr[mLevel][2] = mvPredictUVUr[mLevel][0] - mRefFrame->mbf/xCurLocal[2]*mFactorInv;


    const float u_cur = mvPredictUVUr[mLevel][0];
    const float v_cur = mvPredictUVUr[mLevel][1];
    const float ur_cur = mvPredictUVUr[mLevel][2];


    const int u_cur_i = floorf(u_cur);
    const int v_cur_i = floorf(v_cur);
    const int ur_cur_i = floorf(ur_cur);


//    if ( u_cur_i - mPattern->mBorder<0 || ( u_cur_i+mPattern->mBorder ) >curImg.cols ||
//         ( v_cur_i - mPattern->mBorder ) <0 || ( v_cur_i+mPattern->mBorder ) >curImg.rows
//         ||ur_cur_i - mPattern->mBorder < 0 || ur_cur_i+mPattern->mBorder >= curImg.cols )
    if ( u_cur_i < curLevelOrbEdgeThreshold || u_cur_i >=(curImg.cols - curLevelOrbEdgeThreshold) ||
         v_cur_i < curLevelOrbEdgeThreshold || v_cur_i >=(curImg.rows - curLevelOrbEdgeThreshold)
         ||ur_cur_i < 0 || ur_cur_i >= (curImg.cols - curLevelOrbEdgeThreshold) )
    {
        _error.setZero();
        mIsVisible[mLevel] = false;
    }
    else
    {

        const int patchSize = mPattern->mPatternSize;
        for(int i = 0;i<patchSize;i++)
        {
            int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            _error[i] = getPixelValue(curImg,u_cur+ix,v_cur+iy) - ref_patch_cache_[i];
            _error[i+patchSize] = getPixelValue(curImgRight,ur_cur+ix,v_cur+iy) - ref_patch_cache_right_[i];
            if(useInformation)
            {
                _information(i,i) = Weight(_error[i]);
                _information(i+patchSize,i+patchSize) = Weight(_error[i+patchSize]);
            }
        }
//            std::cout << "_error.transpose()" << _error.transpose() << std::endl;
        mIsVisible[mLevel] = true;
    }

}

template<int D>
void EdgeMultiSE3StereoDirectInvDepth<D>::linearizeOplus()
{

    if(mbRefOutBorder[mLevel] || !mIsVisible[mLevel])
    {
        _jacobianOplus[0].setZero();
        _jacobianOplus[1].setZero();
        _jacobianOplus[2].setZero();
    }
    else
    {
        const float fx = mRefFrame->fx*mFactorInv;
        const float fy = mRefFrame->fy*mFactorInv;
        const float cx = mRefFrame->cx*mFactorInv;
        const float cy = mRefFrame->cy*mFactorInv;

        const cv::Mat& curImg = mCurFrame->mImgPyrL[mLevel];
        const cv::Mat& curImgRight = mCurFrame->mImgPyrR[mLevel];
        const VertexSE3ExpmapRight* v0 = static_cast<const VertexSE3ExpmapRight*> ( _vertices[0] );
        const VertexSE3ExpmapRight* v1 = static_cast<const VertexSE3ExpmapRight*> ( _vertices[1] );
        const DirectVertexDepth* v2 = static_cast<const DirectVertexDepth*> ( _vertices[2] );
        const SE3Quat& Ti = v0->estimate();
        const SE3Quat& Tj = v1->estimate();
        const double depth = /*1/*/v2->estimate();

        Eigen::Vector3d ptw = Ti.inverse().map(mPtLeft);


        SE3Quat Tij = Tj*Ti.inverse();

        Eigen::Matrix3d R = Tij.rotation().toRotationMatrix();
        Eigen::Vector3d plk = Tij.map(mPtLeft);

        mvPredictUVUr[mLevel][0] = (plk[0]*mRefFrame->fx/plk[2] + mRefFrame->cx)*mFactorInv;
        mvPredictUVUr[mLevel][1] = (plk[1]*mRefFrame->fy/plk[2] + mRefFrame->cy)*mFactorInv;
        mvPredictUVUr[mLevel][2] = mvPredictUVUr[mLevel][0] - mRefFrame->mbf/plk[2]*mFactorInv;


        Matrix<double,3,6> DR;
        DR.setZero();
        DR.block<3,3>(0,0) = skew(-ptw);
        DR.block<3,3>(0,3).setIdentity();
        DR = Tj.rotation().toRotationMatrix()*DR;

        double x = plk[0];
        double y = plk[1];
        double z = plk[2];

        Matrix<double,2,3> tmp;
        tmp(0,0) = fx/z;
        tmp(0,1) = 0;
        tmp(0,2) = -x/z/z*fx;

        tmp(1,0) = 0;
        tmp(1,1) = fy/z;
        tmp(1,2) = -y/z/z*fy;



        x = plk[0]-mRefFrame->mb;

        Matrix<double,2,3> tmpRight;
        tmpRight = tmp;
        tmpRight(0,2) = -x/z/z*fx;

        Matrix<double,2,6> JacobianL,JacobianR;
        JacobianL = tmp*DR;
        JacobianR = tmpRight*DR;

        const float u_cur = mvPredictUVUr[mLevel][0];
        const float v_cur = mvPredictUVUr[mLevel][1];
        const float ur_cur = mvPredictUVUr[mLevel][2];

//        double Ddepthinv = -depth*depth;
        for(int i = 0;i<mPattern->mPatternSize;i++)
        {
            int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            float u_offset = u_cur+ix,
                    v_offset = v_cur+iy,
                    ur_offset = ur_cur+ix;
//            float IuvL = getPixelValue(curImg,u_offset,v_offset);
//                double dxL = (getPixelValue(curImg,u_offset+1,v_offset) - IuvL);
//                double dyL = (getPixelValue(curImg,u_offset,v_offset+1) - IuvL);
            double dxL = 0.5f*(getPixelValue(curImg,u_offset+1,v_offset) - getPixelValue(curImg,u_offset-1,v_offset));
            double dyL = 0.5f*(getPixelValue(curImg,u_offset,v_offset+1) - getPixelValue(curImg,u_offset,v_offset-1));

            Eigen::Vector2d JdepthLeft,JdepthRight;

            computeJacobianDepth(u_offset, v_offset, depth, fx, fy,
                                 cx,cy, mRefFrame->mb,
                                 plk, R,
                                 JdepthLeft, JdepthRight);
            _jacobianOplus[2](i,0) = (dxL*JdepthLeft[0] + dyL*JdepthLeft[1])/**Ddepthinv*/;

//            float IuvR = getPixelValue(curImgRight,ur_offset,v_offset);
//                double dxR = (getPixelValue(curImgRight,ur_offset+1,v_offset) - IuvR);
//                double dyR = (getPixelValue(curImgRight,ur_offset,v_offset+1) - IuvR);
            double dxR = 0.5f*(getPixelValue(curImgRight,ur_offset+1,v_offset) - getPixelValue(curImgRight,ur_offset-1,v_offset));
            double dyR = 0.5f*(getPixelValue(curImgRight,ur_offset,v_offset+1) - getPixelValue(curImgRight,ur_offset,v_offset-1));

            _jacobianOplus[2](i+mPattern->mPatternSize,0) = (dxR*JdepthRight[0] + dyR*JdepthRight[1])/**Ddepthinv*/;

            _jacobianOplus[1].row(i) = dxL*JacobianL.row(0)+dyL*JacobianL.row(1);
            _jacobianOplus[1].row(i+mPattern->mPatternSize) = dxR*JacobianR.row(0)+dyR*JacobianR.row(1);

            _jacobianOplus[0].row(i) = -_jacobianOplus[1].row(i);
            _jacobianOplus[0].row(i+mPattern->mPatternSize) = -_jacobianOplus[1].row(i+mPattern->mPatternSize);

        }
    }
}

template<int D>
void EdgeMultiSE3StereoDirectDepthWithAffine<D>::UpdateRefPatch()
{

    const cv::Mat &refImg = mRefFrame->mImgPyrL[mLevel];
    const cv::Mat &refImgRight = mRefFrame->mImgPyrR[mLevel];
    const float u_ref = mMeasUVUr[0]*mFactorInv;
    const float v_ref = mMeasUVUr[1]*mFactorInv;
    const float ur_ref = mMeasUVUr[2]*mFactorInv;

    if(IsOutofBorder(refImg.rows,refImg.cols,
                     static_cast<int>(u_ref),
                     static_cast<int>(v_ref),
                     static_cast<int>(ur_ref)))
    {
//            std::cout << "edge is invalid" << std::endl;
        mbRefOutBorder[mLevel] = true;
        return;
    }

    for(int i = 0; i< D;i++)
    {
        const int ix = mPattern->mPatternPoints(i,0),
                iy = mPattern->mPatternPoints(i,1);
        const float u_offset = u_ref+ix,
                v_offset = v_ref+iy,
                ur_offset = ur_ref+ix;
        ref_patch_cache_[i] = getPixelValue(refImg,u_offset,v_offset);
        ref_patch_cache_right_[i] = getPixelValue(refImgRight,ur_offset,v_offset);
    }
}

template <int D>
void EdgeMultiSE3StereoDirectDepthWithAffine<D>::DecreaseLevel()
{
    --mLevel;
    mFactorInv = 1.f/(1<<mLevel);
    curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
    assert(mLevel>=0 && "level should be greater than zero");
    UpdateRefPatch();
}

template<int D>
void EdgeMultiSE3StereoDirectDepthWithAffine<D>::computeError()
{
    if(mbRefOutBorder[mLevel])
    {
        _error.setZero();
        return;
    }

    const cv::Mat &curImg = mCurFrame->mImgPyrL[mLevel];
    const cv::Mat &curImgRight = mCurFrame->mImgPyrR[mLevel];

    const VertexSE3ExpmapRight* v0 = static_cast<const VertexSE3ExpmapRight*> ( _vertices[0] );
    const VertexSE3ExpmapRight* v1 = static_cast<const VertexSE3ExpmapRight*> ( _vertices[1] );
    const DirectVertexDepth* v2 = static_cast<const DirectVertexDepth*> ( _vertices[2] );
    const IntensityAffine* v3 = static_cast<const IntensityAffine*> ( _vertices[3] );

    const float &affine_a = v3->estimate()[0];
    const float &affine_b = v3->estimate()[1];

    const SE3Quat& Ti = v0->estimate();
    const SE3Quat& Tj = v1->estimate();
    const double depth = /*1/*/v2->estimate();
    SE3Quat Tij = Tj*Ti.inverse();
    cv::Mat pos = ORB_SLAM2::Frame::UnprojectStereoInCamera(mMeasUVUr[0],mMeasUVUr[1],depth);
    for(int i = 0;i<3;i++)
        mPtLeft[i] = pos.at<float>(i);

    Eigen::Vector3d xCurLocal = Tij.map ( mPtLeft );
//        std::cout <<xCurLocal<<std::endl;
    //save the predicted location of tracked feature
    mvPredictUVUr[mLevel][0] = (xCurLocal[0]*mRefFrame->fx/xCurLocal[2] + mRefFrame->cx)*mFactorInv;
    mvPredictUVUr[mLevel][1] = (xCurLocal[1]*mRefFrame->fy/xCurLocal[2] + mRefFrame->cy)*mFactorInv;
    mvPredictUVUr[mLevel][2] = mvPredictUVUr[mLevel][0] - mRefFrame->mbf/xCurLocal[2]*mFactorInv;


    const float u_cur = mvPredictUVUr[mLevel][0];
    const float v_cur = mvPredictUVUr[mLevel][1];
    const float ur_cur = mvPredictUVUr[mLevel][2];


    const int u_cur_i = floorf(u_cur);
    const int v_cur_i = floorf(v_cur);
    const int ur_cur_i = floorf(ur_cur);


//        if ( u_cur_i - mPattern->mBorder<0 || ( u_cur_i+mPattern->mBorder ) >curImg.cols ||
//             ( v_cur_i - mPattern->mBorder ) <0 || ( v_cur_i+mPattern->mBorder ) >curImg.rows
//             ||ur_cur_i - mPattern->mBorder < 0 || ur_cur_i+mPattern->mBorder >= curImg.cols )
    if ( u_cur_i < curLevelOrbEdgeThreshold || u_cur_i >=(curImg.cols - curLevelOrbEdgeThreshold) ||
         v_cur_i < curLevelOrbEdgeThreshold || v_cur_i >=(curImg.rows - curLevelOrbEdgeThreshold)
         ||ur_cur_i < 0 || ur_cur_i >= (curImg.cols - curLevelOrbEdgeThreshold) )
    {
        _error.setZero();
        mIsVisible[mLevel] = false;
    }
    else
    {

        const int patchSize = mPattern->mPatternSize;
        for(int i = 0;i<patchSize;i++)
        {
            int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            _error[i] = affine_a*getPixelValue(curImg,u_cur+ix,v_cur+iy)+affine_b - ref_patch_cache_[i];
            _error[i+patchSize] = affine_a*getPixelValue(curImgRight,ur_cur+ix,v_cur+iy)+affine_b - ref_patch_cache_right_[i];
            if(useInformation)
            {
                _information(i,i) = Weight(_error[i]);
                _information(i+patchSize,i+patchSize) = Weight(_error[i+patchSize]);
            }
        }
//            std::cout << "_error.transpose()" << _error.transpose() << std::endl;
        mIsVisible[mLevel] = true;
    }

}

template<int D>
void EdgeMultiSE3StereoDirectDepthWithAffine<D>::linearizeOplus()
{

    if(mbRefOutBorder[mLevel] || !mIsVisible[mLevel])
    {
        _jacobianOplus[0].setZero();
        _jacobianOplus[1].setZero();
        _jacobianOplus[2].setZero();
    }
    else
    {
        const float fx = mRefFrame->fx*mFactorInv;
        const float fy = mRefFrame->fy*mFactorInv;
        const float cx = mRefFrame->cx*mFactorInv;
        const float cy = mRefFrame->cy*mFactorInv;

        const cv::Mat& curImg = mCurFrame->mImgPyrL[mLevel];
        const cv::Mat& curImgRight = mCurFrame->mImgPyrR[mLevel];
        const VertexSE3ExpmapRight* v0 = static_cast<const VertexSE3ExpmapRight*> ( _vertices[0] );
        const VertexSE3ExpmapRight* v1 = static_cast<const VertexSE3ExpmapRight*> ( _vertices[1] );
        const DirectVertexDepth* v2 = static_cast<const DirectVertexDepth*> ( _vertices[2] );
        const SE3Quat& Ti = v0->estimate();
        const SE3Quat& Tj = v1->estimate();
        const double depth = /*1/*/v2->estimate();

        Eigen::Vector3d ptw = Ti.inverse().map(mPtLeft);


        SE3Quat Tij = Tj*Ti.inverse();

        Eigen::Matrix3d R = Tij.rotation().toRotationMatrix();
        Eigen::Vector3d plk = Tij.map(mPtLeft);

        mvPredictUVUr[mLevel][0] = (plk[0]*mRefFrame->fx/plk[2] + mRefFrame->cx)*mFactorInv;
        mvPredictUVUr[mLevel][1] = (plk[1]*mRefFrame->fy/plk[2] + mRefFrame->cy)*mFactorInv;
        mvPredictUVUr[mLevel][2] = mvPredictUVUr[mLevel][0] - mRefFrame->mbf/plk[2]*mFactorInv;


        Matrix<double,3,6> DR;
        DR.setZero();
        DR.block<3,3>(0,0) = skew(-ptw);
        DR.block<3,3>(0,3).setIdentity();
        DR = Tj.rotation().toRotationMatrix()*DR;

        double x = plk[0];
        double y = plk[1];
        double z = plk[2];

        Matrix<double,2,3> tmp;
        tmp(0,0) = fx/z;
        tmp(0,1) = 0;
        tmp(0,2) = -x/z/z*fx;

        tmp(1,0) = 0;
        tmp(1,1) = fy/z;
        tmp(1,2) = -y/z/z*fy;



        x = plk[0]-mRefFrame->mb;

        Matrix<double,2,3> tmpRight;
        tmpRight = tmp;
        tmpRight(0,2) = -x/z/z*fx;

        Matrix<double,2,6> JacobianL,JacobianR;
        JacobianL = tmp*DR;
        JacobianR = tmpRight*DR;

        const float u_cur = mvPredictUVUr[mLevel][0];
        const float v_cur = mvPredictUVUr[mLevel][1];
        const float ur_cur = mvPredictUVUr[mLevel][2];

//            double Ddepthinv = -depth*depth;
        for(int i = 0;i<mPattern->mPatternSize;i++)
        {
            int ix = mPattern->mPatternPoints(i,0),
                    iy = mPattern->mPatternPoints(i,1);
            float u_offset = u_cur+ix,
                    v_offset = v_cur+iy,
                    ur_offset = ur_cur+ix;
            float IuvL = getPixelValue(curImg,u_offset,v_offset);
//                double dxL = (getPixelValue(curImg,u_offset+1,v_offset) - IuvL);
//                double dyL = (getPixelValue(curImg,u_offset,v_offset+1) - IuvL);
            double dxL = 0.5f*(getPixelValue(curImg,u_offset+1,v_offset) - getPixelValue(curImg,u_offset-1,v_offset));
            double dyL = 0.5f*(getPixelValue(curImg,u_offset,v_offset+1) - getPixelValue(curImg,u_offset,v_offset-1));

            Eigen::Vector2d JdepthLeft,JdepthRight;

            computeJacobianDepth(u_offset, v_offset, depth, fx, fy,
                                 cx,cy, mRefFrame->mb,
                                 plk, R,
                                 JdepthLeft, JdepthRight);
            _jacobianOplus[2](i,0) = (dxL*JdepthLeft[0] + dyL*JdepthLeft[1])/**Ddepthinv*/;

            float IuvR = getPixelValue(curImgRight,ur_offset,v_offset);
//                double dxR = (getPixelValue(curImgRight,ur_offset+1,v_offset) - IuvR);
//                double dyR = (getPixelValue(curImgRight,ur_offset,v_offset+1) - IuvR);
            double dxR = 0.5f*(getPixelValue(curImgRight,ur_offset+1,v_offset) - getPixelValue(curImgRight,ur_offset-1,v_offset));
            double dyR = 0.5f*(getPixelValue(curImgRight,ur_offset,v_offset+1) - getPixelValue(curImgRight,ur_offset,v_offset-1));

            _jacobianOplus[2](i+mPattern->mPatternSize,0) = (dxR*JdepthRight[0] + dyR*JdepthRight[1])/**Ddepthinv*/;

            _jacobianOplus[1].row(i) = dxL*JacobianL.row(0)+dyL*JacobianL.row(1);
            _jacobianOplus[1].row(i+mPattern->mPatternSize) = dxR*JacobianR.row(0)+dyR*JacobianR.row(1);

            _jacobianOplus[0].row(i) = -_jacobianOplus[1].row(i);
            _jacobianOplus[0].row(i+mPattern->mPatternSize) = -_jacobianOplus[1].row(i+mPattern->mPatternSize);

            _jacobianOplus[3](i,0) = IuvL;
            _jacobianOplus[3](i,1) = 1;

            _jacobianOplus[3](i+mPattern->mPatternSize,0) = IuvR;
            _jacobianOplus[3](i+mPattern->mPatternSize,1) = 1;
        }
    }
}