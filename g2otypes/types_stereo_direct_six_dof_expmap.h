//
// Created by zsk on 16-7-26.
//

#ifndef ORB_SLAM2_TYPES_STEREO_DIRECT_SIX_DOF_EXPMAP_H
#define ORB_SLAM2_TYPES_STEREO_DIRECT_SIX_DOF_EXPMAP_H


#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/se3_ops.h"
#include "Thirdparty/g2o/g2o/types/se3quat.h"
#include "Thirdparty/g2o/g2o/core/factory.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include <opencv2/core/core.hpp>
#include "Frame.h"


namespace g2o{
    using namespace Eigen;

    typedef Matrix<double, 6, 6> Matrix6d;
/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */
    class  VertexSE3ExpmapRight : public BaseVertex<6, SE3Quat>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexSE3ExpmapRight()
        {}

        bool read(std::istream& is)
        {
            return true;
        }

        bool write(std::ostream& os) const
        {
            return true;
        }

        virtual void setToOriginImpl() {
            _estimate = SE3Quat();
        }

        virtual void oplusImpl(const double* update_)  {
            Eigen::Map<const Vector6d> update(update_);
            setEstimate(estimate()*SE3Quat::exp(update));
        }
    };


    class  InverseDirectVertexSE3Expmap : public BaseVertex<6, SE3Quat>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        InverseDirectVertexSE3Expmap()
        {}

        bool read(std::istream& is)
        {
            return true;
        }

        bool write(std::ostream& os) const
        {
            return  true;
        }

        virtual void setToOriginImpl() {
            _estimate = SE3Quat();
        }

        virtual void oplusImpl(const double* update_)  {
            Eigen::Map<const Vector6d> update(update_);
            setEstimate(estimate()*SE3Quat::exp(-update));
        }
    };

    class  DirectVertexDepth : public BaseVertex<1, double>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DirectVertexDepth()
        {}

        bool read(std::istream& is)
        {
            return true;
        }

        bool write(std::ostream& os) const
        {
            return true;
        }

        virtual void setToOriginImpl() {
            _estimate = 0;
        }

        virtual void oplusImpl(const double* update_)  {
            setEstimate(estimate() + update_[0]);
        }
    };

    class  IntensityAffine : public BaseVertex<2, Eigen::Vector2d>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        IntensityAffine()
        {}

        bool read(std::istream& is)
        {
            return true;
        }

        bool write(std::ostream& os) const
        {
            return true;
        }

        virtual void setToOriginImpl() {

        }

        virtual void oplusImpl(const double* update_)  {
            Eigen::Map<const Vector2d> v(update_);
            _estimate += v;
        }
    };





    template <int D>
    struct Pattern{
        typedef Eigen::Matrix<int,D,2> PatternMatrix;
        PatternMatrix mPatternPoints;
        int mBorder,mPatternSize;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    template <int D>
    class EdgeDirectBase{

    public:
        typedef typename Pattern<D>::PatternMatrix PatternMatrix;
        typedef Eigen::Matrix<double,D,1> PatchVector;
        typedef Eigen::Matrix<double, 6, D, Eigen::ColMajor> JocabianCacheType;
        EdgeDirectBase():have_ref_patch_cache_(false),
                         have_ref_patch_cache_right_(false)
        {}

        Pattern<D> *mPattern;

        //stolen from svo
        JocabianCacheType jacobian_cache_, jacobian_cache_right_;
        bool have_ref_patch_cache_,have_ref_patch_cache_right_;
        PatchVector ref_patch_cache_, ref_patch_cache_right_;
        std::vector<bool> mbRefOutBorder;


        void precomputeJacobianLeft(const cv::Mat& ref_img, /*const cv::Mat &ftrPos,*/const Eigen::Vector3d &ftrPos,
                                    const float &u_ref, const float &v_ref, const float &focal_length,
                                    const int &level);

        void precomputeJacobianRight(const cv::Mat& ref_img, /*const cv::Mat &ftrPos,*/const Eigen::Vector3d &ftrPos,
                                     const float &u_ref, const float &v_ref,
                                     const float &focal_length,const float &baseline,
                                     const int &level);


        bool IsOutofBorder(int rows,int cols,int u,int v, int ur);

        bool IsOutofBorder(int rows,int cols,int u,int v);

        void computeJacobianDepth(const float &u,const float &v, const float &depth,
                                  const float &fu,const float &fv,const float cx,
                                  const float cy,const float &baseline,
                                  const Eigen::Vector3d &plk, const Eigen::Matrix3d &R,
                                  Eigen::Vector2d &JLeft,Eigen::Vector2d &JRight);

    public:
        inline static void jacobian_xyz2uv(/*const cv::Mat &ftrPos,*/
                                            const Eigen::Vector3d &ftrPos,
                                            Eigen::Matrix<double,2,6>& J)
        {
//            const double x = ftrPos.at<float>(0);
//            const double y = ftrPos.at<float>(1);
//            const double z_inv = 1./ftrPos.at<float>(2);
            const double x = ftrPos(0);
            const double y = ftrPos(1);
            const double z_inv = 1./ftrPos(2);
            const double z_inv_2 = z_inv*z_inv;

            J(0,5) = x*z_inv_2;           // x/z^2
            J(0,0) = y*J(0,5);            // x*y/z^2
            J(0,1) = -(1.0 + x*J(0,5));   // -(1.0 + x^2/z^2)
            J(0,2) = y*z_inv;             // y/z
            J(0,3) = -z_inv;              // -1/z
            J(0,4) = 0.0;                 // 0


            J(1,5) = y*z_inv_2;           // y/z^2
            J(1,0) = 1.0 + y*J(1,5);      // 1.0 + y^2/z^2
            J(1,1) = -J(0,0);             // -x*y/z^2
            J(1,2) = -x*z_inv;            // x/z
            J(1,3) = 0.0;                 // 0
            J(1,4) = -z_inv;              // -1/z
        }

        inline static void jacobian_xyz2uv(/*const cv::Mat &ftrPos,*/
                                           const Eigen::Vector3d &ftrPos,float baseline,
                                           Eigen::Matrix<double,2,6>& J)
        {
//            const double x = ftrPos.at<float>(0);
//            const double y = ftrPos.at<float>(1);
//            const double z_inv = 1./ftrPos.at<float>(2);
            const double x = ftrPos(0);
            const double y = ftrPos(1);
            const double z_inv = 1./ftrPos(2);
            const double z_inv_2 = z_inv*z_inv;
            const double xr = x - baseline;

            J(0,5) = xr*z_inv_2;           // x/z^2
            J(0,0) = y*J(0,5);            // x*y/z^2
            J(0,1) = -(1.0 + x*J(0,5));   // -(1.0 + x^2/z^2)
            J(0,2) = y*z_inv;             // y/z
            J(0,3) = -z_inv;              // -1/z
            J(0,4) = 0.0;                 // 0


            J(1,5) = y*z_inv_2;           // y/z^2
            J(1,0) = 1.0 + y*J(1,5);      // 1.0 + y^2/z^2
            J(1,1) = -x*J(1,5);             // -x*y/z^2
            J(1,2) = -x*z_inv;            // x/z
            J(1,3) = 0.0;                 // 0
            J(1,4) = -z_inv;              // -1/z
        }

        inline static float getPixelValue( const cv::Mat &img, float x, float y )
        {
            assert(img.type() == CV_8UC1);
            uchar* data = & img.data[ int(y) * img.step + int(x) ];
            float xx = x - floor ( x );
            float yy = y - floor ( y );
            float v = (
                    ( 1-xx ) * ( 1-yy ) * data[0] +
                    xx* ( 1-yy ) * data[1] +
                    ( 1-xx ) *yy*data[ img.step ] +
                    xx*yy*data[img.step+1]
            );
            return v;
        }

    protected:

        const int orbEdgeThreshold = 19;
        int curLevelOrbEdgeThreshold;
        bool useInformation{false};
        const float weightcoeff = 5.f;

        void SetUseInformation(bool useinfor)
        {
            useInformation = useinfor;
        }

        float Weight(const float &err)
        {
            return fabsf(err) < weightcoeff ? 1 : weightcoeff / fabsf(err);
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };


    //提前计算jacboain和参考patch的像素，如果越界，置越界标示
    //并且全部元素设为0
    //
    template <int D>
    class EdgeSE3StereoProjectDirect: public BaseUnaryEdge< D*2, Eigen::Matrix<double,D*2,1>, InverseDirectVertexSE3Expmap >,
                                      public EdgeDirectBase<D>
    {

    public:
        typedef Eigen::Matrix<double,D*2,1> ErrorVector;

        typedef Eigen::Matrix<double,D,2> PatternMatrix;

        EdgeSE3StereoProjectDirect()
        {}

        EdgeSE3StereoProjectDirect(ORB_SLAM2::Frame* refFrame, ORB_SLAM2::Frame* curFrame,
        const Eigen::Vector3d& measUVUr, const Eigen::Vector3d &ptLeft,Pattern<D> *pattern):
                mRefFrame(refFrame),mCurFrame(curFrame),mPtLeft(ptLeft),mMeasUVUr(measUVUr),
                mLevel(refFrame->mnPyrLevel-1),mFactorInv(1.f/(1<<mLevel))
        {
            curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
            mvPredictUVUr.resize(refFrame->mnPyrLevel);
            mIsVisible.resize(refFrame->mnPyrLevel,true);
            mbRefOutBorder.resize(refFrame->mnPyrLevel,false);
            assert(pattern->mPatternSize == pattern->mPatternPoints.rows() && "pattern rows is not compatible to error dimension");
            mPattern = pattern;
            precomputeJacobianLeft(refFrame->mImgPyrL[mLevel],mPtLeft,mMeasUVUr[0]*mFactorInv,
                                   mMeasUVUr[1]*mFactorInv,refFrame->fx,mLevel);
            precomputeJacobianRight(refFrame->mImgPyrR[mLevel],mPtLeft,mMeasUVUr[2]*mFactorInv,
                                    mMeasUVUr[1]*mFactorInv,refFrame->fx,refFrame->mb,mLevel);

//            assert(pattern.rows() == D && "pattern rows is not compatible to error dimension");
        }

        void DecreaseLevel()
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
        ORB_SLAM2::Frame *mRefFrame,*mCurFrame;

        Eigen::Vector3d mPtLeft;

        Eigen::Vector3d mMeasUVUr;

        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > mvPredictUVUr;

        int mLevel;
        float mFactorInv;



        std::vector<bool> mIsVisible;
        

    protected:


        typedef BaseUnaryEdge< D*2, Eigen::Matrix<double,D*2,1>, InverseDirectVertexSE3Expmap> BaseType;
        //继承的baseedge的成员函数

        virtual void computeError();

        // plus in manifold
        virtual void linearizeOplus( );

        // dummy read and write functions because we don't care...
        virtual bool read( std::istream& in );


        virtual bool write( std::ostream& out ) const;

        using BaseType::_measurement;
        using BaseType::_information;
        using BaseType::_error;
        using BaseType::_vertices;
        using BaseType::_dimension;
        using BaseType::_jacobianOplusXi;
    public:
        using EdgeDirectBase<D>::mPattern;
        using EdgeDirectBase<D>::jacobian_xyz2uv;
        using EdgeDirectBase<D>::jacobian_cache_;
        using EdgeDirectBase<D>::jacobian_cache_right_;
        using EdgeDirectBase<D>::ref_patch_cache_;
        using EdgeDirectBase<D>::ref_patch_cache_right_;
        using EdgeDirectBase<D>::have_ref_patch_cache_;
        using EdgeDirectBase<D>::have_ref_patch_cache_right_;
        using EdgeDirectBase<D>::getPixelValue;
        using EdgeDirectBase<D>::precomputeJacobianLeft;
        using EdgeDirectBase<D>::precomputeJacobianRight;
        using EdgeDirectBase<D>::mbRefOutBorder;
        using EdgeDirectBase<D>::IsOutofBorder;
        using EdgeDirectBase<D>::orbEdgeThreshold;
        using EdgeDirectBase<D>::curLevelOrbEdgeThreshold;
        using EdgeDirectBase<D>::SetUseInformation;
        using EdgeDirectBase<D>::useInformation;
        using EdgeDirectBase<D>::Weight;
        using EdgeDirectBase<D>::weightcoeff;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };

//    template <int D>
//    class EdgeSE3StereoDepthProjectDirect: public BaseBinaryEdge< D*2, Eigen::Matrix<double,D*2,1>,
//            DirectVertexDepth, VertexSE3Expmap >, public EdgeDirectBase<D>
//    {
//
//    protected:
//
//        typedef BaseBinaryEdge< D*2, Eigen::Matrix<double,D*2,1>,
//                DirectVertexDepth, InverseDirectVertexSE3Expmap > BaseType;
//        //继承的baseedge的成员函数
//
//        virtual void computeError();
//
//        // plus in manifold
//        virtual void linearizeOplus( );
//
//        // dummy read and write functions because we don't care...
//        virtual bool read( std::istream& in )
//        {
//            return true;
//        }
//
//        virtual bool write( std::ostream& out ) const
//        {
//            return true;
//        }
//
//    public:
//        using EdgeDirectBase<D>::mPattern;
//        using EdgeDirectBase<D>::jacobian_xyz2uv;
//        using EdgeDirectBase<D>::jacobian_cache_;
//        using EdgeDirectBase<D>::jacobian_cache_right_;
//        using EdgeDirectBase<D>::ref_patch_cache_;
//        using EdgeDirectBase<D>::ref_patch_cache_right_;
//        using EdgeDirectBase<D>::have_ref_patch_cache_;
//        using EdgeDirectBase<D>::have_ref_patch_cache_right_;
//        using EdgeDirectBase<D>::getPixelValue;
//        using EdgeDirectBase<D>::precomputeJacobianLeft;
//        using EdgeDirectBase<D>::precomputeJacobianRight;
//        using EdgeDirectBase<D>::mbRefOutBorder;
//        using EdgeDirectBase<D>::IsOutofBorder;
//        using EdgeDirectBase<D>::orbEdgeThreshold;
//        using EdgeDirectBase<D>::curLevelOrbEdgeThreshold;
//    public:
//        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//    };

    template <int D>
    class EdgeSE3StereoDepthProjectDirect: public BaseBinaryEdge< D*2, Eigen::Matrix<double,D*2,1>,
            DirectVertexDepth, InverseDirectVertexSE3Expmap >,
                                      public EdgeDirectBase<D>
    {

    public:
        EdgeSE3StereoDepthProjectDirect()
        {}

        EdgeSE3StereoDepthProjectDirect(ORB_SLAM2::Frame* refFrame, ORB_SLAM2::Frame* curFrame,
        const Eigen::Vector3d& measUVUr, const Eigen::Vector3d &ptLeft,Pattern<D> *pattern):
        mRefFrame(refFrame),mCurFrame(curFrame),mPtLeft(ptLeft),mMeasUVUr(measUVUr),
                mLevel(refFrame->mnPyrLevel-1),mFactorInv(1.f/(1<<mLevel))
        {
            curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
            mvPredictUVUr.resize(refFrame->mnPyrLevel);
            mIsVisible.resize(refFrame->mnPyrLevel,true);
            mbRefOutBorder.resize(refFrame->mnPyrLevel,false);
            assert(pattern->mPatternSize == pattern->mPatternPoints.rows() && "pattern rows is not compatible to error dimension");
            mPattern = pattern;
            precomputeJacobianLeft(refFrame->mImgPyrL[mLevel],mPtLeft,mMeasUVUr[0]*mFactorInv,
                                   mMeasUVUr[1]*mFactorInv,refFrame->fx,mLevel);
            precomputeJacobianRight(refFrame->mImgPyrR[mLevel],mPtLeft,mMeasUVUr[2]*mFactorInv,
                                    mMeasUVUr[1]*mFactorInv,refFrame->fx,refFrame->mb,mLevel);

//            assert(pattern.rows() == D && "pattern rows is not compatible to error dimension");
        }

        void DecreaseLevel();


        ORB_SLAM2::Frame *mRefFrame,*mCurFrame;

        Eigen::Vector3d mPtLeft;

        Eigen::Vector3d mMeasUVUr;

        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > mvPredictUVUr;


        int mLevel;
        float mFactorInv;

        std::vector<bool> mIsVisible;

        Eigen::Matrix<double, D, 1> mJacboianDepthCache, mJacboianDepthCacheRight;


    protected:

        typedef BaseBinaryEdge< D*2, Eigen::Matrix<double,D*2,1>,
                DirectVertexDepth, InverseDirectVertexSE3Expmap > BaseType;
        //继承的baseedge的成员函数

        virtual void computeError();

        // plus in manifold
        virtual void linearizeOplus( );

        // dummy read and write functions because we don't care...
        virtual bool read( std::istream& in )
        {
            return true;
        }

        virtual bool write( std::ostream& out ) const
        {
            return true;
        }

        using BaseType::_measurement;
        using BaseType::_information;
        using BaseType::_error;
        using BaseType::_vertices;
        using BaseType::_dimension;
        using BaseType::_jacobianOplusXi;
        using BaseType::_jacobianOplusXj;

    public:
        using EdgeDirectBase<D>::mPattern;
        using EdgeDirectBase<D>::jacobian_xyz2uv;
        using EdgeDirectBase<D>::jacobian_cache_;
        using EdgeDirectBase<D>::jacobian_cache_right_;
        using EdgeDirectBase<D>::ref_patch_cache_;
        using EdgeDirectBase<D>::ref_patch_cache_right_;
        using EdgeDirectBase<D>::have_ref_patch_cache_;
        using EdgeDirectBase<D>::have_ref_patch_cache_right_;
        using EdgeDirectBase<D>::getPixelValue;
        using EdgeDirectBase<D>::precomputeJacobianLeft;
        using EdgeDirectBase<D>::precomputeJacobianRight;
        using EdgeDirectBase<D>::mbRefOutBorder;
        using EdgeDirectBase<D>::IsOutofBorder;
        using EdgeDirectBase<D>::orbEdgeThreshold;
        using EdgeDirectBase<D>::curLevelOrbEdgeThreshold;
        using EdgeDirectBase<D>::computeJacobianDepth;
        using EdgeDirectBase<D>::SetUseInformation;
        using EdgeDirectBase<D>::useInformation;
        using EdgeDirectBase<D>::Weight;
        using EdgeDirectBase<D>::weightcoeff;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };



    template <int D>
    class EdgeSE3StereoDirectInverseDepthWithAffine: public BaseMultiEdge< D*2, Eigen::Matrix<double,D*2,1> >,
                                                     public EdgeDirectBase<D>
    {
    public:
        EdgeSE3StereoDirectInverseDepthWithAffine()
        {}

        EdgeSE3StereoDirectInverseDepthWithAffine(ORB_SLAM2::Frame* refFrame, ORB_SLAM2::Frame* curFrame,
        const Eigen::Vector3d& measUVUr, const Eigen::Vector3d &ptLeft,Pattern<D> *pattern):
        mRefFrame(refFrame),mCurFrame(curFrame),mPtLeft(ptLeft),mMeasUVUr(measUVUr),
                mLevel(refFrame->mnPyrLevel-1),mFactorInv(1.f/(1<<mLevel))
        {
            curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
            mvPredictUVUr.resize(refFrame->mnPyrLevel);
            mIsVisible.resize(refFrame->mnPyrLevel,true);
            mbRefOutBorder.resize(refFrame->mnPyrLevel,false);
            assert(pattern->mPatternSize == pattern->mPatternPoints.rows() && "pattern rows is not compatible to error dimension");
            mPattern = pattern;
            precomputeJacobianLeft(refFrame->mImgPyrL[mLevel],mPtLeft,mMeasUVUr[0]*mFactorInv,
                                   mMeasUVUr[1]*mFactorInv,refFrame->fx,mLevel);
            precomputeJacobianRight(refFrame->mImgPyrR[mLevel],mPtLeft,mMeasUVUr[2]*mFactorInv,
                                    mMeasUVUr[1]*mFactorInv,refFrame->fx,refFrame->mb,mLevel);


            _vertices.resize(3);
            {
                MatrixXd::MapType a(0,0,0);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType b(0,0,0);
                _jacobianOplus.push_back(b);
            }

            {
                MatrixXd::MapType a(0,0,0);
                _jacobianOplus.push_back(a);
            }
            _hessian.resize(6);
//            assert(pattern.rows() == D && "pattern rows is not compatible to error dimension");
        }

        void DecreaseLevel();


        ORB_SLAM2::Frame *mRefFrame,*mCurFrame;

        Eigen::Vector3d mPtLeft;

        Eigen::Vector3d mMeasUVUr;

        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > mvPredictUVUr;

        int mLevel;
        float mFactorInv;

        std::vector<bool> mIsVisible;

        Eigen::Matrix<double, D, 1> mJacboianDepthCache, mJacboianDepthCacheRight;
    protected:

        typedef BaseMultiEdge< D*2, Eigen::Matrix<double,D*2,1> > BaseType;
        //继承的baseedge的成员函数

        virtual void computeError();

        // plus in manifold
        virtual void linearizeOplus( );

        // dummy read and write functions because we don't care...
        virtual bool read( std::istream& in )
        {
            return true;
        }

        virtual bool write( std::ostream& out ) const
        {
            return true;
        }

        using BaseType::_measurement;
        using BaseType::_information;
        using BaseType::_error;
        using BaseType::_vertices;
        using BaseType::_dimension;
        using BaseType::_jacobianOplus;
        using BaseType::_hessian;
        using BaseType::JacobianType;

    public:
        using EdgeDirectBase<D>::mPattern;
        using EdgeDirectBase<D>::jacobian_xyz2uv;
        using EdgeDirectBase<D>::jacobian_cache_;
        using EdgeDirectBase<D>::jacobian_cache_right_;
        using EdgeDirectBase<D>::ref_patch_cache_;
        using EdgeDirectBase<D>::ref_patch_cache_right_;
        using EdgeDirectBase<D>::have_ref_patch_cache_;
        using EdgeDirectBase<D>::have_ref_patch_cache_right_;
        using EdgeDirectBase<D>::getPixelValue;
        using EdgeDirectBase<D>::precomputeJacobianLeft;
        using EdgeDirectBase<D>::precomputeJacobianRight;
        using EdgeDirectBase<D>::mbRefOutBorder;
        using EdgeDirectBase<D>::IsOutofBorder;
        using EdgeDirectBase<D>::orbEdgeThreshold;
        using EdgeDirectBase<D>::curLevelOrbEdgeThreshold;
        using EdgeDirectBase<D>::computeJacobianDepth;
        using EdgeDirectBase<D>::SetUseInformation;
        using EdgeDirectBase<D>::useInformation;
        using EdgeDirectBase<D>::Weight;
        using EdgeDirectBase<D>::weightcoeff;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    template <int D>
    class EdgeMultiSE3StereoDirectInvDepth: public BaseMultiEdge< D*2, Eigen::Matrix<double,D*2,1> >,
                                            public EdgeDirectBase<D>
    {
    public:
        EdgeMultiSE3StereoDirectInvDepth()
        {
            _vertices.resize(3);
            {
                MatrixXd::MapType a(0,2*D,1);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,6);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,2);
                _jacobianOplus.push_back(a);
            }
            _hessian.resize(6);
        }

        EdgeMultiSE3StereoDirectInvDepth(ORB_SLAM2::Frame* refFrame, ORB_SLAM2::Frame* curFrame,
                                         const Eigen::Vector3d &measUVUr,const Eigen::Vector3d &ptLeft,
                                         int level, Pattern<D> *pattern):
                mRefFrame(refFrame),mCurFrame(curFrame),mPtLeft(ptLeft),mMeasUVUr(measUVUr),
                mLevel(level),mFactorInv(1.f/(1<<mLevel))
        {

            curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
            mvPredictUVUr.resize(refFrame->mnPyrLevel);
            mIsVisible.resize(refFrame->mnPyrLevel,true);
            mbRefOutBorder.resize(refFrame->mnPyrLevel,false);
            assert(pattern->mPatternSize == pattern->mPatternPoints.rows() && "pattern rows is not compatible to error dimension");
            mPattern = pattern;
            UpdateRefPatch();
            _vertices.resize(3);
            {
                MatrixXd::MapType a(0,2*D,1);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,6);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,2);
                _jacobianOplus.push_back(a);
            }
            _hessian.resize(6);
        }

        void UpdateRefPatch();

        void DecreaseLevel();

        ORB_SLAM2::Frame* mRefFrame,*mCurFrame;
        Eigen::Vector3d mPtLeft,mMeasUVUr;

        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > mvPredictUVUr;
        int mLevel;
        float mFactorInv;
        std::vector<bool> mIsVisible;
    protected:

        typedef BaseMultiEdge< D*2, Eigen::Matrix<double,D*2,1> > BaseType;
        //继承的baseedge的成员函数

        virtual void computeError();

        // plus in manifold
        virtual void linearizeOplus();

        // dummy read and write functions because we don't care...
        virtual bool read( std::istream& in )
        {
            return true;
        }

        virtual bool write( std::ostream& out ) const
        {
            return true;
        }

        using BaseType::_measurement;
        using BaseType::_information;
        using BaseType::_error;
        using BaseType::_vertices;
        using BaseType::_dimension;
        using BaseType::_jacobianOplus;
        using BaseType::_hessian;

    public:
        using EdgeDirectBase<D>::ref_patch_cache_;
        using EdgeDirectBase<D>::ref_patch_cache_right_;
        using EdgeDirectBase<D>::getPixelValue;
        using EdgeDirectBase<D>::mPattern;
        using EdgeDirectBase<D>::jacobian_xyz2uv;
        using EdgeDirectBase<D>::IsOutofBorder;
        using EdgeDirectBase<D>::orbEdgeThreshold;
        using EdgeDirectBase<D>::curLevelOrbEdgeThreshold;
        using EdgeDirectBase<D>::computeJacobianDepth;
        using EdgeDirectBase<D>::mbRefOutBorder;
        using EdgeDirectBase<D>::SetUseInformation;
        using EdgeDirectBase<D>::useInformation;
        using EdgeDirectBase<D>::Weight;
        using EdgeDirectBase<D>::weightcoeff;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };


    template <int D>
    class EdgeMultiSE3StereoDirectDepthWithAffine: public BaseMultiEdge< D*2, Eigen::Matrix<double,D*2,1> >,
                                            public EdgeDirectBase<D>
    {
    public:
        EdgeMultiSE3StereoDirectDepthWithAffine()
        {
            _vertices.resize(3);
            {
                MatrixXd::MapType a(0,2*D,1);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,6);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,2);
                _jacobianOplus.push_back(a);
            }
            _hessian.resize(6);
        }

        EdgeMultiSE3StereoDirectDepthWithAffine(ORB_SLAM2::Frame* refFrame, ORB_SLAM2::Frame* curFrame,
                                         const Eigen::Vector3d &measUVUr,const Eigen::Vector3d &ptLeft,
                                         int level, Pattern<D> *pattern):
                mRefFrame(refFrame),mCurFrame(curFrame),mPtLeft(ptLeft),mMeasUVUr(measUVUr),
                mLevel(level),mFactorInv(1.f/(1<<mLevel))
        {

            curLevelOrbEdgeThreshold = static_cast<int>(orbEdgeThreshold*mFactorInv);
            mvPredictUVUr.resize(refFrame->mnPyrLevel);
            mIsVisible.resize(refFrame->mnPyrLevel,true);
            mbRefOutBorder.resize(refFrame->mnPyrLevel,false);
            assert(pattern->mPatternSize == pattern->mPatternPoints.rows() && "pattern rows is not compatible to error dimension");
            mPattern = pattern;
            UpdateRefPatch();

            _vertices.resize(4);
            {
                MatrixXd::MapType a(0,2*D,1);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,6);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,2);
                _jacobianOplus.push_back(a);
            }

            {
                MatrixXd::MapType a(0,2*D,2);
                _jacobianOplus.push_back(a);
            }
            _hessian.resize(12);
        }

        void UpdateRefPatch();

        void DecreaseLevel();

        ORB_SLAM2::Frame* mRefFrame,*mCurFrame;
        Eigen::Vector3d mPtLeft,mMeasUVUr;

        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > mvPredictUVUr;
        int mLevel;
        float mFactorInv;
        std::vector<bool> mIsVisible;
    protected:

        typedef BaseMultiEdge< D*2, Eigen::Matrix<double,D*2,1> > BaseType;
        //继承的baseedge的成员函数

        virtual void computeError();

        // plus in manifold
        virtual void linearizeOplus();

        // dummy read and write functions because we don't care...
        virtual bool read( std::istream& in )
        {
            return true;
        }

        virtual bool write( std::ostream& out ) const
        {
            return true;
        }

        using BaseType::_measurement;
        using BaseType::_information;
        using BaseType::_error;
        using BaseType::_vertices;
        using BaseType::_dimension;
        using BaseType::_jacobianOplus;
        using BaseType::_hessian;

    public:
        using EdgeDirectBase<D>::ref_patch_cache_;
        using EdgeDirectBase<D>::ref_patch_cache_right_;
        using EdgeDirectBase<D>::getPixelValue;
        using EdgeDirectBase<D>::mPattern;
        using EdgeDirectBase<D>::jacobian_xyz2uv;
        using EdgeDirectBase<D>::IsOutofBorder;
        using EdgeDirectBase<D>::orbEdgeThreshold;
        using EdgeDirectBase<D>::curLevelOrbEdgeThreshold;
        using EdgeDirectBase<D>::computeJacobianDepth;
        using EdgeDirectBase<D>::mbRefOutBorder;
        using EdgeDirectBase<D>::SetUseInformation;
        using EdgeDirectBase<D>::useInformation;
        using EdgeDirectBase<D>::Weight;
        using EdgeDirectBase<D>::weightcoeff;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

#include "types_stereo_direct_six_dof_expmap.hpp"
}




#endif //ORB_SLAM2_TYPES_STEREO_DIRECT_SIX_DOF_EXPMAP_H
