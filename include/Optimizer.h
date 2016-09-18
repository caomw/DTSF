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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H


#include "Frame.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "MathUtilis.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "types_stereo_direct_six_dof_expmap.h"
//#define PRINT_DEBUG

namespace ORB_SLAM2
{

class Optimizer
{
public:



    template <int D>
    bool static poseEstimationDirect ( Frame * refFrame, Frame *curFrame, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
//        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;//第二个参数对一元边无影响
//        DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();//就是6*6矩阵的求解器
//        DirectBlock* solver_ptr = new DirectBlock( linearSolver );
//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        static int counters{0};
        const int counterTh = 90;
//        const float chi2Th = 10000;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
//        g2o::BlockSolver_6_3::PoseMatrixType
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( false );       // 打开调试输出

        g2o::InverseDirectVertexSE3Expmap* pose = new g2o::InverseDirectVertexSE3Expmap();
        pose->setEstimate( Converter::toSE3Quat(curFrame->mTcw*refFrame->GetInversePose()) );
        pose->setId(0);
        optimizer.addVertex( pose );

        // 添加边
        int id=1;
        std::vector<EdgeInfo> vEdgeInfo;
        for(int i = 0;i<refFrame->N;i++)
        {

            if(refFrame->mvDepth[i]>0)
            {

                const cv::KeyPoint &kpt = refFrame->mvKeys[i];
                cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(i);
                Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                g2o::EdgeSE3StereoProjectDirect<D>* edge = new g2o::EdgeSE3StereoProjectDirect<D>(
                        refFrame,curFrame,
                        Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[i]),
                        feature,pattern);
                edge->setVertex( 0, pose );
                edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                edge->setId( id++ );
                edge->SetUseInformation(true);
//                if(bRobust)
//                {
//                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                    rk->setDelta(100);
//                    edge->setRobustKernel(rk);
//                }
                optimizer.addEdge(edge);
                vEdgeInfo.push_back(EdgeInfo(edge,i,EdgeInfo::EdgeTpye::FixedDepth));
            }

        }
        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        int nValidEdges=0, nInvalidEdges=0;
        for(int idxLevel = refFrame->mnPyrLevel-2;idxLevel >= 0;idxLevel--)
        {
            id = 0;
            nValidEdges = 0,nInvalidEdges = 0;
            cv::Mat Twc = Converter::toCvMat(pose->estimate());

            for(auto edgeInfo:vEdgeInfo)
            {

                if(edgeInfo.eType == EdgeInfo::EdgeTpye::FixedDepth)
                {
                    g2o::EdgeSE3StereoProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoProjectDirect<D>*>(edgeInfo.edge);
                    const int &level = e->mLevel;
#ifdef PRINT_DEBUG
                    if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                    {

//                        const int &idxL = edgeInfo.kpIdx;
                        nValidEdges++;
                    }
                    else
                    {
                        nInvalidEdges++;
                    }
#endif

                    e->DecreaseLevel();
                }
            }
#ifdef PRINT_DEBUG
            cout << "# of valid edges:" << nValidEdges <<
                 "\t# of invalid edges:" << nInvalidEdges << endl;
#endif

            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }


        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
        id = 0;
        nValidEdges = 0,nInvalidEdges = 0;
        cv::Mat Twc = Converter::toCvMat(pose->estimate());
        for(auto edgeInfo:vEdgeInfo)
        {
            edgeInfo.edge->computeError();
//            if(edgeInfo.edge->chi2()>chi2Th)
//            {
//                continue;
//            }


            if(edgeInfo.eType == EdgeInfo::EdgeTpye::FixedDepth)
            {
                g2o::EdgeSE3StereoProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoProjectDirect<D>*>(edgeInfo.edge);
                const int &level = e->mLevel;

                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                {

                    const int &idxL = edgeInfo.kpIdx;
                    if(refFrame->mvLives[idxL]>3)
                    {
                        ++refFrame->mvLives[idxL];
                        continue;
                    }

                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(idxL);

                    float d = Twc.at<float>(2,0)*ftrPos.at<float>(0)+Twc.at<float>(2,1)*ftrPos.at<float>(1)+
                              Twc.at<float>(2,2)*ftrPos.at<float>(2)+Twc.at<float>(2,3);
                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL]);

                    //add associations of map point for cur frame
                    MapPoint* &mp = refFrame->mvpMapPoints[idxL];
                    mp->AddObservation(curFrame,nValidEdges);
                    curFrame->mvpMapPoints.push_back(mp);

//#ifdef PRINT_DEBUG
//                    if(counters == counterTh)
//                    {
                        //                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
//                                                                   refFrame->cy,refFrame->mb,
//                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
//                                                                   e->mvPredictUVUr[level][0]*(1<<level),
//                                                                   e->mvPredictUVUr[level][1]*(1<<level),
//                                                                   e->mvPredictUVUr[level][2]*(1<<level));
//                    cout << "before: " << vDepthBefore[id]
//                         << "\tafter:" << v->estimate()
//                            << "\tquad triangulation:" << x3D.at<float>(2)
//                            << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
//                        cout << "ch2: " << e->chi2() << endl;
//                        MathUtilis::DrawQuasiFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                                      refFrame->mImgPyrR[level],curFrame->mImgPyrR[level],
//                                                      e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
//                                                      e->mMeasUVUr[2]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][2],e->mvPredictUVUr[level][1]);
//                    }


//#endif
                    nValidEdges++;
                }
                else
                {
                    nInvalidEdges++;
                }

                ++id;
            }
        }
        ++counters;
        curFrame->SetPose(Twc*refFrame->mTcw);

        return false;
    }

    template <int D>
    int static PoseEstimationMultiFrames( std::list<Frame*> &frames, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
        static int counters{0};
        const int counterTh = 90;
//        const float chi2Th = 10000;
//        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> DirectBlock;//第二个参数对一元边无影响
        g2o::BlockSolver<g2o::BlockSolverX>::LinearSolverType * linearSolver = new g2o::LinearSolverDense< g2o::BlockSolver<g2o::BlockSolverX>::PoseMatrixType > ();//就是6*6矩B阵的求解器
        g2o::BlockSolver<g2o::BlockSolverX>* solver_ptr = new g2o::BlockSolver<g2o::BlockSolverX>( linearSolver );

//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( true );       // 打开调试输出

        std::map<Frame*,g2o::VertexSE3ExpmapRight*> framesVertex;

        int id = 0;
        for(auto f:frames)
        {
            g2o::VertexSE3ExpmapRight* pose = new g2o::VertexSE3ExpmapRight();

            pose->setEstimate( Converter::toSE3Quat(f->mTcw) );
            if(id == 0)
                pose->setFixed(true);
            pose->setId(id++);

            optimizer.addVertex( pose );
            framesVertex.insert(std::make_pair(f,pose));
        }


        // 添加边
        std::vector<double> vDepthBefore;
        std::vector<EdgeInfo> vEdgeInfo;
        int edgeId = 0;
        for(auto refFrame:frames)
        {
            for(int j = 0;j<refFrame->N;j++)
            {

                if(refFrame->mvDepth[j]>0)
                {
                    MapPoint* mp = refFrame->mvpMapPoints[j];
                    if(mp->mnExtractFrameId != refFrame->mnId)
                    {
                        continue;
                    }
                    g2o::DirectVertexDepth* vDepth = new g2o::DirectVertexDepth();
                    vDepth->setId(id++);
                    Eigen::Vector3d depthEstimate;
                    vDepth->setEstimate(/*1/*/refFrame->mvDepth[j]);
                    vDepth->setMarginalized(true);
                    optimizer.addVertex(vDepth);
                    vDepthBefore.push_back(refFrame->mvDepth[j]);

                    const cv::KeyPoint &kpt = refFrame->mvKeys[j];
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(j);
                    Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                    auto v0 = framesVertex[refFrame];

                    for(auto ob:mp->mObservations)
                    {
                        Frame* curFrame = ob.first;
                        if(refFrame == curFrame)
                            continue;

                        g2o::EdgeMultiSE3StereoDirectInvDepth<D>* edge = new g2o::EdgeMultiSE3StereoDirectInvDepth<D>(
                                refFrame,curFrame,
                                Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[j]),
                                feature, 0, pattern);

                        edge->setVertex( 0, v0 );

                        auto findvIter = framesVertex.find(curFrame);
                        if(findvIter == framesVertex.end())
                            continue;
                        auto v1 = findvIter->second;
                        edge->setVertex( 1, v1 );
                        edge->setVertex(2, vDepth);
                        edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                        edge->setId( edgeId++ );
//                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                        rk->setDelta(35);
//                        edge->setRobustKernel(rk);
                        optimizer.addEdge(edge);
                        vEdgeInfo.push_back(EdgeInfo(edge, j, EdgeInfo::EdgeTpye::OptimizedDepth));
                    }

                }

            }
        }

        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(20);

        for(auto framev:framesVertex)
        {
            Frame *f = framev.first;
            auto v = framev.second;
            cout << "pose optimze before:" << endl;
            cout << /*f->mTcw */ f->GetInversePose() <<endl;
            cv::Mat Tcw = Converter::toCvMat(v->estimate());
            cout << "pose optimze after:" << endl;
            f->SetPose(Tcw);
            cout << /*Tcw*/f->GetInversePose() << endl;

        }

        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
        id = 0;
        int nValidEdges = 0,nInvalidEdges = 0;
        for(auto edgeInfo:vEdgeInfo)
        {
//            edgeInfo.edge->computeError();
//
//            if(edgeInfo.edge->chi2()>chi2Th)
//            {
//                continue;
//            }

            if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
            {
                g2o::EdgeMultiSE3StereoDirectInvDepth<D>* e =
                        static_cast<g2o::EdgeMultiSE3StereoDirectInvDepth<D>*>(edgeInfo.edge);

                const g2o::VertexSE3ExpmapRight* v0 =
                        static_cast<const g2o::VertexSE3ExpmapRight*>(edgeInfo.edge->vertex(0));
                const g2o::VertexSE3ExpmapRight* v1 =
                        static_cast<const g2o::VertexSE3ExpmapRight*>(edgeInfo.edge->vertex(1));
                const g2o::DirectVertexDepth* v2 =
                        static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(2));

                const int &level = e->mLevel;
                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                {
                    Frame *refFrame = e->mRefFrame;
                    Frame *curFrame = e->mCurFrame;
                    const int &idxL = edgeInfo.kpIdx;



//#ifdef PRINT_DEBUG
//                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
//                                                                   refFrame->cy,refFrame->mb,
//                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
//                                                                   e->mvPredictUVUr[level][0]*(1<<level),
//                                                                   e->mvPredictUVUr[level][1]*(1<<level),
//                                                                   e->mvPredictUVUr[level][2]*(1<<level));
//                    cout << "before: " << refFrame->mvDepth[idxL]
//                         << "\tafter:" << v->estimate()
//                         << "\tquad triangulation:" << x3D.at<float>(2)
//                         << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
//#endif

//                        cout << "(v->estimate()-refFrame->mvDepth[" << idxL << "]):" << v->estimate()-refFrame->mvDepth[idxL] << endl;


                    const cv::KeyPoint &kpt = refFrame->mvKeys[idxL];
                    const double updateDepth = 1/v2->estimate();
//                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(kpt.pt.x,kpt.pt.y,updateDepth);
                    refFrame->mvDepth[idxL] = updateDepth;
//                    cout << "ftrPos[" << idxL << "]):" << ftrPos.t() << endl;
//                        float * Tr3 = Twc.ptr<float>(2);
//                        float *pFtrPos = (float*)(ftrPos.data);
//                        float d = Tr3[0]*pFtrPos[0]+Tr3[1]*pFtrPos[1]+Tr3[2]*pFtrPos[2]+Tr3[3];
//                    cv::Mat Twc = g2o::toCvMat(v1->estimate()*v0->estimate().inverse());
//                    float d = Twc.at<float>(2,0)*ftrPos.at<float>(0)+Twc.at<float>(2,1)*ftrPos.at<float>(1)+
//                              Twc.at<float>(2,2)*ftrPos.at<float>(2)+Twc.at<float>(2,3);

//                        float d = Tr3[0]*ftrPos.at<float>(0)+Tr3[1]*ftrPos.at<float>(0)+Tr3[2]*ftrPos.at<float>(0)+pFtrPos[3];
//                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
//                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL],
//                                            idxL,refFrame->mvbIsFixed[idxL]);

//                    nValidEdges++;

//                    if(counters == counterTh)
//                    {
//                        cout << "ch2: " << e->chi2() << endl;
//                        MathUtilis::DrawQuasiFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                                      refFrame->mImgPyrR[level],curFrame->mImgPyrR[level],
//                                                      e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
//                                                      e->mMeasUVUr[2]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][2],e->mvPredictUVUr[level][1]);
//                    }


                }
                else
                {
                    nInvalidEdges++;
                }

            }
        }
        ++counters;

        return nValidEdges;


    }


    template <int D>
    int static OptimizeKeyFrameDepth( std::list<Frame*> &frames, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
        static int counters{0};
//        const int counterTh = 90;
//        const float chi2Th = 10000;
//        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> DirectBlock;//第二个参数对一元边无影响
        g2o::BlockSolver<g2o::BlockSolverX>::LinearSolverType * linearSolver = new g2o::LinearSolverDense< g2o::BlockSolver<g2o::BlockSolverX>::PoseMatrixType > ();//就是6*6矩B阵的求解器
        g2o::BlockSolver<g2o::BlockSolverX>* solver_ptr = new g2o::BlockSolver<g2o::BlockSolverX>( linearSolver );

//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( false );       // 打开调试输出

        std::map<Frame*,g2o::VertexSE3ExpmapRight*> framesVertex;

        int id = 0;

        for(auto f:frames)
        {
            g2o::VertexSE3ExpmapRight* pose = new g2o::VertexSE3ExpmapRight();

            pose->setEstimate( Converter::toSE3Quat(f->mTcw) );
            if(id == 0)
                pose->setFixed(true);
            pose->setId(id++);

            optimizer.addVertex( pose );
            framesVertex.insert(std::make_pair(f,pose));
        }


        // 添加边
        std::vector<double> vDepthBefore;
        std::vector<EdgeInfo> vEdgeInfo;
        int edgeId = 0;


        auto frameIterStart = frames.begin();
        Frame *refFrame = *frameIterStart;
        frameIterStart++;

        for(int j = 0;j<refFrame->N;j++)
        {

            if(refFrame->mvDepth[j]>0)
            {
                g2o::DirectVertexDepth* vDepth = new g2o::DirectVertexDepth();
                vDepth->setId(id++);
                Eigen::Vector3d depthEstimate;
                vDepth->setEstimate(/*1/*/refFrame->mvDepth[j]);
                vDepth->setMarginalized(true);
                optimizer.addVertex(vDepth);
                vDepthBefore.push_back(refFrame->mvDepth[j]);

                const cv::KeyPoint &kpt = refFrame->mvKeys[j];
                cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(j);
                Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                auto v0 = framesVertex[refFrame];

                for(auto iter = frameIterStart,iterEnd = frames.end();iter != iterEnd;iter++)
                {
                    Frame* curFrame = *iter;
                    if(refFrame == curFrame)
                        continue;

                    g2o::EdgeMultiSE3StereoDirectInvDepth<D>* edge = new g2o::EdgeMultiSE3StereoDirectInvDepth<D>(
                            refFrame,curFrame,
                            Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[j]),
                            feature, refFrame->mnPyrLevel-1, pattern);

                    edge->setVertex( 0, v0 );

                    auto findvIter = framesVertex.find(curFrame);
                    if(findvIter == framesVertex.end())
                        continue;
                    auto v1 = findvIter->second;
                    edge->setVertex( 1, v1 );
                    edge->setVertex(2, vDepth);
                    edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                    edge->setId(edgeId++);
                    edge->SetUseInformation(true);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(10);
                    edge->setRobustKernel(rk);
                    optimizer.addEdge(edge);
                    vEdgeInfo.push_back(EdgeInfo(edge, j, EdgeInfo::EdgeTpye::OptimizedDepth));
                }

            }

        }


        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        for(int idxLevel = frames.front()->mnPyrLevel-2;idxLevel >=0;idxLevel--)
        {
            id = 0;


            for(auto edgeInfo:vEdgeInfo)
            {

                if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
                {
                    g2o::EdgeMultiSE3StereoDirectInvDepth<D>* e = static_cast<g2o::EdgeMultiSE3StereoDirectInvDepth<D>*>(edgeInfo.edge);
                    e->DecreaseLevel();
                }
            }
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        for(auto framev:framesVertex)
        {
            Frame *f = framev.first;
            auto v = framev.second;
//            cout << "pose optimze before:" << endl;
//            cout << /*f->mTcw */ f->GetInversePose() <<endl;
            cv::Mat Tcw = Converter::toCvMat(v->estimate());
//            cout << "pose optimze after:" << Tcw <<  endl;
            f->SetPose(Tcw);
//            cout << "pose optimze was set:" << endl;
//            cout << /*Tcw*/f->GetInversePose() << endl;
        }

        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
        id = 0;
//        int nValidEdges = 0,nInvalidEdges = 0;
//        for(auto edgeInfo:vEdgeInfo)
//        {
////            edgeInfo.edge->computeError();
////
////            if(edgeInfo.edge->chi2()>chi2Th)
////            {
////                continue;
////            }
//
//            if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
//            {
//                g2o::EdgeMultiSE3StereoDirectInvDepth<D>* e =
//                        static_cast<g2o::EdgeMultiSE3StereoDirectInvDepth<D>*>(edgeInfo.edge);
//
//                const g2o::VertexSE3ExpmapRight* v0 =
//                        static_cast<const g2o::VertexSE3ExpmapRight*>(edgeInfo.edge->vertex(0));
//                const g2o::VertexSE3ExpmapRight* v1 =
//                        static_cast<const g2o::VertexSE3ExpmapRight*>(edgeInfo.edge->vertex(1));
//                const g2o::DirectVertexDepth* v2 =
//                        static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(2));
//
//                const int &level = e->mLevel;
//                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
//                {
//                    Frame *refFrame = e->mRefFrame;
//                    Frame *curFrame = e->mCurFrame;
//                    const int &idxL = edgeInfo.kpIdx;
//
//
//
////#ifdef PRINT_DEBUG
////                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
////                                                                   refFrame->cy,refFrame->mb,
////                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
////                                                                   e->mvPredictUVUr[level][0]*(1<<level),
////                                                                   e->mvPredictUVUr[level][1]*(1<<level),
////                                                                   e->mvPredictUVUr[level][2]*(1<<level));
////                    cout << "before: " << refFrame->mvDepth[idxL]
////                         << "\tafter:" << v->estimate()
////                         << "\tquad triangulation:" << x3D.at<float>(2)
////                         << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
////                         << "\tvalid edge" << endl;
////#endif
//
////                        cout << "(v->estimate()-refFrame->mvDepth[" << idxL << "]):" << v->estimate()-refFrame->mvDepth[idxL] << endl;
//
//
//                    const cv::KeyPoint &kpt = refFrame->mvKeys[idxL];
//                    const double updateDepth = 1/v2->estimate();
////                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(kpt.pt.x,kpt.pt.y,updateDepth);
//                    refFrame->mvDepth[idxL] = updateDepth;
////                    cout << "ftrPos[" << idxL << "]):" << ftrPos.t() << endl;
////                        float * Tr3 = Twc.ptr<float>(2);
////                        float *pFtrPos = (float*)(ftrPos.data);
////                        float d = Tr3[0]*pFtrPos[0]+Tr3[1]*pFtrPos[1]+Tr3[2]*pFtrPos[2]+Tr3[3];
////                    cv::Mat Twc = g2o::toCvMat(v1->estimate()*v0->estimate().inverse());
////                    float d = Twc.at<float>(2,0)*ftrPos.at<float>(0)+Twc.at<float>(2,1)*ftrPos.at<float>(1)+
////                              Twc.at<float>(2,2)*ftrPos.at<float>(2)+Twc.at<float>(2,3);
//
////                        float d = Tr3[0]*ftrPos.at<float>(0)+Tr3[1]*ftrPos.at<float>(0)+Tr3[2]*ftrPos.at<float>(0)+pFtrPos[3];
////                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
////                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL],
////                                            idxL,refFrame->mvbIsFixed[idxL]);
//
////                    nValidEdges++;
//
////                    if(counters == counterTh)
////                    {
////                        cout << "ch2: " << e->chi2() << endl;
////                        MathUtilis::DrawQuasiFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
////                                                      refFrame->mImgPyrR[level],curFrame->mImgPyrR[level],
////                                                      e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
////                                                      e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
////                                                      e->mMeasUVUr[2]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
////                                                      e->mvPredictUVUr[level][2],e->mvPredictUVUr[level][1]);
////                    }
//
//
//                }
//                else
//                {
//                    nInvalidEdges++;
//                }
//
//            }
//        }
        ++counters;

        return 0;
//        return nValidEdges;
    }

    template <int D>
    int static OptimizeKeyFrameDepthWithAffine( std::list<Frame*> &frames, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
        static int counters{0};
//        const int counterTh = 90;
//        const float chi2Th = 10000;
//        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> DirectBlock;//第二个参数对一元边无影响
        g2o::BlockSolver<g2o::BlockSolverX>::LinearSolverType * linearSolver = new g2o::LinearSolverDense< g2o::BlockSolver<g2o::BlockSolverX>::PoseMatrixType > ();//就是6*6矩B阵的求解器
        g2o::BlockSolver<g2o::BlockSolverX>* solver_ptr = new g2o::BlockSolver<g2o::BlockSolverX>( linearSolver );

//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( false );       // 打开调试输出

        std::map<Frame*,g2o::VertexSE3ExpmapRight*> framesVertex;

        int id = 0;

        for(auto f:frames)
        {
            g2o::VertexSE3ExpmapRight* pose = new g2o::VertexSE3ExpmapRight();

            {
                std::unique_lock<std::mutex> lock1(f->mMutexTcw);
                pose->setEstimate( Converter::toSE3Quat(f->mTcw) );
            }

            if(id == 0)
                pose->setFixed(true);
            pose->setId(id++);

            optimizer.addVertex( pose );
            framesVertex.insert(std::make_pair(f,pose));
        }

        std::vector<g2o::IntensityAffine*> affineVertex;
        for(int i = 0,imax = i<frames.size()-1;i<imax;i++)
        {
            g2o::IntensityAffine* vAffine = new g2o::IntensityAffine();
            vAffine->setEstimate(Eigen::Vector2d(1,0));
            vAffine->setId(id++);
            vAffine->setFixed(false);
//        vAffine->setMarginalized(true);
            optimizer.addVertex( vAffine );
            affineVertex.push_back(vAffine);
        }


        // 添加边
        std::vector<double> vDepthBefore;
        std::vector<EdgeInfo> vEdgeInfo;
        int edgeId = 0;
        auto frameIterStart = frames.begin();
        Frame *refFrame = *frameIterStart;
        frameIterStart++;
        for(int j = 0;j<refFrame->N;j++)
        {

            if(refFrame->mvDepth[j]>0)
            {
                g2o::DirectVertexDepth* vDepth = new g2o::DirectVertexDepth();
                vDepth->setId(id++);
                Eigen::Vector3d depthEstimate;
                vDepth->setEstimate(/*1/*/refFrame->mvDepth[j]);
                vDepth->setMarginalized(true);
                optimizer.addVertex(vDepth);
                vDepthBefore.push_back(refFrame->mvDepth[j]);

                const cv::KeyPoint &kpt = refFrame->mvKeys[j];
                cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(j);
                Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                auto v0 = framesVertex[refFrame];

                int affineId = 0;
                for(auto iter = frameIterStart,iterEnd = frames.end();iter != iterEnd;iter++)
                {
                    Frame* curFrame = *iter;
                    if(refFrame == curFrame)
                        continue;

                    g2o::EdgeMultiSE3StereoDirectDepthWithAffine<D>* edge = new g2o::EdgeMultiSE3StereoDirectDepthWithAffine<D>(
                            refFrame,curFrame,
                            Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[j]),
                            feature, 0, pattern);

                    edge->setVertex( 0, v0 );

                    auto findvIter = framesVertex.find(curFrame);
                    if(findvIter == framesVertex.end())
                        continue;
                    auto v1 = findvIter->second;
                    edge->setVertex( 1, v1 );
                    edge->setVertex(2, vDepth);
                    edge->setVertex(3, affineVertex[affineId]);
                    edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                    edge->setId(edgeId++);
                    edge->SetUseInformation(true);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(5);
                    edge->setRobustKernel(rk);
                    optimizer.addEdge(edge);
                    vEdgeInfo.push_back(EdgeInfo(edge, j, EdgeInfo::EdgeTpye::OptimizedDepth));
                }
            }

        }


        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(20);

        for(auto framev:framesVertex)
        {
            Frame *f = framev.first;
            auto v = framev.second;
//            cout << "pose optimze before:" << endl;
//            cout << /*f->mTcw */ f->GetInversePose() <<endl;
            cv::Mat Tcw = Converter::toCvMat(v->estimate());
//            cout << "pose optimze after:" << endl;
            f->SetPose(Tcw);
//            cout << /*Tcw*/f->GetInversePose() << endl;

        }

        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
//        id = 0;
//        int nValidEdges = 0,nInvalidEdges = 0;
//        for(auto edgeInfo:vEdgeInfo)
//        {
////            edgeInfo.edge->computeError();
////
////            if(edgeInfo.edge->chi2()>chi2Th)
////            {
////                continue;
////            }
//
//            if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
//            {
//                g2o::EdgeMultiSE3StereoDirectDepthWithAffine<D>* e =
//                        static_cast<g2o::EdgeMultiSE3StereoDirectDepthWithAffine<D>*>(edgeInfo.edge);
//
//                const g2o::VertexSE3ExpmapRight* v0 =
//                        static_cast<const g2o::VertexSE3ExpmapRight*>(edgeInfo.edge->vertex(0));
//                const g2o::VertexSE3ExpmapRight* v1 =
//                        static_cast<const g2o::VertexSE3ExpmapRight*>(edgeInfo.edge->vertex(1));
//                const g2o::DirectVertexDepth* v2 =
//                        static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(2));
//
//                const int &level = e->mLevel;
//                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
//                {
//                    Frame *refFrame = e->mRefFrame;
//                    Frame *curFrame = e->mCurFrame;
//                    const int &idxL = edgeInfo.kpIdx;
//
//
//
//
//
////                        cout << "(v->estimate()-refFrame->mvDepth[" << idxL << "]):" << v->estimate()-refFrame->mvDepth[idxL] << endl;
//
//
//                    const cv::KeyPoint &kpt = refFrame->mvKeys[idxL];
//                    const double updateDepth = /*1/*/v2->estimate();
////                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(kpt.pt.x,kpt.pt.y,updateDepth);
//                    refFrame->mvDepth[idxL] = updateDepth;
////                    cout << "ftrPos[" << idxL << "]):" << ftrPos.t() << endl;
////                        float * Tr3 = Twc.ptr<float>(2);
////                        float *pFtrPos = (float*)(ftrPos.data);
////                        float d = Tr3[0]*pFtrPos[0]+Tr3[1]*pFtrPos[1]+Tr3[2]*pFtrPos[2]+Tr3[3];
////                    cv::Mat Twc = g2o::toCvMat(v1->estimate()*v0->estimate().inverse());
////                    float d = Twc.at<float>(2,0)*ftrPos.at<float>(0)+Twc.at<float>(2,1)*ftrPos.at<float>(1)+
////                              Twc.at<float>(2,2)*ftrPos.at<float>(2)+Twc.at<float>(2,3);
//
////                        float d = Tr3[0]*ftrPos.at<float>(0)+Tr3[1]*ftrPos.at<float>(0)+Tr3[2]*ftrPos.at<float>(0)+pFtrPos[3];
////                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
////                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL],
////                                            idxL,refFrame->mvbIsFixed[idxL]);
//
////                    nValidEdges++;
//
//
//                }
//                else
//                {
//                    nInvalidEdges++;
//                }
//
//            }
//        }
        ++counters;

        return 0;
    }

    template <int D>
    int static PoseEstimationDirectDepthTopLayer ( Frame * refFrame, Frame *curFrame, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;//第二个参数对一元边无影响
        DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();//就是6*6矩阵的求解器
        DirectBlock* solver_ptr = new DirectBlock( linearSolver );

//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
//        optimizer.setVerbose( true );       // 打开调试输出

        g2o::InverseDirectVertexSE3Expmap* pose = new g2o::InverseDirectVertexSE3Expmap();
        pose->setEstimate( Converter::toSE3Quat(curFrame->mTcw) );
        pose->setId(0);
        optimizer.addVertex( pose );

        // 添加边
        int id=1;
//        for( Measurement m: measurements )

        std::vector<double> vDepthBefore;
        std::vector<int> vKeyIndices;
        std::vector<g2o::OptimizableGraph::Edge*> vEdges;
        for(int i = 0;i<refFrame->N;i++)
        {

            if(refFrame->mvDepth[i]>0)
            {

                g2o::DirectVertexDepth* vDepth = new g2o::DirectVertexDepth();
                vDepth->setId(id);
                vDepth->setEstimate(refFrame->mvDepth[i]);
                vDepth->setMarginalized(true);
                optimizer.addVertex(vDepth);

                vDepthBefore.push_back(refFrame->mvDepth[i]);
                const cv::KeyPoint &kpt = refFrame->mvKeys[i];
                cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(i);
                Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                g2o::EdgeSE3StereoDepthProjectDirect<D>* edge = new g2o::EdgeSE3StereoDepthProjectDirect<D>(
                        refFrame,curFrame,
                        Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[i]),
                        feature,pattern);
                edge->setVertex( 0, vDepth );
                edge->setVertex( 1, pose);
                edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                edge->setId( id++ );
//                if(bRobust)
//                {
//                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                    edge->setRobustKernel(rk);
//                    rk->setDelta(1);
//                }
                optimizer.addEdge(edge);
                vEdges.push_back(edge);
                vKeyIndices.push_back(i);
            }

        }
        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(30);
        int nValidEdges=0,nInvalidEdges=0;
        for(int idxLevel = 1;idxLevel >=0;idxLevel--)
        {
            id = 0;
            nValidEdges = 0,nInvalidEdges = 0;
            cv::Mat Twc = Converter::toCvMat(pose->estimate());

            for(auto edge:vEdges)
            {

                g2o::EdgeSE3StereoDepthProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoDepthProjectDirect<D>*>(edge);


                const int &level = e->mLevel;
                const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edge->vertex(0));
                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                {

                    if(level == 0)
                    {
                        const int &idxL = vKeyIndices[id];
                        refFrame->mvDepth[vKeyIndices[id]] = v->estimate();
                        cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(idxL);
                        float d = Twc.row(2).colRange(0,3).dot(ftrPos) + Twc.at<float>(2,3);
                        curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                                e->mvPredictUVUr[level][2],d,refFrame->mvLives[idxL]);
                    }
#ifdef PRINT_DEBUG
                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
                                                                   refFrame->cy,refFrame->mb,
                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
                                                                   e->mvPredictUVUr[level][0]*(1<<level),
                                                                   e->mvPredictUVUr[level][1]*(1<<level),
                                                                   e->mvPredictUVUr[level][2]*(1<<level));
                    cout << "before: " << vDepthBefore[id]
                         << "\tafter:" << v->estimate()
                            << "\tquad triangulation:" << x3D.at<float>(2)
//                            << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
                         << "\tvalid edge" << endl;

//                DrawTrackedFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                    e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                    e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1]);
#endif
                    nValidEdges++;
                }
                else
                {
//#ifdef PRINT_DEBUG
//                    cout << "before: " << vDepthBefore[id]
//                         << "\tafter:" << v->estimate()
//                         << "\tinvalid edge" << endl;
//#endif
                    nInvalidEdges++;
                }

                ++id;
                e->DecreaseLevel();
            }
#ifdef PRINT_DEBUG
            cout << "# of valid edges:" << nValidEdges <<
                 "\t# of invalid edges:" << nInvalidEdges << endl;
#endif

            optimizer.initializeOptimization();
            optimizer.optimize(10);
        }


        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
        id = 0;
        nValidEdges = 0,nInvalidEdges = 0;
        cv::Mat Twc = Converter::toCvMat(pose->estimate());
        for(auto edge:vEdges)
        {

            g2o::EdgeSE3StereoDepthProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoDepthProjectDirect<D>*>(edge);


            const int &level = e->mLevel;
            const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edge->vertex(0));
            if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
            {

                if(level == 0)
                {
                    refFrame->mvDepth[vKeyIndices[id]] = v->estimate();
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(vKeyIndices[id]);
                    float * Tr3 = Twc.ptr<float>(2);
                    float *pFtrPos = (float*)(ftrPos.data);
                    float d = Tr3[0]*pFtrPos[0]+Tr3[1]*pFtrPos[1]+Tr3[2]*pFtrPos[2]+pFtrPos[3];
                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[vKeyIndices[id]]);
                }
#ifdef PRINT_DEBUG
                cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
                                                                   refFrame->cy,refFrame->mb,
                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
                                                                   e->mvPredictUVUr[level][0]*(1<<level),
                                                                   e->mvPredictUVUr[level][1]*(1<<level),
                                                                   e->mvPredictUVUr[level][2]*(1<<level));
                    cout << "before: " << vDepthBefore[id]
                         << "\tafter:" << v->estimate()
                            << "\tquad triangulation:" << x3D.at<float>(2)
//                            << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
                         << "\tvalid edge" << endl;

//                DrawTrackedFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                    e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                    e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1]);
#endif
                nValidEdges++;
            }
            else
            {
#ifdef PRINT_DEBUG
                cout << "before: " << vDepthBefore[id]
                         << "\tafter:" << v->estimate()
                         << "\tinvalid edge" << endl;
#endif
                nInvalidEdges++;
            }

            ++id;
        }

        curFrame->SetPose(Converter::toCvMat(pose->estimate())*refFrame->mTcw);
        return nValidEdges;
    }


    template <int D>
    int static PoseEstimationDirectDepthPyr( Frame * refFrame, Frame *curFrame, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;//第二个参数对一元边无影响
        DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();//就是6*6矩阵的求解器
        DirectBlock* solver_ptr = new DirectBlock( linearSolver );

//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( true );       // 打开调试输出

        g2o::InverseDirectVertexSE3Expmap* pose = new g2o::InverseDirectVertexSE3Expmap();

        pose->setEstimate( Converter::toSE3Quat(curFrame->mTcw*refFrame->GetInversePose()) );
        pose->setId(0);
        optimizer.addVertex( pose );

        // 添加边
        int id=1;
//        for( Measurement m: measurements )

        std::vector<double> vDepthBefore;
        std::vector<int> vKeyIndices;
        std::vector<g2o::OptimizableGraph::Edge*> vEdges;
        for(int i = 0;i<refFrame->N;i++)
        {

            if(refFrame->mvDepth[i]>0)
            {

                g2o::DirectVertexDepth* vDepth = new g2o::DirectVertexDepth();
                vDepth->setId(id);
                vDepth->setEstimate(refFrame->mvDepth[i]);
                vDepth->setMarginalized(true);
//                vDepth->setFixed(true);
                optimizer.addVertex(vDepth);

                vDepthBefore.push_back(refFrame->mvDepth[i]);
                const cv::KeyPoint &kpt = refFrame->mvKeys[i];
                cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(i);
                Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                g2o::EdgeSE3StereoDepthProjectDirect<D>* edge = new g2o::EdgeSE3StereoDepthProjectDirect<D>(
                        refFrame,curFrame,
                        Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[i]),
                        feature,pattern);
                edge->setVertex( 0, vDepth );
                edge->setVertex( 1, pose);
                edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                edge->setId( id++ );
//                if(bRobust)
//                {
//                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                    edge->setRobustKernel(rk);
//                    rk->setDelta(1);
//                }
                optimizer.addEdge(edge);
                vEdges.push_back(edge);
                vKeyIndices.push_back(i);
            }

        }
        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(30);
        int nValidEdges=0,nInvalidEdges=0;
        for(int idxLevel = refFrame->mnPyrLevel-2;idxLevel >=0;idxLevel--)
        {
            id = 0;
            nValidEdges = 0,nInvalidEdges = 0;
            cv::Mat Twc = Converter::toCvMat(pose->estimate());

            for(auto edge:vEdges)
            {

                g2o::EdgeSE3StereoDepthProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoDepthProjectDirect<D>*>(edge);


                const int &level = e->mLevel;
                const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edge->vertex(0));
                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                {

                    const int &idxL = vKeyIndices[id];
                    if(level == 0)
                    {
                        refFrame->mvDepth[vKeyIndices[id]] = v->estimate();
                        cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(idxL);
                        float d = Twc.row(2).colRange(0,3).dot(ftrPos) + Twc.at<float>(2,3);
                        curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                                e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL]);
                    }
#ifdef PRINT_DEBUG
                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
                                                                   refFrame->cy,refFrame->mb,
                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
                                                                   e->mvPredictUVUr[level][0]*(1<<level),
                                                                   e->mvPredictUVUr[level][1]*(1<<level),
                                                                   e->mvPredictUVUr[level][2]*(1<<level));
                    cout << "before: " << vDepthBefore[id]
                         << "\tafter:" << v->estimate()
                            << "\tquad triangulation:" << x3D.at<float>(2)
//                            << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
                         << "\tvalid edge" << endl;

//                DrawTrackedFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                    e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                    e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1]);
#endif
                    nValidEdges++;
                }
                else
                {
#ifdef PRINT_DEBUG
                    cout << "before: " << vDepthBefore[id]
                         << "\tafter:" << v->estimate()
                         << "\tinvalid edge" << endl;
#endif
                    nInvalidEdges++;
                }

                ++id;
                e->DecreaseLevel();
            }
#ifdef PRINT_DEBUG
            cout << "# of valid edges:" << nValidEdges <<
                 "\t# of invalid edges:" << nInvalidEdges << endl;
#endif

            optimizer.initializeOptimization();
            optimizer.optimize(10);
        }


        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
        id = 0;
        nValidEdges = 0,nInvalidEdges = 0;
        cv::Mat Twc = Converter::toCvMat(pose->estimate());
        for(auto edge:vEdges)
        {

            g2o::EdgeSE3StereoDepthProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoDepthProjectDirect<D>*>(edge);


            const int &level = e->mLevel;
            const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edge->vertex(0));
            if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
            {

                if(level == 0)
                {
                    refFrame->mvDepth[vKeyIndices[id]] = v->estimate();
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(vKeyIndices[id]);
                    float * Tr3 = Twc.ptr<float>(2);
                    float *pFtrPos = (float*)(ftrPos.data);
                    float d = Tr3[0]*pFtrPos[0]+Tr3[1]*pFtrPos[1]+Tr3[2]*pFtrPos[2]+pFtrPos[3];
                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[vKeyIndices[id]]);
                }
#ifdef PRINT_DEBUG
                cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
                                                                   refFrame->cy,refFrame->mb,
                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
                                                                   e->mvPredictUVUr[level][0]*(1<<level),
                                                                   e->mvPredictUVUr[level][1]*(1<<level),
                                                                   e->mvPredictUVUr[level][2]*(1<<level));
                    cout << "before: " << vDepthBefore[id]
                         << "\tafter:" << v->estimate()
                            << "\tquad triangulation:" << x3D.at<float>(2)
//                            << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
                         << "\tvalid edge" << endl;

//                DrawTrackedFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                    e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                    e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1]);
#endif
                nValidEdges++;
            }
            else
            {
#ifdef PRINT_DEBUG
                cout << "before: " << vDepthBefore[id]
                         << "\tafter:" << v->estimate()
                         << "\tinvalid edge" << endl;
#endif
                nInvalidEdges++;
            }

            ++id;
        }

        curFrame->SetPose(Converter::toCvMat(pose->estimate())*refFrame->mTcw);
        return nValidEdges;
    }

    struct EdgeInfo{

        enum EdgeTpye{
            FixedDepth,
            OptimizedDepth
        };
        Frame* pf;
        g2o::OptimizableGraph::Edge *edge;
        int kpIdx;
        EdgeTpye eType;
        EdgeInfo(g2o::OptimizableGraph::Edge *edge,const int &idx,const EdgeTpye &eType):edge(edge),
                                                                                         kpIdx(idx),
                                                                                         eType(eType)
        {}
        EdgeInfo(g2o::OptimizableGraph::Edge *edge,Frame* pf,const int &idx,const EdgeTpye &eType):pf(pf),edge(edge),
                                                                                         kpIdx(idx),
                                                                                         eType(eType)
        {}
    };

    template <int D>
    int static PoseEstimationDirectDepth2( Frame * refFrame, Frame *curFrame, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
        static int counters{0};
//        const int counterTh = 90;
//        const float chi2Th = 100;
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;//第二个参数对一元边无影响
        DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();//就是6*6矩阵的求解器
        DirectBlock* solver_ptr = new DirectBlock( linearSolver );

//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( false );       // 打开调试输出

        g2o::InverseDirectVertexSE3Expmap* pose = new g2o::InverseDirectVertexSE3Expmap();

        pose->setEstimate( Converter::toSE3Quat(curFrame->mTcw*refFrame->GetInversePose()) );
        pose->setId(0);
        optimizer.addVertex( pose );

        // 添加边
        int id=1;

        std::vector<double> vDepthBefore;
        std::vector<int> vKeyIndices;
        std::vector<EdgeInfo> vEdgeInfo;
        for(int i = 0;i<refFrame->N;i++)
        {

            if(refFrame->mvDepth[i]>0)
            {


                g2o::DirectVertexDepth* vDepth = new g2o::DirectVertexDepth();
                vDepth->setId(id);
                vDepth->setEstimate(refFrame->mvDepth[i]);
                vDepth->setMarginalized(true);
                optimizer.addVertex(vDepth);

                vDepthBefore.push_back(refFrame->mvDepth[i]);
                const cv::KeyPoint &kpt = refFrame->mvKeys[i];
                cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(i);
                Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                g2o::EdgeSE3StereoDepthProjectDirect<D>* edge = new g2o::EdgeSE3StereoDepthProjectDirect<D>(
                        refFrame,curFrame,
                        Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[i]),
                        feature,pattern);
                edge->setVertex( 0, vDepth );
                edge->setVertex( 1, pose);
                edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                edge->setId( id++ );
                edge->SetUseInformation(true);
//                if(bRobust)
//                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(5);
                    edge->setRobustKernel(rk);
//                }
                optimizer.addEdge(edge);
                vEdgeInfo.push_back(EdgeInfo(edge,i,EdgeInfo::EdgeTpye::OptimizedDepth));
            }

        }
        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(30);
        int nValidEdges=0, nInvalidEdges=0;
        for(int idxLevel = refFrame->mnPyrLevel-2;idxLevel >=0;idxLevel--)
        {
            id = 0;
            nValidEdges = 0,nInvalidEdges = 0;
            cv::Mat Twc = Converter::toCvMat(pose->estimate());

            for(auto edgeInfo:vEdgeInfo)
            {

//                edgeInfo.edge->computeError();
//                float chi2 = edgeInfo.edge->chi2();
//                if(idxLevel<2)
//                {
//
//                    if(edgeInfo.edge->chi2()>chi2Th)
//                    {
//                        edgeInfo.edge->setLevel(1);
//                    }
//                    else
//                    {
//                        edgeInfo.edge->setLevel(0);
//                    }
//                }

                if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
                {
                    g2o::EdgeSE3StereoDepthProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoDepthProjectDirect<D>*>(edgeInfo.edge);
                    cv::Mat meas = (cv::Mat_<float>(2,1) << e->mMeasUVUr[0], e->mMeasUVUr[1]);


//                    const int &level = e->mLevel;
//                    cout << chi2/cv::norm(meas,pred) << endl;
//                    const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(0));
//                    cout << "level " << level << "before: " << refFrame->mvDepth[edgeInfo.kpIdx]
//                         << "\tafter:" << v->estimate()
//                         << "\tquad triangulation:" << x3D.at<float>(2)
//                         << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
#ifdef PRINT_DEBUG
                    if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                    {
//                        const int &idxL = edgeInfo.kpIdx;
                        nValidEdges++;
                    }
                    else
                    {
                        nInvalidEdges++;
                    }
#endif
                    e->DecreaseLevel();
                }
            }
#ifdef PRINT_DEBUG
            cout << "# of valid edges:" << nValidEdges <<
                 "\t# of invalid edges:" << nInvalidEdges << endl;
#endif

            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }


        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
        id = 0;
        nValidEdges = 0,nInvalidEdges = 0;
        cv::Mat Twc = Converter::toCvMat(pose->estimate());
//        cout << "Twc:"  << Twc << endl;
        for(auto edgeInfo:vEdgeInfo)
        {
            edgeInfo.edge->computeError();

//            if(edgeInfo.edge->chi2()>chi2Th)
//            {
//                continue;
//            }



            if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
            {
                g2o::EdgeSE3StereoDepthProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoDepthProjectDirect<D>*>(edgeInfo.edge);
                const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(0));

                const int &level = e->mLevel;
                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                {
                    const int &idxL = edgeInfo.kpIdx;
                    if(refFrame->mvLives[idxL]>2)
                    {
                        ++refFrame->mvLives[idxL];
                        continue;
                    }


//#ifdef PRINT_DEBUG
//                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
//                                                                   refFrame->cy,refFrame->mb,
//                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
//                                                                   e->mvPredictUVUr[level][0]*(1<<level),
//                                                                   e->mvPredictUVUr[level][1]*(1<<level),
//                                                                   e->mvPredictUVUr[level][2]*(1<<level));
//                    cout << "before: " << refFrame->mvDepth[idxL]
//                         << "\tafter:" << v->estimate()
//                         << "\tquad triangulation:" << x3D.at<float>(2)
//                         << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
//#endif

//                        cout << "(v->estimate()-refFrame->mvDepth[" << idxL << "]):" << v->estimate()-refFrame->mvDepth[idxL] << endl;


                    const cv::KeyPoint &kpt = refFrame->mvKeys[idxL];
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(kpt.pt.x,kpt.pt.y,v->estimate());
                    refFrame->mvDepth[idxL] = v->estimate();
//                    cout << "ftrPos[" << idxL << "]):" << ftrPos.t() << endl;
//                        float * Tr3 = Twc.ptr<float>(2);
//                        float *pFtrPos = (float*)(ftrPos.data);
//                        float d = Tr3[0]*pFtrPos[0]+Tr3[1]*pFtrPos[1]+Tr3[2]*pFtrPos[2]+Tr3[3];
                    float d = Twc.at<float>(2,0)*ftrPos.at<float>(0)+Twc.at<float>(2,1)*ftrPos.at<float>(1)+
                              Twc.at<float>(2,2)*ftrPos.at<float>(2)+Twc.at<float>(2,3);

                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL],
                                            idxL,refFrame->mvbIsFixed[idxL]);

                    nValidEdges++;

//                    if(counters == counterTh)
//                    {
//                        cout << "ch2: " << e->chi2() << endl;
//                        MathUtilis::DrawQuasiFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                                      refFrame->mImgPyrR[level],curFrame->mImgPyrR[level],
//                                                      e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
//                                                      e->mMeasUVUr[2]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][2],e->mvPredictUVUr[level][1]);
//                    }


                }
                else
                {
                    nInvalidEdges++;
                }

            }
        }
        ++counters;
        curFrame->SetPose(/*Converter::toCvMat(pose->estimate())*/Twc*refFrame->mTcw);
        return nValidEdges;
    }


    template <int D>
    int static PoseEstimationDirectDepthAffine( Frame * refFrame, Frame *curFrame, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
        static int counters{0};
//        const int counterTh = 90;
//        const float chi2Th = 100;
//        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> DirectBlock;//第二个参数对一元边无影响
        g2o::BlockSolver<g2o::BlockSolverX>::LinearSolverType * linearSolver = new g2o::LinearSolverDense< g2o::BlockSolver<g2o::BlockSolverX>::PoseMatrixType > ();//就是6*6矩B阵的求解器
        g2o::BlockSolver<g2o::BlockSolverX>* solver_ptr = new g2o::BlockSolver<g2o::BlockSolverX>( linearSolver );

//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( false );       // 打开调试输出

        g2o::InverseDirectVertexSE3Expmap* pose = new g2o::InverseDirectVertexSE3Expmap();

        pose->setEstimate( Converter::toSE3Quat(curFrame->mTcw*refFrame->GetInversePose()) );
        pose->setId(0);
        optimizer.addVertex( pose );

        g2o::IntensityAffine* vAffine = new g2o::IntensityAffine();
        vAffine->setEstimate(Eigen::Vector2d(1,0));
        vAffine->setId(1);
        vAffine->setFixed(false);
//        vAffine->setMarginalized(true);
        optimizer.addVertex( vAffine );
        // 添加边
        int id=2;

        std::vector<double> vDepthBefore;
        std::vector<int> vKeyIndices;
        std::vector<EdgeInfo> vEdgeInfo;
        for(int i = 0;i<refFrame->N;i++)
        {

            if(refFrame->mvDepth[i]>0)
            {


                g2o::DirectVertexDepth* vDepth = new g2o::DirectVertexDepth();
                vDepth->setId(id);
                Eigen::Vector3d depthEstimate;
                vDepth->setEstimate(/*1/*/refFrame->mvDepth[i]);
                vDepth->setMarginalized(true);
                optimizer.addVertex(vDepth);
                vDepthBefore.push_back(refFrame->mvDepth[i]);

                const cv::KeyPoint &kpt = refFrame->mvKeys[i];
                cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(i);
                Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                g2o::EdgeSE3StereoDirectInverseDepthWithAffine<D>* edge = new g2o::EdgeSE3StereoDirectInverseDepthWithAffine<D>(
                        refFrame,curFrame,
                        Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[i]),
                        feature,pattern);
                edge->setVertex( 0, vDepth );
                edge->setVertex( 1, pose);
                edge->setVertex( 2, vAffine);
                edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                edge->setId( id++ );
                edge->SetUseInformation(true);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                rk->setDelta(5);
                edge->setRobustKernel(rk);

                optimizer.addEdge(edge);
                vEdgeInfo.push_back(EdgeInfo(edge,i,EdgeInfo::EdgeTpye::OptimizedDepth));
            }

        }
        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        int nValidEdges=0, nInvalidEdges=0;
        for(int idxLevel = refFrame->mnPyrLevel-2;idxLevel >=0;idxLevel--)
        {
            id = 0;
            nValidEdges = 0,nInvalidEdges = 0;
            cv::Mat Twc = Converter::toCvMat(pose->estimate());

            if(idxLevel<1)
                vAffine->setFixed(false);
            for(auto edgeInfo:vEdgeInfo)
            {

//                edgeInfo.edge->computeError();
//                float chi2 = edgeInfo.edge->chi2();
//                cout << "chi2: " << chi2 << endl;
//                if(idxLevel<2)
//                {
//
//                    if(edgeInfo.edge->chi2()>chi2Th)
//                    {
//                        edgeInfo.edge->setLevel(1);
//                    }
//                    else
//                    {
//                        edgeInfo.edge->setLevel(0);
//                    }
//                }

                if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
                {
                    g2o::EdgeSE3StereoDirectInverseDepthWithAffine<D>* e = static_cast<g2o::EdgeSE3StereoDirectInverseDepthWithAffine<D>*>(edgeInfo.edge);

//                    const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(0));
//                    cout << "level " << level << "before: " << refFrame->mvDepth[edgeInfo.kpIdx]
//                         << "\tafter:" << v->estimate()
//                         << "\tquad triangulation:" << x3D.at<float>(2)
//                         << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
#ifdef PRINT_DEBUG
                    if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                    {
//                        const int &idxL = edgeInfo.kpIdx;
                        nValidEdges++;
                    }
                    else
                    {
                        nInvalidEdges++;
                    }
#endif
                    e->DecreaseLevel();
                }
            }
#ifdef PRINT_DEBUG
            cout << "# of valid edges:" << nValidEdges <<
                 "\t# of invalid edges:" << nInvalidEdges << endl;
#endif

            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }


        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
        id = 0;
        nValidEdges = 0,nInvalidEdges = 0;
        cv::Mat Twc = Converter::toCvMat(pose->estimate());
//        cout << "Twc:"  << Twc << endl;
        for(auto edgeInfo:vEdgeInfo)
        {
//            edgeInfo.edge->computeError();
//
//            if(edgeInfo.edge->chi2()>chi2Th)
//            {
//                continue;
//            }


            if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
            {
                g2o::EdgeSE3StereoDirectInverseDepthWithAffine<D>* e = static_cast<g2o::EdgeSE3StereoDirectInverseDepthWithAffine<D>*>(edgeInfo.edge);
                const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(0));

                const int &level = e->mLevel;
                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                {
                    const int &idxL = edgeInfo.kpIdx;
                    if(refFrame->mvLives[idxL]>4)
                    {
                        ++refFrame->mvLives[idxL];
                        continue;
                    }


//#ifdef PRINT_DEBUG
//                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
//                                                                   refFrame->cy,refFrame->mb,
//                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
//                                                                   e->mvPredictUVUr[level][0]*(1<<level),
//                                                                   e->mvPredictUVUr[level][1]*(1<<level),
//                                                                   e->mvPredictUVUr[level][2]*(1<<level));
//                    cout << "before: " << refFrame->mvDepth[idxL]
//                         << "\tafter:" << v->estimate()
//                         << "\tquad triangulation:" << x3D.at<float>(2)
//                         << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
//#endif

//                        cout << "(v->estimate()-refFrame->mvDepth[" << idxL << "]):" << v->estimate()-refFrame->mvDepth[idxL] << endl;


                    const cv::KeyPoint &kpt = refFrame->mvKeys[idxL];
                    const double updateDepth = /*1/*/v->estimate();
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(kpt.pt.x,kpt.pt.y,updateDepth);

                    refFrame->mvDepth[idxL] = updateDepth;
//                    cout << "ftrPos[" << idxL << "]):" << ftrPos.t() << endl;
                    float d = Twc.at<float>(2,0)*ftrPos.at<float>(0)+Twc.at<float>(2,1)*ftrPos.at<float>(1)+
                              Twc.at<float>(2,2)*ftrPos.at<float>(2)+Twc.at<float>(2,3);

//                        float d = Tr3[0]*ftrPos.at<float>(0)+Tr3[1]*ftrPos.at<float>(0)+Tr3[2]*ftrPos.at<float>(0)+pFtrPos[3];
                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL],
                                            idxL,refFrame->mvbIsFixed[idxL]);

                    nValidEdges++;

//                    if(counters == counterTh)
//                    {
//                        cout << "ch2: " << e->chi2() << endl;
//                        MathUtilis::DrawQuasiFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                                      refFrame->mImgPyrR[level],curFrame->mImgPyrR[level],
//                                                      e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
//                                                      e->mMeasUVUr[2]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][2],e->mvPredictUVUr[level][1]);
//                    }


                }
                else
                {
                    nInvalidEdges++;
                }

            }
        }
        ++counters;

        cout << "vAffine->estimate(): " << vAffine->estimate().transpose() <<endl;
        curFrame->SetPose(/*Converter::toCvMat(pose->estimate())*/Twc*refFrame->mTcw);
//        cout << curFrame->mTcw;
        return nValidEdges;
    }

    template <int D>
    int static PoseEstimationDirectDepth( Frame * refFrame, Frame *curFrame, g2o::Pattern<D> *pattern)
    {
        // 初始化g2o
        static int counters{0};
//        const int counterTh = 90;
//        const float chi2Th = 1000;
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;//第二个参数对一元边无影响
        DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();//就是6*6矩阵的求解器
        DirectBlock* solver_ptr = new DirectBlock( linearSolver );

//        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );
        optimizer.setVerbose( false );       // 打开调试输出

        g2o::InverseDirectVertexSE3Expmap* pose = new g2o::InverseDirectVertexSE3Expmap();

        pose->setEstimate( Converter::toSE3Quat(curFrame->mTcw*refFrame->GetInversePose()) );
        pose->setId(0);
        optimizer.addVertex( pose );

        // 添加边
        int id=1;

        std::vector<double> vDepthBefore;
        std::vector<int> vKeyIndices;
        std::vector<EdgeInfo> vEdgeInfo;
        for(int i = 0;i<refFrame->N;i++)
        {

            if(refFrame->mvDepth[i]>0)
            {

//                if(!refFrame->mvbIsValid[i])
//                    continue;

                if(/*refFrame->mvLives[i]>5*/refFrame->mvbIsFixed[i])
                {
                    const cv::KeyPoint &kpt = refFrame->mvKeys[i];
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(i);
                    Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                    g2o::EdgeSE3StereoProjectDirect<D>* edge = new g2o::EdgeSE3StereoProjectDirect<D>(
                            refFrame,curFrame,
                            Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[i]),
                            feature,pattern);

                    edge->setVertex( 0, pose );
                    edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                    edge->setId( id++ );
//                if(bRobust)
//                {
//                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                    rk->setDelta(100);
//                    edge->setRobustKernel(rk);

//                }
                    optimizer.addEdge(edge);
                    vEdgeInfo.push_back(EdgeInfo(edge,i,EdgeInfo::EdgeTpye::FixedDepth));
                }
                else
                {
                    g2o::DirectVertexDepth* vDepth = new g2o::DirectVertexDepth();
                    vDepth->setId(id);
                    vDepth->setEstimate(refFrame->mvDepth[i]);
//                    vDepth->setEstimate(10);
                    vDepth->setMarginalized(true);
                    optimizer.addVertex(vDepth);

                    vDepthBefore.push_back(refFrame->mvDepth[i]);
                    const cv::KeyPoint &kpt = refFrame->mvKeys[i];
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(i);
                    Eigen::Vector3d feature(ftrPos.at<float>(0),ftrPos.at<float>(1),ftrPos.at<float>(2));

                    g2o::EdgeSE3StereoDepthProjectDirect<D>* edge = new g2o::EdgeSE3StereoDepthProjectDirect<D>(
                            refFrame,curFrame,
                            Eigen::Vector3d(kpt.pt.x,kpt.pt.y,refFrame->mvuRight[i]),
                            feature,pattern);
                    edge->setVertex( 0, vDepth );
                    edge->setVertex( 1, pose);
                    edge->setInformation( Eigen::Matrix<double,D*2,D*2>::Identity() );
                    edge->setId( id++ );
                    edge->SetUseInformation(true);
//                if(bRobust)
//                {
//                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                    rk->setDelta(100);
//                    edge->setRobustKernel(rk);
//                }
                    optimizer.addEdge(edge);
                    vEdgeInfo.push_back(EdgeInfo(edge,i,EdgeInfo::EdgeTpye::OptimizedDepth));

                }

            }

        }
        cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(30);
        int nValidEdges=0, nInvalidEdges=0;
        for(int idxLevel = refFrame->mnPyrLevel-2;idxLevel >=0;idxLevel--)
        {
            id = 0;
            nValidEdges = 0,nInvalidEdges = 0;
            cv::Mat Twc = Converter::toCvMat(pose->estimate());

            for(auto edgeInfo:vEdgeInfo)
            {

//                if(idxLevel <2)
//                {
//
//                    edgeInfo.edge->computeError();
//                    if(edgeInfo.edge->chi2()>chi2Th)
//                    {
//                        edgeInfo.edge->setLevel(1);
//                    }
//                    else
//                    {
//                        edgeInfo.edge->setLevel(0);
//                    }
//                }

                if(edgeInfo.eType == EdgeInfo::EdgeTpye::FixedDepth)
                {
                    g2o::EdgeSE3StereoProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoProjectDirect<D>*>(edgeInfo.edge);
//                    const int &level = e->mLevel;
#ifdef PRINT_DEBUG
                    if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                    {

//                        const int &idxL = edgeInfo.kpIdx;
                        nValidEdges++;
                    }
                    else
                    {
                        nInvalidEdges++;
                    }
#endif

                    e->DecreaseLevel();
                }
                if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
                {
                    g2o::EdgeSE3StereoDepthProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoDepthProjectDirect<D>*>(edgeInfo.edge);
//                    const int &level = e->mLevel;
//                    const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(0));
//                    cout << "level " << level << "before: " << refFrame->mvDepth[edgeInfo.kpIdx]
//                         << "\tafter:" << v->estimate()
//                         << "\tquad triangulation:" << x3D.at<float>(2)
//                         << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
#ifdef PRINT_DEBUG
                    if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                    {
//                        const int &idxL = edgeInfo.kpIdx;
                        nValidEdges++;
                    }
                    else
                    {
                        nInvalidEdges++;
                    }
#endif
                    e->DecreaseLevel();
                }
            }
#ifdef PRINT_DEBUG
            cout << "# of valid edges:" << nValidEdges <<
                 "\t# of invalid edges:" << nInvalidEdges << endl;
#endif

            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }


        //add tracked features to  current frame
        // including location in both camera, depth in left camera, tracked lives
        // update referrence frame's feature lives
        id = 0;
        nValidEdges = 0,nInvalidEdges = 0;
        cv::Mat Twc = Converter::toCvMat(pose->estimate());
        cout << "Twc:"  << Twc << endl;
        for(auto edgeInfo:vEdgeInfo)
        {
            edgeInfo.edge->computeError();

            float chi2 = edgeInfo.edge->chi2();
//            cout << "edgeInfo.edge->chi2():" << edgeInfo.edge->chi2() << endl;

            if(edgeInfo.eType == EdgeInfo::EdgeTpye::FixedDepth)
            {
                g2o::EdgeSE3StereoProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoProjectDirect<D>*>(edgeInfo.edge);

                const int &level = e->mLevel;
                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                {

                    const int &idxL = edgeInfo.kpIdx;
                    if(refFrame->mvLives[idxL]>3)
                    {
                        ++refFrame->mvLives[idxL];
                        continue;
                    }
                    const cv::KeyPoint &kpt = refFrame->mvKeys[idxL];
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(kpt.pt.x,kpt.pt.y,refFrame->mvDepth[idxL]);
//                    cv::Mat OR = (cv::Mat_<float>(3,1)<<refFrame->mb,0,0);
//                    float cosparallex = (OR/cv::norm(OR)).dot(-ftrPos/cv::norm(ftrPos));
//                    if(refFrame->mvLives[idxL]>3 && cosparallex<0.5)
//                    {
//
//                        refFrame->mvbIsFixed[idxL] = true;
//                    }
//                        float * Tr3 = Twc.ptr<float>(2);
//                        float *pFtrPos = (float*)(ftrPos.data);
//                        float d = Tr3[0]*pFtrPos[0]+Tr3[1]*pFtrPos[1]+Tr3[2]*pFtrPos[2]+pFtrPos[3];
                    float d = Twc.at<float>(2,0)*ftrPos.at<float>(0)+Twc.at<float>(2,1)*ftrPos.at<float>(1)+
                              Twc.at<float>(2,2)*ftrPos.at<float>(2)+Twc.at<float>(2,3);
                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL],
                                            idxL,refFrame->mvbIsFixed[idxL]);
//#ifdef PRINT_DEBUG
//                    if(counters == counterTh)
//                    {
//                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
//                                                                   refFrame->cy,refFrame->mb,
//                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
//                                                                   e->mvPredictUVUr[level][0]*(1<<level),
//                                                                   e->mvPredictUVUr[level][1]*(1<<level),
//                                                                   e->mvPredictUVUr[level][2]*(1<<level));
//                    cout << "before: " << vDepthBefore[id]
//                         << "\tafter:" << v->estimate()
//                            << "\tquad triangulation:" << x3D.at<float>(2)
////                            << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
//                        cout << "ch2: " << e->chi2() << endl;
//                        MathUtilis::DrawQuasiFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                                      refFrame->mImgPyrR[level],curFrame->mImgPyrR[level],
//                                                      e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
//                                                      e->mMeasUVUr[2]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][2],e->mvPredictUVUr[level][1]);
//                    }


//#endif
                    nValidEdges++;
                }
                else
                {
                    nInvalidEdges++;
                }
            }
            if(edgeInfo.eType == EdgeInfo::EdgeTpye::OptimizedDepth)
            {
                g2o::EdgeSE3StereoDepthProjectDirect<D>* e = static_cast<g2o::EdgeSE3StereoDepthProjectDirect<D>*>(edgeInfo.edge);
                const g2o::DirectVertexDepth* v = static_cast<const g2o::DirectVertexDepth*>(edgeInfo.edge->vertex(0));

                const int &level = e->mLevel;
//                cv::Mat meas = (cv::Mat_<float>(2,1) << e->mMeasUVUr[0], e->mMeasUVUr[1]);
//                cv::Mat pred = (cv::Mat_<float>(2,1) << e->mvPredictUVUr[level][0], e->mvPredictUVUr[level][1])/e->mFactorInv;
//
//                float score = chi2/cv::norm(meas,pred);
//                cout << "chi2: " << chi2 << " cv::norm(meas,pred): " << cv::norm(meas,pred) << " score: " << score << endl;
//                MathUtilis::DrawQuasiFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                              refFrame->mImgPyrR[level],curFrame->mImgPyrR[level],
//                                              e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                              e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
//                                              e->mMeasUVUr[2]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                              e->mvPredictUVUr[level][2],e->mvPredictUVUr[level][1]);
//                if(edgeInfo.edge->chi2()>chi2Th)
//                {
//                    continue;
//                }
                if(!e->mbRefOutBorder[level] && e->mIsVisible[level])
                {
                    const int &idxL = edgeInfo.kpIdx;
                    if(refFrame->mvLives[idxL]>5)
                    {
                        ++refFrame->mvLives[idxL];
                        continue;
                    }


//#ifdef PRINT_DEBUG
//                    cv::Mat x3D = MathUtilis::TriangulateQuadMatch(Twc,refFrame->fx,refFrame->fy,refFrame->cx,
//                                                                   refFrame->cy,refFrame->mb,
//                                                                   e->mMeasUVUr[0],e->mMeasUVUr[1],e->mMeasUVUr[2],
//                                                                   e->mvPredictUVUr[level][0]*(1<<level),
//                                                                   e->mvPredictUVUr[level][1]*(1<<level),
//                                                                   e->mvPredictUVUr[level][2]*(1<<level));
//                    cout << "before: " << refFrame->mvDepth[idxL]
//                         << "\tafter:" << v->estimate()
//                         << "\tquad triangulation:" << x3D.at<float>(2)
//                         << "\tdepth in k:" << refFrame->mbf/((e->mvPredictUVUr[level][0] - e->mvPredictUVUr[level][2])*(1<<level))
//                         << "\tvalid edge" << endl;
//#endif

//                        cout << "(v->estimate()-refFrame->mvDepth[" << idxL << "]):" << v->estimate()-refFrame->mvDepth[idxL] << endl;


                    const cv::KeyPoint &kpt = refFrame->mvKeys[idxL];
                    cv::Mat ftrPos = refFrame->UnprojectStereoInCamera(kpt.pt.x,kpt.pt.y,v->estimate());
//                    cv::Mat OR = (cv::Mat_<float>(3,1)<<refFrame->mb,0,0);
//                    float cosparallex = (OR/cv::norm(OR)).dot(-ftrPos/cv::norm(ftrPos));
//                    if(refFrame->mvLives[idxL]>3 && cosparallex<0.5)
//                    {
//
//                        refFrame->mvbIsFixed[idxL] = true;
//                    }
                    if(fabs(v->estimate()-refFrame->mvDepth[idxL])<0.01)
                        refFrame->mvbIsFixed[idxL] = true;
                    refFrame->mvDepth[idxL] = v->estimate();
//                    cout << "ftrPos[" << idxL << "]):" << ftrPos.t() << endl;
//                        float * Tr3 = Twc.ptr<float>(2);
//                        float *pFtrPos = (float*)(ftrPos.data);
//                        float d = Tr3[0]*pFtrPos[0]+Tr3[1]*pFtrPos[1]+Tr3[2]*pFtrPos[2]+Tr3[3];
                    float d = Twc.at<float>(2,0)*ftrPos.at<float>(0)+Twc.at<float>(2,1)*ftrPos.at<float>(1)+
                            Twc.at<float>(2,2)*ftrPos.at<float>(2)+Twc.at<float>(2,3);

//                        float d = Tr3[0]*ftrPos.at<float>(0)+Tr3[1]*ftrPos.at<float>(0)+Tr3[2]*ftrPos.at<float>(0)+pFtrPos[3];
                    curFrame->AddSteroMatch(e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
                                            e->mvPredictUVUr[level][2],d,++refFrame->mvLives[idxL],
                                            idxL,refFrame->mvbIsFixed[idxL]);

                    nValidEdges++;

//                    if(counters == counterTh)
//                    {
//                        cout << "ch2: " << e->chi2() << endl;
//                        MathUtilis::DrawQuasiFeatures(refFrame->mImgPyrL[level],curFrame->mImgPyrL[level],
//                                                      refFrame->mImgPyrR[level],curFrame->mImgPyrR[level],
//                                                      e->mMeasUVUr[0]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][0],e->mvPredictUVUr[level][1],
//                                                      e->mMeasUVUr[2]*e->mFactorInv,e->mMeasUVUr[1]*e->mFactorInv,
//                                                      e->mvPredictUVUr[level][2],e->mvPredictUVUr[level][1]);
//                    }


                }
                else
                {
                    nInvalidEdges++;
                }

            }
        }
        ++counters;
        curFrame->SetPose(/*Converter::toCvMat(pose->estimate())*/Twc*refFrame->mTcw);
        return nValidEdges;
    }
};

} //namespace ORB_SLAM

#endif // OPTIMIZER_H
