//*Binocular Stereo Calibration and Rectification using Minoru 3D camera*// 
//20 Feb 2016//
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;



void stereo_calibrate()
{

 //Create objects to access left and right camera images.
VideoCapture cap_1(1);
VideoCapture cap_2(2);

//cap_1.set(CV_CAP_PROP_FRAME_WIDTH, 320);
//cap_1.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
 if(cap_1.isOpened()==0)
 {
   cout<<"Cannot open video cam."<<endl;
   exit;
 }
 
//cap_2.set(CV_CAP_PROP_FRAME_WIDTH, 320);
//cap_2.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
 if(cap_2.isOpened()==0)
 {
   cout<<"Cannot open video cam."<<endl;
   exit;
 }
 namedWindow("Left Image",CV_WINDOW_AUTOSIZE);
 namedWindow("Right Image",CV_WINDOW_AUTOSIZE);
 //namedWindow("Detect",CV_WINDOW_AUTOSIZE);

 Mat image_l,image_r;
 Mat image_lg,image_rg;
 Size imageSize;
 //Acquire checkerboard size.
 const float squareSize=2.5f;
 Size boardSize;
 cout<<"Enter the board height:"<<endl;
 cin>>boardSize.height;
 cout<<"Enter the board width:"<<endl;
 cin>>boardSize.width;
 //specify vectors for storing checkerboard image and object points.
 vector<vector<Point2f> > imagePoints[2];
 vector<vector<Point3f> > objectPoints;
 //Acquire number of images to capture.
 int n_images;
 int count=0;
 cout<<"Enter the number of checkerboard image pairs to capture:"<<endl;
 cin>>n_images;
 n_images*=20;

 while(count<=n_images)
 {
    cap_1.read(image_l);
    cap_2.read(image_r);
    //converting images to grayscale.
    cvtColor(image_l,image_lg,CV_BGR2GRAY);
    cvtColor(image_r,image_rg,CV_BGR2GRAY);

    vector<Point2f> corners_1,corners_2;

    //Calculate the checkerboard corners.
    bool found_1 = findChessboardCorners(image_lg, boardSize, corners_1, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    bool found_2 = findChessboardCorners(image_rg, boardSize, corners_2, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    
    //Display corners.
    drawChessboardCorners(image_lg, boardSize, corners_1, found_1); 
    drawChessboardCorners(image_rg, boardSize, corners_2, found_2);
    //Display both the images.
    imshow("Left Image",image_lg);
    imshow("Right Image",image_rg); 
    waitKey(1);
    

    
    if( found_1 && found_2 )
    {   
        if(count!=0 && count%20==0)
      {
        //refine corners using subpixel accuracy interpolation.
        cornerSubPix(image_lg, corners_1, Size(11,11), Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
        cornerSubPix(image_rg, corners_2, Size(11,11), Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));

        //Save corners in vector.
        imagePoints[0].push_back(corners_1);
        imagePoints[1].push_back(corners_2);
        cout<<"Checkerboard image pair "<<count/20<<" successfully saved."<<endl;
      }
        count++;   
    }
  }
    destroyAllWindows();
    imagePoints[0].resize(n_images/20);
    imagePoints[1].resize(n_images/20);
    objectPoints.resize(n_images/20);
    imageSize=image_lg.size();

    for( int i = 0; i < n_images/20; i++ )
    {
        for( int j = 0; j < boardSize.height; j++ )
            for( int k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    cout<<"Running Stereo Calibration."<<endl;
    //Defining the calibration parameters.
    //*cameraMatrix[2],distCoeffs[2]=camera intrinsic parameters, distortion parameters.
    //*R,T=Rotational and Translational geometric relationships between cameras.
    //*E,F=Essential and Fundamental matrices of stereo rig.
    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;
    
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "Done with RMS error=" << rms << endl;
    
    // Save intrinsic parameters to file.
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    //Stereo Rectification parameters.
    //*R1,R2= Rotational transform 3x3 matrices of both cameras.
    //*P1,P2=Projection matrix to world co-ordinates.
    //*Q= Reprojection matrix to convert co-ordinates to Depth.
    //validRoi= ROI of the valid points after rectification.
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
    
    //Save extrinsic parameters in file.
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }


    //Produce joint undistort and rectify maps.
    Mat rmap[2][2];
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    namedWindow("Rectified images",CV_WINDOW_AUTOSIZE);
    Mat left_rect,right_rect,rectified;
    
    while(true)
    {

      cap_1.read(image_l);
      cap_2.read(image_r);

      //converting images to grayscale.
      cvtColor(image_l,image_lg,CV_BGR2GRAY);
      cvtColor(image_r,image_rg,CV_BGR2GRAY);

      //Rectify camera images.
      remap(image_lg, left_rect, rmap[0][0], rmap[0][1], INTER_LINEAR);
      remap(image_rg, right_rect, rmap[1][0], rmap[1][1], INTER_LINEAR);

      //Blend the rectified images.
      addWeighted( left_rect, 0.3, right_rect, 0.7, 0.0, rectified);
      imshow("Rectified images",rectified);
      if (waitKey(1) == 27)
       {
            std::cout << "Esc key is pressed by user" << std::endl;
            destroyAllWindows();
            break; 
       }

    }
    //Disparity matrices.
    Mat disp,disp8;
    namedWindow("Disparity map",CV_WINDOW_AUTOSIZE);

    while(true)
   {

    cap_1.read(image_l);
    cap_2.read(image_r);

    //converting images to grayscale.
    cvtColor(image_l,image_lg,CV_BGR2GRAY);
    cvtColor(image_r,image_rg,CV_BGR2GRAY);

    //Rectify camera images.
    remap(image_lg, left_rect, rmap[0][0], rmap[0][1], INTER_LINEAR);
    remap(image_rg, right_rect, rmap[1][0], rmap[1][1], INTER_LINEAR);

    //Semi Global Disparity matching.
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,80,3);
    int sgbmWinSize = 5;
    sgbm->setBlockSize(3);
    sgbm->setPreFilterCap(4);
    sgbm->setP1(8*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(-80);
    sgbm->setNumDisparities(80);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    //sgbm->setMode(StereoSGBM::MODE_SGBM);

    sgbm->compute(left_rect, right_rect, disp);
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    imshow("Disparity map",disp8);
    if (waitKey(1) == 27)
       {
            std::cout << "Esc key is pressed by user" << std::endl;
            destroyAllWindows();
            break; 
       }
    
   }




}

int main( int argc, char** argv )
{

 stereo_calibrate();
 

}