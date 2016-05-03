//*Binocular Stereo disparity generation using Minoru 3D camera*// 
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
 //namedWindow("Left Image",CV_WINDOW_AUTOSIZE);
 //namedWindow("Right Image",CV_WINDOW_AUTOSIZE);
 //namedWindow("Detect",CV_WINDOW_AUTOSIZE);

 Mat image_l,image_r;
 Mat image_lg,image_rg;
 Size imageSize(640,480);

 // reading intrinsic & extrinsic parameters
        FileStorage fs("Calibration_params/intrinsics.yml", FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file\n");
            exit;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        fs.open("Calibration_params/extrinsics.yml", FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file\n");
            exit;
        }

        Mat R, T;
        fs["R"] >> R;
        fs["T"] >> T;

    //Stereo Rectification parameters.
    //*R1,R2= Rotational transform 3x3 matrices of both cameras.
    //*P1,P2=Projection matrix to world co-ordinates.
    //*Q= Reprojection matrix to convert co-ordinates to Depth.
    //validRoi= ROI of the valid points after rectification.
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(M1, D1,
                M2, D2,
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
    
    

    //Produce joint undistort and rectify maps.
    Mat map11,map12,map21,map22;
    initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M1, D1, R2, P2, imageSize, CV_16SC2, map21, map22);
 
    namedWindow("Rectified images",CV_WINDOW_AUTOSIZE);
    Mat left_rect,right_rect,rectified;
    Mat map_matrix;
    //Loop to display the Disparity Map.
    while(true)
    {

      cap_1.read(image_l);
      cap_2.read(image_r);

      //converting images to grayscale.
      cvtColor(image_l,image_lg,CV_BGR2GRAY);
      cvtColor(image_r,image_rg,CV_BGR2GRAY);

      //Rectify camera images.
      remap(image_lg, left_rect, map11, map12, INTER_LINEAR);
      remap(image_rg, right_rect, map21, map22, INTER_LINEAR);
      
      //Blend the rectified images.
      addWeighted( left_rect, 0.3, right_rect, 0.7, 0.0, rectified);
      map_matrix=getRotationMatrix2D(Point2f(320,240),20.0,1);
      warpAffine(rectified, rectified, map_matrix, Size(640, 480));
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
    remap(image_lg, left_rect, map11, map12, INTER_LINEAR);
    remap(image_rg, right_rect, map21, map22, INTER_LINEAR);

    int sgbmWinSize = 5;
    int block_size, filter_cap,min_disp,num_disp,unique_ratio,speck_size;
    speck_size=100;
    unique_ratio=10;
    num_disp=80;
    min_disp=0;
    filter_cap=4;
    block_size=3;
    //Semi Global Disparity matching.
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,80,3);
    sgbm->setBlockSize(block_size);
    sgbm->setPreFilterCap(filter_cap);
    sgbm->setP1(8*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(min_disp);
    sgbm->setNumDisparities(num_disp);
    sgbm->setUniquenessRatio(unique_ratio);
    sgbm->setSpeckleWindowSize(speck_size);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    //sgbm->setMode(StereoSGBM::MODE_SGBM);

    sgbm->compute(right_rect, left_rect, disp);
    warpAffine(disp, disp, map_matrix, Size(640, 480));
    Rect crop(95,95,380,315);
    disp=disp(crop);
    resize(disp,disp,Size(640,480));
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