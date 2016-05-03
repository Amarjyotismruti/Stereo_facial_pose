//Robust face tracking using Viola Jhones, SIFT features, KLT tracker and delaunay triangulation.//
//*27 Feb 2016*//
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>        
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

Mat disp8;
Mat left_rect;
Mat map11,map12,map21,map22;
//Function to generate intrinsic parameters.
static void stereo_params()
{
     // reading intrinsic & extrinsic parameters
        Size imageSize(640,480);
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
    
    initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M1, D1, R2, P2, imageSize, CV_16SC2, map21, map22);
}
 //Create objects to access left and right camera images.
VideoCapture cap_1(1);
VideoCapture cap_2(2);
//Function for stereo generation.
static void stereo_calibrate()
{



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

 

    Mat right_rect,rectified;
    Mat map_matrix;
    
    //Disparity matrices.
    Mat disp;
   // namedWindow("Disparity map",CV_WINDOW_AUTOSIZE);

    

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
    map_matrix=getRotationMatrix2D(Point2f(320,240),20.0,1);
    warpAffine(disp, disp, map_matrix, Size(640, 480));
    Rect crop(95,95,380,315);
    disp=disp(crop);
    resize(disp,disp,Size(640,480));
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    //remap(image_l, left_rect, map11, map12, INTER_LINEAR);
    warpAffine(right_rect, right_rect, map_matrix, Size(640, 480));

   left_rect=right_rect(crop);
    resize(left_rect,left_rect,Size(640,480));
   // imshow("Disparity map",disp8);
    if (waitKey(1) == 27)
       {
            std::cout << "Esc key is pressed by user" << std::endl;
            destroyAllWindows();
        
       }
    
   




}

//Function for Delaunay triangulation.
static void draw_delaunay( Mat& img, Subdiv2D& subdiv)
{
 
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
 
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
         
        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], Scalar(255,255,255), 1, CV_AA, 0);
            line(img, pt[1], pt[2], Scalar(255,255,255), 1, CV_AA, 0);
            line(img, pt[2], pt[0], Scalar(255,255,255), 1, CV_AA, 0);
        }
    }
}

//Function for extracting face mask.
static void face_mask(Mat &image,vector<Point2f> &corner_prev,vector<Point2f> &corners)
{
    Mat frame_gray,mask,mask_image  ;
    frame_gray=image;
    //cvtColor( image, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    vector<Rect> faces;
    //Detect Faces using Haar Classifiers.
    CascadeClassifier cascade_face;
    cascade_face.load("haarcascade_frontalface_alt.xml");
    cascade_face.detectMultiScale( frame_gray, faces, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for( size_t i = 0; i < faces.size(); i++ )
  {
    
    
    corner_prev[0]=Point2f( faces[i].x, faces[i].y);
    corner_prev[1]=Point2f( faces[i].x + faces[i].width, faces[i].y );
    corner_prev[2]=Point2f( faces[i].x, faces[i].height + faces[i].y );
    corner_prev[3]=Point2f( faces[i].x + faces[i].width, faces[i].y + faces[i].height );
    
    rectangle( image, corner_prev[0], corner_prev[3], Scalar( 255, 0, 0 ), 2, 8, 0 );
    //Face image cropping/masking. 
    int corner_x=faces[i].x+faces[i].width/8;
    int corner_y=faces[i].y+faces[i].height/8;
    int width=5*faces[i].width/7;
    int height=6*faces[i].height/9;

    mask= cv::Mat::zeros(480,640 , CV_8UC1);
    mask_image= cv::Mat::zeros(480,640 , CV_8UC3);
    mask(Rect(corner_x,corner_y,width,height)) = 255;
    image.copyTo( mask_image, mask);

    goodFeaturesToTrack(frame_gray,corners,40,0.01,10,mask,3,false,0.04);
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    cornerSubPix(frame_gray, corners, Size(10,10), Size(-1,-1), termcrit);
   }

}

void face_detect()
{
	Mat frame,frame_1,frame_prev,frame_gray,final;
	vector<Rect> faces;
	

 CascadeClassifier cascade_face;
 cascade_face.load("haarcascade_frontalface_alt.xml");

 if(cascade_face.empty())
 	cout<<"Couldn't load classifier file.";
 //namedWindow("Face Detect",CV_WINDOW_AUTOSIZE);

 //Variables assigned for masking and SIFT features.
 Mat mask,mask_image;
 Mat descriptors,descriptors_1;
 vector<KeyPoint> keypoints,keypoints_1;
 int minhessian=400;
 vector<Point2f> corner_prev(4),corner_new(4);
 //Choose between SURF/SIFT/Shi-Thomasi features.
 //Ptr<xfeatures2d::SIFT> sift=xfeatures2d::SIFT::create(0,3,0.04,10,1.6);
 Ptr<xfeatures2d::SURF> sift=xfeatures2d::SURF::create(minhessian);

 int trigger=0;
 FlannBasedMatcher matcher; 
 vector<DMatch> matches;
 vector<DMatch> good_matches;
 double min,max;

 //KLT variables.
 vector<Point2f> corners[2];
 vector<Point> hull(1);
 vector<uchar> status;
 vector<float> err;
 TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);

while(trigger==0)
{   
    stereo_calibrate();
	frame=left_rect.clone();
    frame_gray=frame;
	Size image_size=frame.size();
	//cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //Detect Faces using Haar Classifiers.
    cascade_face.detectMultiScale( frame_gray, faces, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
    trigger++;

    corner_prev[0]=Point2f( faces[i].x, faces[i].y);
    corner_prev[1]=Point2f( faces[i].x + faces[i].width, faces[i].y );
    corner_prev[2]=Point2f( faces[i].x, faces[i].height + faces[i].y );
    corner_prev[3]=Point2f( faces[i].x + faces[i].width, faces[i].y + faces[i].height );
    
    rectangle( frame, corner_prev[0], corner_prev[3], Scalar( 255, 0, 0 ), 2, 8, 0 );
    //Face image cropping/masking. 
    int corner_x=faces[i].x+faces[i].width/8;
    int corner_y=faces[i].y+faces[i].height/8;
    int width=5*faces[i].width/7;
    int height=6*faces[i].height/9;

    mask= cv::Mat::zeros(480,640 , CV_8UC1);
    mask_image= cv::Mat::zeros(480,640 , CV_8UC3);
    mask(Rect(corner_x,corner_y,width,height)) = 255;
    frame.copyTo( mask_image, mask);

   
    }
}

//frame=imread("book_2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//sift->detect(frame,keypoints);
//sift->compute(frame,keypoints,descriptors);
//Calculate features using Shi-Tomasi corner detector.
goodFeaturesToTrack(frame_gray,corners[0],40,0.01,10,mask,3,false,0.04);
cornerSubPix(frame_gray, corners[0], Size(10,10), Size(-1,-1), termcrit);

while(true)
{
    stereo_calibrate();
    
    frame_1=left_rect.clone();
    frame_gray=frame_1.clone();
    //cvtColor( frame_1, frame_gray, COLOR_BGR2GRAY );
    
  
    
    min=400.0;
    max=0.0;


    //KLT Tracker implementation.
    if(!frame_prev.empty())
    {
      calcOpticalFlowPyrLK(frame_prev,frame_gray,corners[0],corners[1],status,err,Size(41,41),4, termcrit,0, 0.001);
    }
    cout<<"No. of tracked points"<<corners[1].size()<<endl;
    char str[35];
    sprintf(str,"No. of Tracked points: %lu",corners[1].size());
    putText(disp8,str,Point(50,50),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255),1,8,false);
    
    
    float avg_error=0,avg_opt=0;      
    

    if(!frame_prev.empty())
    {
    //Find Homography for bounding box  determination.
    Mat homography=findHomography(corners[0],corners[1],CV_RANSAC);
    perspectiveTransform(corner_prev,corner_new,homography);
    corner_prev.swap(corner_new);
    corner_new.clear();
    corners[0].clear();
    for (int i=0;i<corners[1].size();i++)
    {
       avg_opt+=err[i];
    }

    //Find the average error in position.
    avg_error=avg_opt/corners[1].size();
    cout<<avg_error<<endl;

    //Plot the tracked points.
    for (int i=0;i<corners[1].size();i++)
    {

        if((err[i]>3*avg_error))
        {   
            circle(disp8,corners[1][i],4,Scalar(0,0,255),-1,8,0);
            continue;
        }
        //cout<<err[i]<<endl;
        circle(disp8,corners[1][i],4,Scalar(0,255,0),-1,8,0);
        corners[0].push_back(corners[1][i]);
    }
    
    //Call face_mask if features fall below a limit.
    if (corners[0].size()<16)
    face_mask(frame_1,corner_prev,corners[0]);

    line(disp8,corner_new[0],corner_new[1],Scalar(0,255,0),4);
    line(disp8,corner_new[1],corner_new[3],Scalar(0,255,0),4);
    line(disp8,corner_new[0],corner_new[2],Scalar(0,255,0),4);
    line(disp8,corner_new[2],corner_new[3],Scalar(0,255,0),4);

    //display delaunaytriangulation.
    Size size = frame_1.size();
    Rect rect(0, 0, size.width, size.height);
    Subdiv2D subdiv(rect);
    
    // Insert points into subdiv
    for( vector<Point2f>::iterator it = corners[1].begin(); it != corners[1].end(); it++)
    {
        subdiv.insert(*it);
    }
    draw_delaunay( disp8, subdiv);

    imshow("KLT Tracker",disp8);
    imshow("Left Image",left_rect);
    //corners[0].swap(corners[1]);        
    corners[1].clear();
    }

    frame_prev=frame_gray.clone();
    
    

    if(waitKey(1)==27)
    {
    	std::cout << "Esc key is pressed by user" << std::endl;
    	break;
    }
}


}

int main(int argc, char** argv)
{   
    stereo_params();
	face_detect();
}