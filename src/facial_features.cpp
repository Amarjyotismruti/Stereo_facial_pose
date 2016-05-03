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
    cvtColor( image, frame_gray, COLOR_BGR2GRAY );
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
	VideoCapture cap(0);
	if(cap.isOpened()==0)
 {
   cout<<"Cannot open video cam."<<endl;
   exit;
 }

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
 Ptr<xfeatures2d::SIFT> sift=xfeatures2d::SIFT::create(0,3,0.04,10,1.6);
 //Ptr<xfeatures2d::SURF> sift=xfeatures2d::SURF::create(minhessian);

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
	cap>>frame;
	Size image_size=frame.size();
	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //Detect Faces using Haar Classifiers.
    cascade_face.detectMultiScale( frame_gray, faces, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
    trigger++;
    cout<<1<<endl;
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

    //Preliminary reference keypoints/corners extraction.
    sift->detect(frame_gray,keypoints,mask);
    sift->compute(frame_gray,keypoints,descriptors);
    //drawKeypoints(frame,keypoints,frame,Scalar(255,0,0));
    //imshow("Face Mask",mask_image);

    }
}

//frame=imread("book_2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//sift->detect(frame,keypoints);
//sift->compute(frame,keypoints,descriptors);
//Calculate features using Shi-Tomasi corner detector.
goodFeaturesToTrack(frame_gray,corners[0],40,0.01,10,mask,3,false,0.04);
cornerSubPix(frame_gray, corners[0], Size(10,10), Size(-1,-1), termcrit);
namedWindow("KLT Tracker",CV_WINDOW_NORMAL);

while(true)
{
    cap>>frame_1;
    cvtColor( frame_1, frame_gray, COLOR_BGR2GRAY );
    good_matches.clear();
    //SIFT feature and descriptors extraction.
    sift->detect(frame_gray,keypoints_1);
    sift->compute(frame_gray,keypoints_1,descriptors_1);
    matcher.match(descriptors,descriptors_1,matches);
    
    min=400.0;
    max=0.0;


    //Calculating the Max and Min distance parameters of the features.
    for( int i = 0; i < descriptors.rows; i++ )
    { 
    double dist = matches[i].distance;
    if( dist < min ) min = dist;
    if( dist > max ) max = dist;
    }
    cout<<"Min"<<min<<" Max"<<max<<endl;



    //Filtering out matches based on feature eucledian distance.
    for( int i = 0; i < descriptors.rows; i++ )
    { 
      if( matches[i].distance < 2*min )
      {
         good_matches.push_back( matches[i]); 
      }
    }
    cout<<"Good Matches:"<<good_matches.size()<<" Matches:"<<matches.size()<<endl;
    

    //KLT Tracker implementation.
    if(!frame_prev.empty())
    {
      calcOpticalFlowPyrLK(frame_prev,frame_gray,corners[0],corners[1],status,err,Size(41,41),4, termcrit,0, 0.001);
    }
    cout<<corners[1].size()<<endl;
    
    //Draw matches and corners.
    drawMatches(frame,keypoints,frame_1,keypoints_1,good_matches,final,Scalar(0,0,255),Scalar(0,0,255), 
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
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
            circle(frame_1,corners[1][i],4,Scalar(0,0,255),-1,8,0);
            continue;
        }
        //cout<<err[i]<<endl;
        circle(frame_1,corners[1][i],4,Scalar(0,255,0),-1,8,0);
        corners[0].push_back(corners[1][i]);
    }
    
    //Call face_mask if features fall below a limit.
    if (corners[0].size()<16)
    face_mask(frame_1,corner_prev,corners[0]);

    line(frame_1,corner_new[0],corner_new[1],Scalar(0,255,0),4);
    line(frame_1,corner_new[1],corner_new[3],Scalar(0,255,0),4);
    line(frame_1,corner_new[0],corner_new[2],Scalar(0,255,0),4);
    line(frame_1,corner_new[2],corner_new[3],Scalar(0,255,0),4);

    //display delaunaytriangulation.
    Size size = frame_1.size();
    Rect rect(0, 0, size.width, size.height);
    Subdiv2D subdiv(rect);
    
    // Insert points into subdiv
    for( vector<Point2f>::iterator it = corners[1].begin(); it != corners[1].end(); it++)
    {
        subdiv.insert(*it);
    }
    draw_delaunay( frame_1, subdiv);

    imshow("KLT Tracker",frame_1);
    //corners[0].swap(corners[1]);        
    corners[1].clear();
    }

    frame_prev=frame_gray.clone();
    imshow("Face Detect.",final);
    

    if(waitKey(1)==27)
    {
    	std::cout << "Esc key is pressed by user" << std::endl;
    	break;
    }
}


}

int main(int argc, char** argv)
{
	face_detect();
}