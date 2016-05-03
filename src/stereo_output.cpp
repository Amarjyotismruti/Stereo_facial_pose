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

int main( int argc, char** argv )
{
 VideoCapture cap_1(1);
 if(cap_1.isOpened()==0)
 {
   cout<<"Cannot open video cam"<<endl;
   exit;
 }
 
VideoCapture cap_2(2);
 if(cap_2.isOpened()==0)
 {
   cout<<"Cannot open video cam"<<endl;
   exit;
 }
 
 namedWindow("Left Image",CV_WINDOW_AUTOSIZE);
 namedWindow("Right Image",CV_WINDOW_AUTOSIZE);

 Mat image_l,image_r;
 
 
 while(1)
 {
  cap_1.read(image_l);
  imshow("Left Image",image_l);
  cap_2.read(image_r);
  imshow("Right Image",image_r);
  if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
  {
  cout << "Esc key is pressed by user." << endl;
  break; 
  }

 }
}