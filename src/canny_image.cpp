#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>


using namespace cv;

// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 5;
int kernel_size = 3;
char* window_name = "Edge Map";


void CannyThreshold()
{
  if(kernel_size%2==0)
    kernel_size++;

  // Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  // Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  imshow( window_name, detected_edges);
 }


int main( int argc, char** argv )
{
  /// Load an image
  VideoCapture cap(0);

  if (!cap.isOpened())  // if not success, exit program
    {
        std::cout << "Cannot open the video cam" << std::endl;
        return -1;
    }

   cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
   cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

   double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
   double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

   std::cout << "Frame size : " << dWidth << " x " << dHeight << std::endl;

   while(1)
   {
    
    bool bSuccess = cap.read(src); // read a new frame from video
    // Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );
    // Convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );
    // Create a window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );
    // Create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold );
    createTrackbar( "Kernel size:", window_name, &kernel_size, 7 );
    // Show the image
    CannyThreshold();


     if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
       {
            std::cout << "Esc key is pressed by user" << std::endl;
            break; 
       }
    }
  return 0;
}