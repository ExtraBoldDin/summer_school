#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <random>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    VideoCapture cap(0);
    Mat frame;
    Mat grayscale;
    
    if(!cap.isOpened()){return -1;}
    
    while(1){
        cap.read(frame);
        vector<Vec3f> circles;
        cvtColor(frame, grayscale, CV_BGR2GRAY);
        GaussianBlur(grayscale, grayscale, Size(9, 9), 2, 2);
        HoughCircles(grayscale, circles, CV_HOUGH_GRADIENT, 1, grayscale.rows/8, 200, 100, 0, 0);
        for(int i = 0; i < circles.size(); i++){
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            circle(frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
            putText(frame, "radius = " + to_string(radius), center, 2, 0.4, Scalar(0, 0, 255));
        }
        waitKey(27);
        imshow("result", frame);
    }
    return 0;
}

