#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <random>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    VideoCapture cap(0);
    Mat frame;
    Mat grayscale;
    
    if(!cap.isOpened()){return -1;}
    
    int x, y, i;
    double scale = 5;
    
    while(1){
        cap.read(frame);
        if(i%2 != 0 and i%3 != 0){
            putText(frame, "Wow-wow!", Point(frame.rows/2 - 200, frame.cols/2), 3, 7, Scalar(0, 0, 255));
        } else{
            putText(frame, "Wow-wow!", Point(frame.rows/2 - 200, frame.cols/2), 2, scale, Scalar(255, 0, 0));
        }
        for(int i = 0; i < 30; i++){
            x = rand()%frame.rows*2;
            y = rand()%frame.cols*2;
            if(x > frame.cols/2){
                circle(frame, Point(x, y), 20, Scalar(0, 255, 0));
            }
            else{
                rectangle(frame, Point(x, y), Point(x+10, y+10), Scalar(30, 10, 200));
            }
            
        }
        i = i + 1;
        waitKey(27);
        imshow("result", frame);
    }
    return 0;
}
