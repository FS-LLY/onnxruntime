#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
int main()
{
    std::string image_path = "/dataset/cifar-10-batches-py/test/5_1000.jpg";
    Mat img = imread(image_path, IMREAD_COLOR);

    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    return 0;
}