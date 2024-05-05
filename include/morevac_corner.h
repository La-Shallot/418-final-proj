#ifndef _MCORNER_
#define _MCORNER_

#include <deque>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class MorevacCorner {
public:
    MorevacCorner();
    static std::deque<cv::Point> findCorners(cv::Mat img, int xarea, int yarea, int thres, bool verbose, bool visual);
    static std::vector<std::pair<float, float>> findCorners(std::vector<std::vector<float>>& img, int xarea, int yarea, int thres, bool verbose);
};

#endif