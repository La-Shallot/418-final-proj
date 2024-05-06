#ifndef _HCORNER_
#define _HCORNER_

#include <deque>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class HarrisCorner {
public:
    HarrisCorner();
    static std::vector<std::pair<int, int>> findCorners(std::vector<std::vector<float>>& img, int xarea, int yarea, int thres, bool verbose);
};

#endif