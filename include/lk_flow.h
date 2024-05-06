#ifndef _LK_FLOW_
#define _LK_FLOW_

#include <opencv2/imgcodecs.hpp>
#include <deque>

class Lk_Flow {
public:
    Lk_Flow();
    static cv::Mat CalcLkFlow(cv::Mat imgA, cv::Mat imgB, int xsarea, int ysarea, int xarea, int yarea, std::deque<cv::Point> corners, bool);
    static std::vector<std::pair<float, float>> CalcLkFlow(std::vector<std::vector<float>>& imgA, std::vector<std::vector<float>>& imgB, int xsarea, int ysarea, int xarea, int yarea, std::vector<std::pair<int, int>> corners, bool verbose);
};

#endif