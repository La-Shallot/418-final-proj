#ifndef _HLK_
#define _HLK_

#include <vector>

class BothLk {
public:
    BothLk();
    static std::vector<std::pair<std::pair<int, int>, std::pair<float, float>>>  CalcHarrisLkFlow(std::vector<std::vector<float>>& imgA, std::vector<std::vector<float>>& imgB, int lk_xsarea, int lk_ysarea, int lk_xarea, int lk_yarea, int h_xarea, int h_yarea, int h_thres, bool verbose);
    static std::vector<std::pair<std::pair<int, int>, std::pair<float, float>>>  CalcMoreLkFlow(std::vector<std::vector<float>>& imgA, std::vector<std::vector<float>>& imgB, int lk_xsarea, int lk_ysarea, int lk_xarea, int lk_yarea, int h_xarea, int h_yarea, int h_thres, bool verbose);
};

#endif