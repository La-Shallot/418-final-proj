#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <deque>
#include <omp.h>

using namespace cv;
using namespace std;

#include <harris_corner.h>

HarrisCorner::HarrisCorner() {}

std::vector<std::pair<int, int>> HarrisCorner::findCorners(std::vector<std::vector<float>> &img, int xarea, int yarea, int thres, bool verbose)
{
    const float k = 0.04f; // Harris constant

    std::vector<std::pair<int, int>> corners;

    int dimx = img[0].size(), dimy = img.size();

    #pragma omp parallel for schedule(static, 16)
    for (int y = yarea; y < dimy - yarea; ++y)
    {
        for (int x = xarea; x < dimx - xarea; ++x)
        {
            float dx2 = 0.0f, dy2 = 0.0f, dxy = 0.0f;
            float dx = 0.0f, dy = 0.0f;

            for (int j = -yarea; j <= yarea; ++j)
            {
                for (int i = -xarea; i <= xarea; ++i)
                {
                    if (y + j > 0 && y + j < dimy - 1 && x + i > 0 && x + i < dimx - 1)
                    {
                        dx = img[y + j][x + i + 1] - img[y + j][x + i - 1];
                        dy = img[y + j + 1][x + i] - img[y + j - 1][x + i];
                        dx2 += dx * dx;
                        dy2 += dy * dy;
                        dxy += dx * dy;
                    }
                }
            }

            float det = dx2 * dy2 - dxy * dxy;
            float trace = dx2 + dy2;
            float response = det - k * trace * trace;

            if (response > thres)
            {
                #pragma omp critical
                corners.push_back(std::make_pair(x, y));
            }
        }
    }

    return corners;
}