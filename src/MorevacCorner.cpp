#include <iostream>
#include <vector>
#include <deque>

#include <morevac_corner.h>
#include <opencv2/highgui/highgui.hpp>

MorevacCorner::MorevacCorner() {}

std::deque<cv::Point> MorevacCorner::findCorners(cv::Mat img, int xarea, int yarea, int thres, bool verbose = true, bool visual = false)
{
    std::deque<cv::Point> corners;

    cv::Mat outimg = img.clone(); // This will be used to provide a visual indication of the corners present

    int dimx = img.cols, dimy = img.rows;

    int count = 0;
    #pragma omp parallel for schedule(dynamic, 32)
    for (int starty_dummy = 0; (starty_dummy) < dimy / yarea - 1; starty_dummy += 1)
        for (int startx = 0; (startx + xarea) < dimx; startx += xarea)
        {
            int starty = starty_dummy * yarea;
            count++;
            cv::Mat curarea = img(cv::Range(starty, std::min(starty + yarea, dimy)), cv::Range(startx, std::min(dimx, startx + xarea)));
            double results[2] = {0, 0};
            for (int dir = 0; dir < 4; dir++)
            {
                int newsx = startx, newsy = starty;
                // Check similarity in each direction
                switch (dir)
                {
                case 0: // left
                    newsx -= xarea;
                    newsx = std::max(newsx, 0);
                    break;
                case 1: // top
                    newsy -= yarea;
                    newsy = std::max(newsy, 0);
                    break;
                case 2: // right
                    newsx += xarea;
                    newsx = std::min(newsx, dimx);
                    break;
                case 3: // down
                    newsy += yarea;
                    newsy = std::min(newsy, dimy);
                    break;
                default:
                    break;
                }

                cv::Mat newarea = img(cv::Range(newsy, std::min(newsy + yarea, dimy)), cv::Range(newsx, std::min(newsx + xarea, dimx)));

                if (newarea.cols != curarea.cols || newarea.rows != curarea.rows)
                {
                    continue;
                }
                cv::Mat diff = abs(curarea - newarea);
                results[dir % 2] = cv::mean(cv::mean(diff))(0);
            }
            results[0] /= 2;
            results[1] /= 2;

            // thresholding
            if (results[0] >= thres && results[1] >= thres)
            {
                #pragma omp critical
                corners.push_back(cv::Point(startx, starty));
            }
        }

    return corners;
}

std::vector<std::pair<int, int>> MorevacCorner::findCorners(std::vector<std::vector<float>> &img, int xarea, int yarea, int thres, bool verbose = true)
{
    std::vector<std::pair<int, int>> corners;

    int dimx = img[0].size(), dimy = img.size();

    std::vector<std::pair<int, int>> dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    #pragma omp parallel for schedule(static, 16)
    for (int startx_dummy = 0; (startx_dummy) < dimx / xarea - 1; startx_dummy += 1)
        for (int starty = 0; (starty + yarea) < dimy; starty += yarea)
        {
            int startx = startx_dummy * xarea;

            double results[4] = {0, 0, 0, 0};
            // for all four directions calculate the change

            for (int currdir = 0; currdir < 4; currdir++)
            {
                std::pair<int, int> offset = dir[currdir];
                int newsx = startx + offset.first * xarea;
                int newsy = starty + offset.second * yarea;

                if (newsx < 0 || newsy < 0 || newsx + xarea >= dimx || newsy + yarea >= dimy)
                {
                    continue;
                }

                // get E value
                for (int j = 0; j < yarea; ++j)
                {
                    for (int i = 0; i < xarea; ++i)
                    {
                        float oldPixel = img[starty + j][startx + i];
                        float newPixel = img[newsy + j][newsx + i];
                        float result = newPixel - oldPixel;
                        results[currdir] += result * result;
                    }
                }
            }

            // get the smallest value from the directions
            float minval = std::min(results[0], std::min(results[1], std::min(results[2], results[3])));

            // thresholding for the min value obtained per direction: local maxima means corner :)
            if (minval > thres)
            {
                #pragma omp critical
                corners.push_back(std::make_pair(startx, starty));
            }
        }

    return corners;
}