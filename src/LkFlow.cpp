#include "lk_flow.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

Lk_Flow::Lk_Flow() {}

cv::Mat Lk_Flow::CalcLkFlow(cv::Mat imgA, cv::Mat imgB, int xsarea, int ysarea, int xarea, int yarea, std::deque<cv::Point> corners, bool verbose)
{

    cv::Mat outimg = imgB.clone();

    int dimx = imgA.cols, dimy = imgA.rows;

#pragma omp parallel for schedule(static, 32)
    for (int i = 0; i < corners.size(); i++)
    {
        cv::Point cur_corner = corners[i];
        cv::Point corner_mid = cv::Point(corners[i].x + (int)(xarea / 2), cur_corner.y + (int)(yarea / 2));

// Draw the corner in the out-image
#pragma omp critical
        cv::rectangle(outimg, cur_corner, cur_corner + cv::Point(xarea, yarea), cv::Scalar(0), 2);

        // Set range parameters
        cv::Range sry = cv::Range(std::max(0, corner_mid.y - (int)(ysarea / 2)), std::min(dimy - 1, corner_mid.y + (int)(ysarea / 2)));
        cv::Range srx = cv::Range(std::max(0, corner_mid.x - (int)(xsarea / 2)), std::min(dimx - 1, corner_mid.x + (int)(xsarea / 2)));

        int range = 1;
        double G[2][2] = {0, 0, 0, 0};

        // IxIt and IyIt
        double b[2] = {0, 0};

        // This is a convolution!

        for (int x = srx.start; x <= srx.end; x++)
            for (int y = sry.start; y <= sry.end; y++)
            {
                int px = x - range, nx = x + range;
                int py = y - range, ny = y + range;

                // edge conditions
                if (x == 0)
                    px = x;
                if (x >= (dimx - 1))
                    nx = x;
                if (y == 0)
                    py = 0;
                if (y >= (dimy - 1))
                    ny = y;

                double curIx = ((int)imgA.at<uchar>(y, px) - (int)imgA.at<uchar>(y, nx)) / 2;
                double curIy = ((int)imgA.at<uchar>(py, x) - (int)imgA.at<uchar>(ny, x)) / 2;

                // calculate G and b
                G[0][0] += curIx * curIx;
                G[0][1] += curIx * curIy;
                G[1][0] += curIx * curIy;
                G[1][1] += curIy * curIy;

                double curdI = ((int)imgA.at<uchar>(y, x) - (int)imgB.at<uchar>(y, x));

                b[0] += curdI * curIx;
                b[1] += curdI * curIy;
            }

        double detG = (G[0][0] * G[1][1]) - (G[0][1] * G[1][0]);
        double Ginv[2][2] = {0, 0, 0, 0};
        Ginv[0][0] = G[1][1] / detG;
        Ginv[0][1] = -G[0][1] / detG;
        Ginv[1][0] = -G[1][0] / detG;
        Ginv[1][1] = G[0][0] / detG;

        double V[2] = {Ginv[0][0] * b[0] + Ginv[0][1] * b[1], Ginv[1][0] * b[0] + Ginv[1][1] * b[1]};
        if (verbose or i < 100)
            printf("\nFor the corner (%d,%d) - v = (%f,%f)", cur_corner.x, cur_corner.y, V[0], V[1]);
        cv::Scalar color = (0, 255, 0);

#pragma omp critical
        cv::line(outimg, corner_mid, corner_mid + cv::Point(V[0] * 10, V[1] * 10), color, 1);
    }

    return outimg;
}

std::vector<std::pair<float, float>> Lk_Flow::CalcLkFlow(std::vector<std::vector<float>> &imgA, std::vector<std::vector<float>> &imgB, int xsarea, int ysarea, int xarea, int yarea, std::vector<std::pair<int, int>> corners, bool verbose)
{
    int dimx = imgA[0].size(), dimy = imgA.size();

    std::vector<std::pair<float, float>> out(corners.size());

    // iterate through each corner to find flow vectors

#pragma omp parallel for schedule(static, 32)
    for (int i = 0; i < corners.size(); i++)
    {
        auto cur_corner = corners[i];
        auto corner_mid = std::make_pair((int)(cur_corner.first + (int)(xarea / 2)), (int)(cur_corner.second + (int)(yarea / 2)));

        auto sry = std::make_pair(std::max(0, corner_mid.second - (int)(ysarea / 2)), std::min(dimy - 1, corner_mid.second + (int)(ysarea / 2)));
        auto srx = std::make_pair(std::max(0, corner_mid.first - (int)(xsarea / 2)), std::min(dimx - 1, corner_mid.first + (int)(xsarea / 2)));

        int range = 1;
        double G[2][2] = {0, 0, 0, 0};
        double b[2] = {0, 0};

        // This is a convolution?

        for (int x = srx.first; x <= srx.second; x++)
            for (int y = sry.first; y <= sry.second; y++)
            {
                int px = x - range, nx = x + range;
                int py = y - range, ny = y + range;

                // edge conditions
                if (x == 0)
                    px = x;
                if (x >= (dimx - 1))
                    nx = x;
                if (y == 0)
                    py = 0;
                if (y >= (dimy - 1))
                    ny = y;

                double curIx = (imgA[y][px] - imgA[y][nx]) / 2;
                double curIy = (imgA[py][x] - imgA[ny][x]) / 2;

                // calculate G and b
                G[0][0] += curIx * curIx;
                G[0][1] += curIx * curIy;
                G[1][0] += curIx * curIy;
                G[1][1] += curIy * curIy;

                double curdI = (imgA[y][x] - imgB[y][x]);

                b[0] += curdI * curIx;
                b[1] += curdI * curIy;
            }

        double detG = (G[0][0] * G[1][1]) - (G[0][1] * G[1][0]);
        double Ginv[2][2] = {0, 0, 0, 0};
        Ginv[0][0] = G[1][1] / detG;
        Ginv[0][1] = -G[0][1] / detG;
        Ginv[1][0] = -G[1][0] / detG;
        Ginv[1][1] = G[0][0] / detG;

        double V[2] = {Ginv[0][0] * b[0] + Ginv[0][1] * b[1], Ginv[1][0] * b[0] + Ginv[1][1] * b[1]};

#pragma omp critical
        out[i] = std::make_pair(V[0], V[1]);
    }

    return out;
}
