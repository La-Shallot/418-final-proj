#include "lk_flow.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


Lk_Flow::Lk_Flow() {}

cv::Mat Lk_Flow::CalcLkFlow(cv::Mat imgA, cv::Mat imgB, int xsarea, int ysarea, int xarea, int yarea, std::deque<cv::Point> corners, bool verbose)
    {
        // start timer
        
        cv::Mat outimg = imgB.clone();

        int dimx = imgA.cols, dimy = imgA.rows;
        printf("\nDimx: %d, Dimy: %d", dimx, dimy);

        // iterate through each corner to find flow vectors

        #pragma omp parallel for schedule(static, 32)
        for (int i = 0; i < corners.size(); i++)
        {
            cv::Point cur_corner = corners[i];
            cv::Point corner_mid = cv::Point(corners[i].x + (int)(xarea / 2), cur_corner.y + (int)(yarea / 2));

            // Draw the corner in the out-image
            #pragma omp critical
            cv::rectangle(outimg, cur_corner, cur_corner + cv::Point(xarea, yarea), cv::Scalar(0), 2);

            // Set range parameters
            cv::Range sry = cv::Range(std::max(0, corner_mid.y - (int)(ysarea / 2)), std::min(dimy-1, corner_mid.y + (int)(ysarea / 2)));
            cv::Range srx = cv::Range(std::max(0, corner_mid.x - (int)(xsarea / 2)), std::min(dimx-1, corner_mid.x + (int)(xsarea / 2)));

            // Now that we've found search windows, we can proceed to calculate Ix and Iy for each pixel in the search window
            // Ix

            int range = 1;
            // gradient cv::Matrix
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

                    // if (i == 0) {
                    //     printf("\nLLLLLLLCorner: (%d, %d)---------------------------------------------------------------", x, y);
                    //     printf("\nLLLLLLLCorner: (%d, %d) - Ix: %f, Iy: %f", cur_corner.x, cur_corner.y, curIx, curIy);
                    //     printf("\nLLLLLLLCorner: (%d, %d) - x: %d, y: %d - px: %d, nx: %d, py: %d, ny: %d", cur_corner.x, cur_corner.y, x, y, px, nx, py, ny);
                    //     printf("\nLLLLLLLCorner: (%d, %d) - imgA[y][px]: %ud, imgA[y][nx]: %ud, imgA[py][x]: %ud, imgA[ny][x]: %ud", cur_corner.x, cur_corner.y, imgA.at<uchar>(y, px), imgA.at<uchar>(y, nx), imgA.at<uchar>(py, x), imgA.at<uchar>(ny, x));
                    // }

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
        

        // visualization: draw the flow vectors

        // string winCorImg = "Corners found";
        // namedWindow(winCorImg, WINDOW_NORMAL);
        // imshow(winCorImg, outimg);
        // waitKey(0);

        //if (filename != NULL)
        //{
        ///    cout << string("\nWriting lkoutput to o") << string(filename);
        //    imwrite(string("o") + string(filename), outimg);
        //}
        
        return outimg;
    }

std::vector<std::pair<float, float>> Lk_Flow::CalcLkFlow(std::vector<std::vector<float>>& imgA, std::vector<std::vector<float>>& imgB, int xsarea, int ysarea, int xarea, int yarea, std::vector<std::pair<int, int>> corners, bool verbose)
    {
        // start timer
        
        // cv::Mat outimg = imgB.clone();

        // int dimx = imgA.cols, dimy = imgA.rows;
        int dimx = imgA[0].size(), dimy = imgA.size();

        std::vector<std::pair<float, float>> out(corners.size());

        // iterate through each corner to find flow vectors

        #pragma omp parallel for schedule(static, 32)
        for (int i = 0; i < corners.size(); i++)
        {
            auto cur_corner = corners[i];
            auto corner_mid = std::make_pair((int)(cur_corner.first + (int)(xarea / 2)), (int)(cur_corner.second + (int)(yarea / 2)));

            // Draw the corner in the out-image
            // #pragma omp critical
            // cv::rectangle(outimg, cur_corner, cur_corner + cv::Point(xarea, yarea), cv::Scalar(0), 2);

            // Set range parameters
            // cv::Range sry = cv::Range(std::max(0, corner_mid.y - (int)(ysarea / 2)), std::min(dimy-1, corner_mid.y + (int)(ysarea / 2)));
            // cv::Range srx = cv::Range(std::max(0, corner_mid.x - (int)(xsarea / 2)), std::min(dimx-1, corner_mid.x + (int)(xsarea / 2)));

            auto sry = std::make_pair(std::max(0, corner_mid.second - (int)(ysarea / 2)), std::min(dimy-1, corner_mid.second + (int)(ysarea / 2)));
            auto srx = std::make_pair(std::max(0, corner_mid.first - (int)(xsarea / 2)), std::min(dimx-1, corner_mid.first + (int)(xsarea / 2)));

            // Now that we've found search windows, we can proceed to calculate Ix and Iy for each pixel in the search window
            // Ix

            int range = 1;
            // gradient cv::Matrix
            double G[2][2] = {0, 0, 0, 0};

            // IxIt and IyIt
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
                    // printf("\ncurIx: %f, imgA[y][px]: %f, imgA[y][nx]: %f", curIx, imgA[y][px], imgA[y][nx]); 
                    double curIy = (imgA[py][x] - imgA[ny][x]) / 2;

                    // if (i == 0) {
                    //     printf("\nLLLLLLLCorner: (%d, %d) - Ix: %f, Iy: %f", cur_corner.first, cur_corner.second, curIx, curIy);
                    //     printf("\nLLLLLLLCorner: (%d, %d) - x: %d, y: %d - px: %d, nx: %d, py: %d, ny: %d", cur_corner.first, cur_corner.second, x, y, px, nx, py, ny);
                    //     printf("\nLLLLLLLCorner: (%d, %d) - imgA[y][px]: %f, imgA[y][nx]: %f, imgA[py][x]: %f, imgA[ny][x]: %f", cur_corner.first, cur_corner.second, imgA[y][px], imgA[y][nx], imgA[py][x], imgA[ny][x]);
                    // }

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

            // if (i < 100) {
            //     printf("\nFor the corner (%d,%d) - v = (%f,%f)", cur_corner.first, cur_corner.second, V[0], V[1]);
            // }
            #pragma omp critical
            out[i] = std::make_pair(V[0], V[1]);
            // if (verbose)
                // printf("\nFor the corner (%d,%d) - v = (%f,%f)", cur_corner.x, cur_corner.y, V[0], V[1]);



            // #pragma omp critical
            // cv::line(outimg, corner_mid, corner_mid + cv::Point(V[0] * 10, V[1] * 10), color, 1);
        }
        // visualization: draw the flow vectors

        // string winCorImg = "Corners found";
        // namedWindow(winCorImg, WINDOW_NORMAL);
        // imshow(winCorImg, outimg);
        // waitKey(0);

        //if (filename != NULL)
        //{
        ///    cout << string("\nWriting lkoutput to o") << string(filename);
        //    imwrite(string("o") + string(filename), outimg);
        //}

        #pragma omp barrier
        
        return out;
    }

    // std::vector<std::pair<float, float>> Lk_Flow::CalcLkFlow2(std::vector<std::vector<float>>& imgA, std::vector<std::vector<float>>& imgB, int xsarea, int ysarea, int xarea, int yarea, std::vector<std::pair<int, int>> corners, bool verbose)
    // {
    //     // start timer
        
    //     // cv::Mat outimg = imgB.clone();

    //     // int dimx = imgA.cols, dimy = imgA.rows;
    //     int dimx = imgA[0].size(), dimy = imgA.size();

    //     std::vector<std::pair<float, float>> out(corners.size());

    //     // iterate through each corner to find flow vectors

    //     #pragma omp parallel for schedule(static, 32)
    //     for (int i = 0; i < corners.size(); i++)
    //     {
    //         auto cur_corner = corners[i];
    //         auto corner_mid = std::make_pair((int)(cur_corner.first + (int)(xarea / 2)), (int)(cur_corner.second + (int)(yarea / 2)));

    //         // Draw the corner in the out-image
    //         // #pragma omp critical
    //         // cv::rectangle(outimg, cur_corner, cur_corner + cv::Point(xarea, yarea), cv::Scalar(0), 2);

    //         // Set range parameters
    //         // cv::Range sry = cv::Range(std::max(0, corner_mid.y - (int)(ysarea / 2)), std::min(dimy-1, corner_mid.y + (int)(ysarea / 2)));
    //         // cv::Range srx = cv::Range(std::max(0, corner_mid.x - (int)(xsarea / 2)), std::min(dimx-1, corner_mid.x + (int)(xsarea / 2)));

    //         auto sry = std::make_pair(std::max(0, corner_mid.second - (int)(ysarea / 2)), std::min(dimy-1, corner_mid.second + (int)(ysarea / 2)));
    //         auto srx = std::make_pair(std::max(0, corner_mid.first - (int)(xsarea / 2)), std::min(dimx-1, corner_mid.first + (int)(xsarea / 2)));

    //         // Now that we've found search windows, we can proceed to calculate Ix and Iy for each pixel in the search window
    //         // Ix

    //         int range = 1;
    //         // gradient cv::Matrix
    //         double G[2][2] = {0, 0, 0, 0};

    //         // IxIt and IyIt
    //         double b[2] = {0, 0};

    //         // This is a convolution?

    //         double G_00 = 0, G_01 = 0, G_10 = 0, G_11 = 0;
    //         double b_0 = 0, b_1 = 0;

    //         int sr_total = (srx.second - srx.first + 1) * (sry.second - sry.first + 1);
    //         for (int xy = 0; xy < sr_total; xy++)
    //         {
    //             int x = srx.first + xy % (srx.second - srx.first + 1);
    //             int y = sry.first + xy / (srx.second - srx.first + 1);

    //             int px = x - range, nx = x + range;
    //             int py = y - range, ny = y + range;

    //             // edge conditions
    //             if (x == 0)
    //                 px = x;
    //             if (x >= (dimx - 1))
    //                 nx = x;
    //             if (y == 0)
    //                 py = 0;
    //             if (y >= (dimy - 1))
    //                 ny = y;

    //             double curIx = (imgA[y][px] - imgA[y][nx]) / 2;
    //             double curIy = (imgA[py][x] - imgA[ny][x]) / 2;

    //             // calculate G and b
    //             G_00 += curIx * curIx;
    //             G_01 += curIx * curIy;
    //             G_10 += curIx * curIy;
    //             G_11 += curIy * curIy;

    //             double curdI = (imgA[y][x] - imgB[y][x]);

    //             b_0 += curdI * curIx;
    //             b_1 += curdI * curIy;
    //         }
    //         // for (int x = srx.first; x <= srx.second; x++)
    //         //     for (int y = sry.first; y <= sry.second; y++)
    //         //     {
    //         //         int px = x - range, nx = x + range;
    //         //         int py = y - range, ny = y + range;

    //         //         // edge conditions
    //         //         if (x == 0)
    //         //             px = x;
    //         //         if (x >= (dimx - 1))
    //         //             nx = x;
    //         //         if (y == 0)
    //         //             py = 0;
    //         //         if (y >= (dimy - 1))
    //         //             ny = y;

    //         //         double curIx = (imgA[y][px] - imgA[y][nx]) / 2;
    //         //         // printf("\ncurIx: %f, imgA[y][px]: %f, imgA[y][nx]: %f", curIx, imgA[y][px], imgA[y][nx]); 
    //         //         double curIy = (imgA[py][x] - imgA[ny][x]) / 2;

    //         //         // if (i == 0) {
    //         //         //     printf("\nLLLLLLLCorner: (%d, %d) - Ix: %f, Iy: %f", cur_corner.first, cur_corner.second, curIx, curIy);
    //         //         //     printf("\nLLLLLLLCorner: (%d, %d) - x: %d, y: %d - px: %d, nx: %d, py: %d, ny: %d", cur_corner.first, cur_corner.second, x, y, px, nx, py, ny);
    //         //         //     printf("\nLLLLLLLCorner: (%d, %d) - imgA[y][px]: %f, imgA[y][nx]: %f, imgA[py][x]: %f, imgA[ny][x]: %f", cur_corner.first, cur_corner.second, imgA[y][px], imgA[y][nx], imgA[py][x], imgA[ny][x]);
    //         //         // }

    //         //         // calculate G and b
    //         //         G[0][0] += curIx * curIx;
    //         //         G[0][1] += curIx * curIy;
    //         //         G[1][0] += curIx * curIy;
    //         //         G[1][1] += curIy * curIy;

    //         //         double curdI = (imgA[y][x] - imgB[y][x]);

    //         //         b[0] += curdI * curIx;
    //         //         b[1] += curdI * curIy;
    //         //     }

    //         double detG = (G[0][0] * G[1][1]) - (G[0][1] * G[1][0]);
    //         double Ginv[2][2] = {0, 0, 0, 0};
    //         Ginv[0][0] = G[1][1] / detG;
    //         Ginv[0][1] = -G[0][1] / detG;
    //         Ginv[1][0] = -G[1][0] / detG;
    //         Ginv[1][1] = G[0][0] / detG;

    //         double V[2] = {Ginv[0][0] * b[0] + Ginv[0][1] * b[1], Ginv[1][0] * b[0] + Ginv[1][1] * b[1]};

    //         // if (i < 100) {
    //         //     printf("\nFor the corner (%d,%d) - v = (%f,%f)", cur_corner.first, cur_corner.second, V[0], V[1]);
    //         // }
    //         #pragma omp critical
    //         out[i] = std::make_pair(V[0], V[1]);
    //         // if (verbose)
    //             // printf("\nFor the corner (%d,%d) - v = (%f,%f)", cur_corner.x, cur_corner.y, V[0], V[1]);



    //         // #pragma omp critical
    //         // cv::line(outimg, corner_mid, corner_mid + cv::Point(V[0] * 10, V[1] * 10), color, 1);
    //     }
    //     // visualization: draw the flow vectors

    //     // string winCorImg = "Corners found";
    //     // namedWindow(winCorImg, WINDOW_NORMAL);
    //     // imshow(winCorImg, outimg);
    //     // waitKey(0);

    //     //if (filename != NULL)
    //     //{
    //     ///    cout << string("\nWriting lkoutput to o") << string(filename);
    //     //    imwrite(string("o") + string(filename), outimg);
    //     //}

    //     #pragma omp barrier
        
    //     return out;
    // }
    