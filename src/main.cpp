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
#include <getopt.h>

#include <morevac_corner.h>
#include <harris_corner.h>
#include <lk_flow.h>
    
using namespace cv;
using namespace std;

// mat to vectors
std::vector<std::vector<float>> mat2vec(Mat img)
{
    // printf("\nMat size: %d, %d", img.cols, img.rows);
    std::vector<std::vector<float>> out;
    for (int y = 0; y < img.rows; y++)
    {
        std::vector<float> col;
        for (int x = 0; x < img.cols; x++)
        {
            float val = (float)((int)img.at<uchar>(y, x));
            if (val < 0) {
                printf("\nNegative value: %f", val);
            }
            col.push_back(val);
        }
        out.push_back(col);
    }
    // printf("\nVector size: %d, %d", out.size(), out[0].size());
    //print part of the vector
    return out;
}


int main(int argc, char *argv[]) {

    int num_threads = 1;
    int opt;
    while ((opt = getopt(argc, argv, "f:n:p:i:m:b:")) != -1) {
        switch (opt) {
            break;
        case 'n':
            num_threads = atoi(optarg);
            break;
        default:
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Number of threads: " << num_threads << "-----------------------------------------------------------------------\n";

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    int num_frames = 15;

    // corner finding params
    int xarea = 5, yarea = 5, thres = 2000;

    // LK params
    int lk_xarea = 50, lk_yarea = 50;

    int sumCornerTime = 0;
    int sumLKTime = 0;
    int sumTotalTime = 0;
    int sumLK2Time = 0;

    auto previmg = Mat();
    auto curimg = Mat();

    //    Step 2 - Implementing Lucas-Kanade Tracker
    for (int i = 30; i < 30 + num_frames; i++)
    {
        char buffer[50];

        const auto begin = chrono::steady_clock::now();

        sprintf(buffer, "../data/race-%03d.png", i);
        auto src = imread(buffer);
        cvtColor(src, previmg, COLOR_BGR2GRAY);
        std::vector<std::vector<float>> previmgvec = mat2vec(previmg);

        sprintf(buffer, "../data/race-%03d.png", (i + 1));
        src = imread(buffer);
        cvtColor(src, curimg, COLOR_BGR2GRAY);
        std::vector<std::vector<float>> curimgvec = mat2vec(curimg);

        const auto endinit = chrono::steady_clock::now();

        // Compute Time :D

        // deque<Point> corn = MorevacCorner::findCorners(previmg, xarea, yarea, thres, false, false);
        vector<pair<int, int>> corn_vec = MorevacCorner::findCorners(previmgvec, xarea, yarea, thres, false);
        const auto endcorner = chrono::steady_clock::now();

        // sort corners by y axis and x axis
        // sort(corn_vec.begin(), corn_vec.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        //     if (a.first == b.first) {
        //         return a.second < b.second;
        //     }
        //     return a.first < b.first;
        // });
        // Lk_Flow::CalcLkFlow(previmg, curimg, lk_xarea, lk_yarea, xarea, yarea, corn, false);
        auto lk_flow_res = Lk_Flow::CalcLkFlow(previmgvec, curimgvec, lk_xarea, lk_yarea, xarea, yarea, corn_vec, false);
        const auto endlk = chrono::steady_clock::now();

        // printf("\nInit took %ld ms, Corner detection took %ld ms, Lucas-Kanade took %ld ms", chrono::duration_cast<chrono::milliseconds>(endinit - begin).count(), chrono::duration_cast<chrono::milliseconds>(endcorner - endinit).count(), chrono::duration_cast<chrono::milliseconds>(endlk - endcorner).count());
        
        vector<pair<int, int>> corn_vec2 = HarrisCorner::findCorners(previmgvec, xarea, yarea, 1000000000, false);
        const auto endcorner2 = chrono::steady_clock::now();

        printf("\n>>>Found %d corners", corn_vec2.size());

        const auto lk_flow_res2 = Lk_Flow::CalcLkFlow(previmgvec, curimgvec, lk_xarea, lk_yarea, xarea, yarea, corn_vec2, false);
        const auto endlk2 = chrono::steady_clock::now();

        // visual stuff
        std::vector<Point> corn;
        for (int i = 0; i < corn_vec2.size(); i++)
        {
            corn.push_back(Point(corn_vec2[i].first, corn_vec2[i].second));
        }

        Mat visual_corners = previmg.clone();
        for (int i = 0; i < corn.size(); i++)
        {
            circle(visual_corners, corn[i], 100, Scalar(0, 255, 0), 2);
        }

        Mat visual_lk = previmg.clone();
        for (int i = 0; i < lk_flow_res2.size(); i++)
        {   
            // if (i < 100) {
            //     printf("\nCorner: (%d, %d) - Flow: (%f, %f)", corn[i].x, corn[i].y, lk_flow_res[i].first, lk_flow_res[i].second);
            // }
            circle(visual_lk, corn[i], 3, Scalar(0, 255, 0), 2);
            line(visual_lk, corn[i], Point(corn[i].x + lk_flow_res2[i].first*10, corn[i].y + lk_flow_res2[i].second*10), Scalar(0, 255, 0), 1);
        }

        namedWindow("lk", WINDOW_NORMAL);
        setWindowProperty("lk", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        imshow("lk", visual_lk);

        waitKey(0);

        const auto endvisual = chrono::steady_clock::now();
        printf("\n>>>Found %d corners", corn.size());
        printf("\n>>>Init took %ld ms, Corner detection took %ld ms, Lucas-Kanade took %ld ms", chrono::duration_cast<chrono::milliseconds>(endinit - begin).count(), chrono::duration_cast<chrono::milliseconds>(endcorner - endinit).count(), chrono::duration_cast<chrono::milliseconds>(endlk - endcorner).count());
        sumCornerTime += chrono::duration_cast<chrono::milliseconds>(endcorner - endinit).count();
        sumLKTime += chrono::duration_cast<chrono::milliseconds>(endlk - endcorner).count();
        sumTotalTime += chrono::duration_cast<chrono::milliseconds>(endlk - begin).count();
    }

    printf("\nAverage Corner Detection Time: %f", sumCornerTime / ((float)num_frames));
    printf("\nAverage Lucas-Kanade Time: %f", sumLKTime / ((float)num_frames));
    printf("\nAverage Total Time: %f", sumTotalTime / ((float)num_frames));
}