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
#include <lk_flow.h>
    
using namespace cv;
using namespace std;

// mat to vectors
std::vector<std::vector<float>> mat2vec(Mat img)
{
    std::vector<std::vector<float>> out;
    for (int i = 0; i < img.cols; i++)
    {
        std::vector<float> col;
        for (int j = 0; j < img.rows; j++)
        {
            float val = ((int)img.at<uchar>(j, i))/255.0;
            col.push_back(val);
        }
        out.push_back(col);
    }
    return out;
}


int main(int argc, char *argv[]) {

    int num_threads = 0;
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

    std::cout << "Number of threads: " << num_threads << '\n';\

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    int num_frames = 15;

    // corner finding params
    int xarea = 3, yarea = 3, thres = 1;

    // LK params
    int lk_xarea = 40, lk_yarea = 40;

    int sumCornerTime = 0;
    int sumLKTime = 0;
    int sumTotalTime = 0;
    int sumLK2Time = 0;

    auto previmg = Mat();
    auto curimg = Mat();

    //    Step 2 - Implementing Lucas-Kanade Tracker
    for (int i = 1; i < num_frames; i++)
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

        deque<Point> corn = MorevacCorner::findCorners(previmg, xarea, yarea, thres, false, false);
        const auto endcorner = chrono::steady_clock::now();

        vector<pair<float, float>> corn_vec_not_used = MorevacCorner::findCorners(previmgvec, xarea, yarea, thres, false);

        const auto endcorner2 = chrono::steady_clock::now();

        Lk_Flow::CalcLkFlow(previmg, curimg, lk_xarea, lk_yarea, xarea, yarea, corn, false);
        const auto endlk = chrono::steady_clock::now();

        printf("\nInit took %d ms, Corner detection took %d ms, Lucas-Kanade took %d ms", chrono::duration_cast<chrono::milliseconds>(endinit - begin).count(), chrono::duration_cast<chrono::milliseconds>(endcorner - endinit).count(), chrono::duration_cast<chrono::milliseconds>(endlk - endcorner).count());

        // transform vector point to vec pair
        std::vector<std::pair<float, float>> corn_vec(corn.size());
        for (int i = 0; i < corn.size(); i++)
        {
            corn_vec[i] = std::make_pair(corn[i].x, corn[i].y);
        }
        Lk_Flow::CalcLkFlow(previmgvec, curimgvec, lk_xarea, lk_yarea, xarea, yarea, corn_vec, false);

        const auto endlk2 = chrono::steady_clock::now();

        // visual stuff

        const auto endvisual = chrono::steady_clock::now();

        printf("\nInit took %d ms, Corner detection took %d ms, Lucas-Kanade took %d ms", chrono::duration_cast<chrono::milliseconds>(endinit - begin).count(), chrono::duration_cast<chrono::milliseconds>(endcorner - endinit).count(), chrono::duration_cast<chrono::milliseconds>(endlk - endcorner2).count());
        printf("\nLkFlow2 took %d ms, corner2 took %d ms", chrono::duration_cast<chrono::milliseconds>(endlk2 - endlk).count(), chrono::duration_cast<chrono::milliseconds>(endcorner2 - endcorner).count());
        sumCornerTime += chrono::duration_cast<chrono::milliseconds>(endcorner2 - endinit).count();
        sumLKTime += chrono::duration_cast<chrono::milliseconds>(endlk - endcorner2).count();
        sumTotalTime += chrono::duration_cast<chrono::milliseconds>(endlk - begin).count();
        sumLK2Time += chrono::duration_cast<chrono::milliseconds>(endlk2 - endlk).count();

    }

    printf("\nAverage Corner Detection Time: %f", sumCornerTime / ((float)num_frames));
    printf("\nAverage Lucas-Kanade Time: %f", sumLKTime / ((float)num_frames));
    printf("\nAverage Total Time: %f", sumTotalTime / ((float)num_frames));
    printf("\nAverage Lucas-Kanade 2 Time: %f", sumLK2Time / ((float)num_frames));
}