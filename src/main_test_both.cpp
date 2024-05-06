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
#include <both_lk.h>
    
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
    char corner_detector = 'm';
    while ((opt = getopt(argc, argv, "n:k:")) != -1) {
        switch (opt) {
            break;
        case 'n':
            num_threads = atoi(optarg);
            break;
        case 'k':
            corner_detector = optarg[0];
            break;
        default:
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Number of threads: " << num_threads << "-----------------------------------------------------------------------\n";
    std::cout << "Corner Detector: " << corner_detector << "-----------------------------------------------------------------------\n";
    
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    int num_frames = 15;

    // corner finding params
    int xarea = 5, yarea = 5, thres_m = 2000, thres_h = 1000000000;

    // LK params
    int lk_xarea = 50, lk_yarea = 50;

    int sumCornerTime = 0;
    int sumLKTime = 0;
    int sumTotalTime = 0;
    int sumLK2Time = 0;

    auto previmg = Mat();
    auto curimg = Mat();

    char buffer1[50];
    sprintf(buffer1, "../data/race-%03d.png", 1);
    auto src_cur = imread(buffer1);
    auto src_prev = imread(buffer1);

    cvtColor(src_cur, curimg, COLOR_BGR2GRAY);
    cvtColor(src_prev, previmg, COLOR_BGR2GRAY);
    std::vector<std::vector<float>> curimgvec = mat2vec(previmg);
    std::vector<std::vector<float>> previmgvec;

    //    Step 2 - Implementing Lucas-Kanade Tracker
    for (int i = 1; i < 1 + num_frames; i++)
    {
        char buffer[50];

        const auto begin = chrono::steady_clock::now();

        src_prev = src_cur.clone();
        previmgvec = curimgvec;

        sprintf(buffer, "../data/race-%03d.png", (i + 1));
        src_cur = imread(buffer);
        cvtColor(src_cur, curimg, COLOR_BGR2GRAY);
        curimgvec = mat2vec(curimg);

        const auto endinit = chrono::steady_clock::now();

        // Compute Time :D
        
        auto lk_res = std::vector<std::pair<std::pair<int, int>, std::pair<float, float>>>();

        if (corner_detector == 'm') {
            lk_res = BothLk::CalcMoreLkFlow(previmgvec, curimgvec, lk_xarea, lk_yarea, xarea, yarea, xarea, yarea, thres_m, false);
        } else if (corner_detector == 'h') {
            lk_res = BothLk::CalcHarrisLkFlow(previmgvec, curimgvec, lk_xarea, lk_yarea, xarea, yarea, xarea, yarea, thres_h, false);
        }

        const auto endlk = chrono::steady_clock::now();
        // visual stuff

        Mat visual_lk = src_prev.clone();
        for (int i = 0; i < lk_res.size(); i++)
        {   
            auto corn = cv::Point(lk_res[i].first.first, lk_res[i].first.second);
            auto vel = lk_res[i].second;
            circle(visual_lk, corn, 2, Scalar(0, 255, 0, 0.3), 1);
            line(visual_lk, corn, Point(corn.x + vel.first*5, corn.y + vel.second*5), Scalar(0, 255, 255), 1);
        }

        // namedWindow("lk", WINDOW_NORMAL);
        // setWindowProperty("lk", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        // imshow("lk", visual_lk);
        // waitKey(0);

        const auto endvisual = chrono::steady_clock::now();
        printf("\n>>>Found %d corners", lk_res.size());
        printf("\n>>>Lucas-Kanade time: %ld ms", chrono::duration_cast<chrono::milliseconds>(endlk - endinit).count());

        sumLKTime += chrono::duration_cast<chrono::milliseconds>(endlk - endinit).count();
        sumTotalTime += chrono::duration_cast<chrono::milliseconds>(endlk - begin).count();
    }

    // printf("\nAverage Corner Detection Time: %f", sumCornerTime / ((float)num_frames));
    printf("\nAverage Lucas-Kanade Time: %f", sumLKTime / ((float)num_frames));
    printf("\nAverage Total Time: %f", sumTotalTime / ((float)num_frames));
}