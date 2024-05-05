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

// hide the local functions in an anon namespace

HarrisCorner::HarrisCorner() {}

// Function to calculate gradients using Sobel operators
inline void calculateGradients(const vector<vector<float>>& img, int x, int y, int xarea, int yarea, float& dx2, float& dy2, float& dxy) {
    float dx, dy;
    dx2 = dy2 = dxy = 0.0f;


    int dimx = img[0].size(), dimy = img.size();

    for (int j = -yarea; j <= yarea; ++j) {
        for (int i = -xarea; i <= xarea; ++i) {
            if (y + j > 0 && y + j < dimy - 1 && x + i > 0 && x + i < dimx - 1) {
                dx = img[y + j][x + i + 1] - img[y + j][x + i - 1];
                dy = img[y + j + 1][x + i] - img[y + j - 1][x + i];
                dx2 += dx * dx;
                dy2 += dy * dy;
                dxy += dx * dy;
            }
        }
    }
}

/**
 * Finds corners in an image using the Harris corner detection algorithm.
 *
 * @param img The input image represented as a 2D vector of floats.
 * @param xarea The size of the window in the x-direction for corner detection.
 * @param yarea The size of the window in the y-direction for corner detection.
 * @param thres The threshold value for corner detection.
 * @param verbose Flag indicating whether to print verbose output.
 * @return A vector of pairs representing the coordinates of the detected corners.
 */
std::vector<std::pair<int, int>> HarrisCorner::findCorners(std::vector<std::vector<float>>& img, int xarea, int yarea, int thres, bool verbose){
    const float k = 0.04f; // Harris constant

    std::vector<std::pair<int, int>> corners;


    int dimx = img[0].size(), dimy = img.size();


    #pragma omp parallel for schedule(static, 16)
    for (int y = yarea; y < dimy - yarea; ++y) {
        for (int x = xarea; x < dimx - xarea; ++x) {
            float dx2, dy2, dxy;
			
            calculateGradients(img, x, y, xarea, yarea, dx2, dy2, dxy);

            float det = dx2 * dy2 - dxy * dxy;
            float trace = dx2 + dy2;
            float response = det - k * trace * trace;

            if (response > thres) {
                #pragma omp critical
                    corners.push_back(std::make_pair(x, y));
            }
        }
    }
	printf("I am here\n");
	printf("Corners size: %d\n", corners.size());

    return corners;
}