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


// hide the local functions in an anon namespace
namespace
{


deque<Point> findCornersvec(vector<std::vector<float>> img, int xarea, int yarea, int thres, bool verbose = true)
    {
        deque<Point> corners;

        // We're using Morevac Corner Detection here as it is easier to implement.
        // TODO: Consider using Harris Corners
        ofstream log; // This will be used for dumping raw data for corner analysis
        log.open("log.csv");
        log << "x,y,score1,score2\n";

        //vector<std::vector<float>> outimg = floatVectorToMat(img).clone(); // This will be used to provide a visual indication of the corners present

        if (verbose)
            printf("This is still under construction. - areas - (%d,%d), thres - %d", xarea, yarea, thres);

        int dimx = img.cols, dimy = img.rows;
        // omp_lock_t vec_lock;
        // omp_init_lock(&vec_lock);

        int count = 0;
        #pragma omp parallel for schedule(static, 16)
        for (int startx_dummy = 0; (startx_dummy) < dimx/xarea - 1; startx_dummy += 1)
            for (int starty = 0; (starty + yarea) < dimy; starty += yarea)
        // for (int starty_dummy = 0; (starty_dummy) < dimy/yarea - 1; starty_dummy += 1)
            // for (int startx = 0; (startx + xarea) < dimx; startx += xarea)
            {
                int startx = startx_dummy * xarea;
                // int starty = starty_dummy * yarea;
                count++;
                if (verbose){
                    // printf("\n Area %d - Currenty looking at area (%d-%d,%d-%d)\n", count, startx, startx + xarea, starty, starty + yarea);
                }

                double results[2] = {0, 0};
                for (int dir = 0; dir < 4; dir++)
                {
                    int newsx = startx, newsy = starty;
                    // Check similarity in each direction
                    switch (dir)
                    {
                    case 0: // left
                        newsx -= xarea;
                        newsx = max(newsx, 0);
                        break;
                    case 1: // top
                        newsy -= yarea;
                        newsy = max(newsy, 0);
                        break;
                    case 2: // right
                        newsx += xarea;
                        newsx = min(newsx, dimx);
                        break;
                    case 3: // down
                        newsy += yarea;
                        newsy = min(newsy, dimy);
                        break;
                    default:
                        break;
                    }
                    // printf("Current direction is %d. New area is (%d-%d,%d-%d)\n",dir,newsx,min(dimx,newsx+xarea),newsy,min(newsy+yarea,dimy));

                    Range oldyR Range(starty, min(starty + yarea, dimy));
                    Range oldxR Range(startx, min(startx + xarea, dimx));
                    Range newyR Range(newsy, min(newsy + yarea, dimy));
                    Range newxR Range(newsx, min(newsx + xarea, dimx));

                    if (oldxR.size() != newxR.size() || oldyR.size() != newyR.size()) {
                        std::cerr << "Error: Regions have different dimensions." << std::endl;
                        if (verbose){
                            // printf("Skipping due to dimensional or similarity issues");
                        }
                        continue;
                    
                    }

                    float sum=0;
                    int totalelem= oldxR.size() * oldyR.size();
                    for (int i = 0; i < range1.size(); ++i) {
                        for (int j = 0; j < range2.size(); ++j) {
                            float oldPixel = img.at<uchar>(range1.start + i, range2.start + j);
                            float newPixel = img.at<uchar>(range3.start + i, range4.start + j);
                            float result = newPixel - oldPixel;
                            sum += result;
                        }
                    }

                    //get mean
                    double mean = sum / totalelem;

                    results[dir % 2] = mean;
                }

                results[0] /= 2;
                results[1] /= 2;
                if (verbose)
                    // printf("Scores obtained: %f, %f\n", results[0], results[1]);

                // thresholding
                if (results[0] >= thres && results[1] >= thres)
                {   
                    //lock
                    // omp_set_lock(&vec_lock);
                    // printf("Pushing corner %d,%d\n", startx, starty);
                    #pragma omp critical
                    corners.push_back(Point(startx, starty));
                    // #pragma omp critical
                    // rectangle(outimg, Point(startx, starty), Point(startx + xarea, starty + yarea), Scalar(0), 2);
                    // omp_unset_lock(&vec_lock);
                }

                // log << startx << "," << starty << "," << results[0] << "," << results[1] << "\n";
            }
        log.close();

        if (verbose)
        {
            // string winCorImg = "Corners found";
            // namedWindow(winCorImg, WINDOW_AUTOSIZE);
            // imshow(winCorImg, outimg);
            // waitKey(0);
        }

        printf("\nFound %d corners", corners.size());

        // omp_destroy_lock(&vec_lock);

        return corners;
    }
  }
