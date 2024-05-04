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

    deque<Point> findCorners(Mat img, int xarea, int yarea, int thres, bool verbose = true)
    {
        deque<Point> corners;

        // We're using Morevac Corner Detection here as it is easier to implement.
        // TODO: Consider using Harris Corners
        ofstream log; // This will be used for dumping raw data for corner analysis
        log.open("log.csv");
        log << "x,y,score1,score2\n";

        Mat outimg = img.clone(); // This will be used to provide a visual indication of the corners present

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
                Mat curarea = img(Range(starty, min(starty + yarea, dimy)), Range(startx, min(dimx, startx + xarea)));
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
                    //                    printf("Current direction is %d. New area is (%d-%d,%d-%d)\n",dir,newsx,min(dimx,newsx+xarea),newsy,min(newsy+yarea,dimy));

                    Mat newarea = img(Range(newsy, min(newsy + yarea, dimy)), Range(newsx, min(newsx + xarea, dimx)));

                    if (newarea.cols != curarea.cols || newarea.rows != curarea.rows)
                    {
                        if (verbose){
                            // printf("Skipping due to dimensional or similarity issues");
                        }
                        continue;
                    }
                    Mat diff = abs(curarea - newarea);
                    results[dir % 2] = mean(mean(diff))(0);
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

    deque<Point> findCorners(vector<vector<float>> img, int xarea, int yarea, int thres, bool verbose = true)
    {
        deque<Point> corners;

        // Using Harris Corner Detection

        if (verbose)
            printf("\nHarris Corner Detection - areas - (%d,%d), thres - %d", xarea, yarea, thres);

        // int dimx = img.cols, dimy = img.rows;
        int dimx = img[0].size(), dimy = img.size();
        // omp_lock_t vec_lock;
        // omp_init_lock(&vec_lock);

        int count = 0;



        printf("\nFound %d corners", corners.size());

        // omp_destroy_lock(&vec_lock);

        return corners;
    }

    Mat lucasKanade(Mat imgA, Mat imgB, int xsarea, int ysarea, int xarea, int yarea, deque<Point> corners, bool verbose = true, char filename[] = NULL)
    {
        // start timer
        


        Mat outimg = imgB.clone();

        int dimx = imgA.cols, dimy = imgA.rows;

        // iterate through each corner to find flow vectors
        printf("\nFound %d corners", corners.size());



        #pragma omp parallel for schedule(static, 16)
        for (int i = 0; i < corners.size(); i++)
        {
            Point cur_corner = corners[i];
            Point corner_mid = Point(corners[i].x + (int)(xarea / 2), cur_corner.y + (int)(yarea / 2));

            // Draw the corner in the out-image
            #pragma omp critical
            rectangle(outimg, cur_corner, cur_corner + Point(xarea, yarea), Scalar(0), 2);

            // Set range parameters
            Range sry = Range(max(0, corner_mid.y - (int)(ysarea / 2)), min(dimy-1, corner_mid.y + (int)(ysarea / 2)));
            Range srx = Range(max(0, corner_mid.x - (int)(xsarea / 2)), min(dimx-1, corner_mid.x + (int)(xsarea / 2)));

            // Now that we've found search windows, we can proceed to calculate Ix and Iy for each pixel in the search window
            // Ix

            int range = 1;
            // gradient matrix
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
            if (verbose)
                printf("\nFor the corner (%d,%d) - v = (%f,%f)", cur_corner.x, cur_corner.y, V[0], V[1]);
            Scalar color = (0, 255, 0); 

            #pragma omp critical
            line(outimg, corner_mid, corner_mid + Point(V[0] * 10, V[1] * 10), color, 1);
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
    // end of namespace
}

// mat to vectors
std::vector<std::vector<float>> mat2vec(Mat img)
{
    std::vector<std::vector<float>> out;
    for (int i = 0; i < img.rows; i++)
    {
        std::vector<float> row;
        for (int j = 0; j < img.cols; j++)
        {
            float val = ((int)img.at<uchar>(i, j))/255.0;
            row.push_back(val);
        }
        out.push_back(row);
    }
    return out;
}

int main(int ac, char **av)
{

    //    Step 1 - Implementing Corner Detection
    //    Read the file
    Mat src_color, img1, img2; // To store the file after conversion
    Mat src = imread("../data/race-001.png");
    cvtColor(src, img1, COLOR_BGR2GRAY);
    src = imread("../data/race-002.png");
    cvtColor(src, img2, COLOR_BGR2GRAY);

    

    printf("Dimensions of the Image: %d, %d", src_color.cols, src_color.rows);

    int xarea = 3, yarea = 3, thres = 1;

    //    Step 2 - Implementing Lucas-Kanade Tracker
    Mat previmg = img1, curimg;
    for (int i = 1; i < 30; i++)
    {
        char buffer[17];

        const auto begin = chrono::steady_clock::now();

        // sprintf(buffer, "filename%.3d.jpg", i-1);
        sprintf(buffer, "../data/race-%03d.png", i);
        printf("\nLoaded %s for prev", buffer);
        src = imread(buffer);
        cvtColor(src, previmg, COLOR_BGR2GRAY);
        auto previmgvec = mat2vec(previmg);

        sprintf(buffer, "../data/race-%03d.png", (i + 1));

        printf("\nLoaded %s", buffer);
        src = imread(buffer);
        cvtColor(src, curimg, COLOR_BGR2GRAY);
        auto curimgvec = mat2vec(curimg);

        //check they have same sime
        if (previmg.cols != curimg.cols || previmg.rows != curimg.rows)
        {
            printf("\nImages are not of the same size. Skipping");
            continue;
        }

        const auto endinit = chrono::steady_clock::now();

        deque<Point> corn = findCorners(previmg, xarea, yarea, thres, true);

        const auto endcorner = chrono::steady_clock::now();

        lucasKanade(previmg, curimg, 40, 40, xarea, yarea, corn, false, buffer);

        const auto endlk = chrono::steady_clock::now();
        printf("\nInit took %d ms", chrono::duration_cast<chrono::milliseconds>(endinit - begin).count());
        printf("\nCorner detection took %d ms", chrono::duration_cast<chrono::milliseconds>(endcorner - endinit).count());
        printf("\nLucas-Kanade took %d ms", chrono::duration_cast<chrono::milliseconds>(endlk - endcorner).count());
    }

    // Next we try to implement Horn-Shunck
}