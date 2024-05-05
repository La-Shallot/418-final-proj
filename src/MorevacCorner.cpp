#include <iostream>
#include <vector>
#include <deque>

#include <morevac_corner.h>
#include <opencv2/highgui/highgui.hpp>

MorevacCorner::MorevacCorner() {}

std::deque<cv::Point> MorevacCorner::findCorners(cv::Mat img, int xarea, int yarea, int thres, bool verbose = true, bool visual = false) {
    std::deque<cv::Point> corners;

            cv::Mat outimg = img.clone(); // This will be used to provide a visual indication of the corners present

            int dimx = img.cols, dimy = img.rows;
            // omp_lock_t vec_lock;
            // omp_init_lock(&vec_lock);

            int count = 0;
            #pragma omp parallel for schedule(dynamic, 32)
            // for (int startx_dummy = 0; (startx_dummy) < dimx/xarea - 1; startx_dummy += 1)
            //     for (int starty = 0; (starty + yarea) < dimy; starty += yarea)
            for (int starty_dummy = 0; (starty_dummy) < dimy/yarea - 1; starty_dummy += 1)
                for (int startx = 0; (startx + xarea) < dimx; startx += xarea)
                {
                    // int startx = startx_dummy * xarea;
                    int starty = starty_dummy * yarea;
                    count++;
                    if (verbose){
                        // printf("\n Area %d - Currenty looking at area (%d-%d,%d-%d)\n", count, startx, startx + xarea, starty, starty + yarea);
                    }
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
                        //                    printf("Current direction is %d. New area is (%d-%d,%d-%d)\n",dir,newsx,min(dimx,newsx+xarea),newsy,min(newsy+yarea,dimy));

                        cv::Mat newarea = img(cv::Range(newsy, std::min(newsy + yarea, dimy)), cv::Range(newsx, std::min(newsx + xarea, dimx)));

                        if (newarea.cols != curarea.cols || newarea.rows != curarea.rows)
                        {
                            continue;
                        }
                        cv::Mat diff = abs(curarea - newarea);
                        //print cv::mean(diff)
                        auto mean = cv::mean(diff).val;
                        printf("CV mean %d, %d, %d, %d\n", mean[0], mean[1], mean[2], mean[3]);
                        auto mean_mean = cv::mean(cv::mean(diff)).val;
                        printf("CV mean mean: %d, %d, %d, %d\n", mean_mean[0], mean_mean[1], mean_mean[2], mean_mean[3]);
                        auto mean_mean_final = cv::mean(cv::mean(diff))(0);
                        printf("CV mean mean final: %d\n", mean_mean_final);
                        
                        // printf("CV mean mean: %s\n", cv::mean(cv::mean(diff)));
                        // printf("Type of cv::mean(diff) is %s\n", typeid(cv::mean(diff)).name());
                        results[dir % 2] = cv::mean(cv::mean(diff))(0);
                    }
                    results[0] /= 2;
                    results[1] /= 2;

                    // thresholding
                    if (results[0] >= thres && results[1] >= thres)
                    {   
                        //lock
                        // omp_set_lock(&vec_lock);
                        // printf("Pushing corner %d,%d\n", startx, starty);
                        #pragma omp critical
                        corners.push_back(cv::Point(startx, starty));
                        // #pragma omp critical
                        // rectangle(outimg, Point(startx, starty), Point(startx + xarea, starty + yarea), Scalar(0), 2);
                        // omp_unset_lock(&vec_lock);
                    }

                    // log << startx << "," << starty << "," << results[0] << "," << results[1] << "\n";
                }

            printf("\nFound %d corners", corners.size());

            // omp_destroy_lock(&vec_lock);

            return corners;
}

// std::vector<std::pair<float, float>> MorevacCorner::findCorners(std::vector<std::vector<float>>& img, int xarea, int yarea, int thres, bool verbose)
// {
//     return std::vector<std::pair<float, float>>();
// }

std::vector<std::pair<float, float>> MorevacCorner::findCorners(std::vector<std::vector<float>>& img, int xarea, int yarea, int thres, bool verbose = true)
    {
        std::vector<std::pair<float, float>> corners;
        //vector<std::vector<float>> outimg = floatVectorToMat(img).clone(); // This will be used to provide a visual indication of the corners present



        if (verbose)
            printf("This is still under construction. - areas - (%d,%d), thres - %d", xarea, yarea, thres);

        // int dimx = img.cols, dimy = img.rows;
        int dimy = img[0].size(), dimx = img.size();
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
                    // printf("Current direction is %d. New area is (%d-%d,%d-%d)\n",dir,newsx,min(dimx,newsx+xarea),newsy,min(newsy+yarea,dimy));

                    cv::Range oldyR = cv::Range(starty, std::min(starty + yarea, dimy));
                    cv::Range oldxR = cv::Range(startx, std::min(startx + xarea, dimx));
                    cv::Range newyR = cv::Range(newsy, std::min(newsy + yarea, dimy));
                    cv::Range newxR = cv::Range(newsx, std::min(newsx + xarea, dimx));

                    if (oldxR.size() != newxR.size() || oldyR.size() != newyR.size()) {
                        std::cerr << "Error: Regions have different dimensions." << std::endl;
                        if (verbose){
                            // printf("Skipping due to dimensional or similarity issues");
                        }
                        continue;
                    
                    }

                    float sum=0;
                    int totalelem= oldxR.size() * oldyR.size();
                    for (int i = 0; i < oldyR.size(); ++i) {
                        for (int j = 0; j < oldxR.size(); ++j) {
                            float oldPixel = img[oldyR.start + i][oldxR.start + j];
                            float newPixel = img[newyR.start + i][newxR.start + j];
                            float result = std::abs(oldPixel - newPixel);
                            sum += result * result;
                        }
                    }



                    //get mean
                    //double mean = sum / totalelem;


                    results[dir % 2] = sum;
                }

                results[0] /= 2;
                results[1] /= 2;
                if (verbose)
                    // printf("Scores obtained: %f, %f\n", results[0], results[1]);

                // thresholding
                float thres = 0.001f;
                // printf("Results: %f, %f\n", results[0], results[1]);
                if (results[0] >= thres && results[1] >= thres)
                {   
                    //lock
                    // omp_set_lock(&vec_lock);
                    // printf("Pushing corner %d,%d\n", startx, starty);
                    #pragma omp critical
                    corners.push_back(std::make_pair(startx, starty));
                    // #pragma omp critical
                    // rectangle(outimg, Point(startx, starty), Point(startx + xarea, starty + yarea), Scalar(0), 2);
                    // omp_unset_lock(&vec_lock);
                }

            }

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