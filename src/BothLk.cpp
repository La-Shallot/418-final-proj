#include <both_lk.h>
#include <omp.h>

BothLk::BothLk() {}

std::vector<std::pair<std::pair<int, int>, std::pair<float, float>>> BothLk::CalcHarrisLkFlow(std::vector<std::vector<float>> &imgA, std::vector<std::vector<float>> &imgB, int lk_xsarea, int lk_ysarea, int lk_xarea, int lk_yarea, int h_xarea, int h_yarea, int h_thres, bool verbose)
{
    const float k = 0.04f; // Harris constant

    std::vector<std::pair<std::pair<int, int>, std::pair<float, float>>> out;

    int dimx = imgA[0].size(), dimy = imgA.size();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = h_yarea; y < dimy - h_yarea; ++y)
    {
        for (int x = h_xarea; x < dimx - h_xarea; ++x)
        {
            float dx2 = 0.0f, dy2 = 0.0f, dxy = 0.0f;
            float dx = 0.0f, dy = 0.0f;

            for (int j = -h_yarea; j <= h_yarea; ++j)
            {
                for (int i = -h_xarea; i <= h_xarea; ++i)
                {
                    if (y + j > 0 && y + j < dimy - 1 && x + i > 0 && x + i < dimx - 1)
                    {
                        dx = imgA[y + j][x + i + 1] - imgA[y + j][x + i - 1];
                        dy = imgA[y + j + 1][x + i] - imgA[y + j - 1][x + i];
                        dx2 += dx * dx;
                        dy2 += dy * dy;
                        dxy += dx * dy;
                    }
                }
            }

            float det = dx2 * dy2 - dxy * dxy;
            float trace = dx2 + dy2;
            float response = det - k * trace * trace;

            if (response > h_thres)
            {
                auto cur_corner = std::make_pair(x, y);
                auto corner_mid = std::make_pair((int)(cur_corner.first + (int)(lk_xarea / 2)), (int)(cur_corner.second + (int)(lk_yarea / 2)));

                auto sry = std::make_pair(std::max(0, corner_mid.second - (int)(lk_ysarea / 2)), std::min(dimy - 1, corner_mid.second + (int)(lk_ysarea / 2)));
                auto srx = std::make_pair(std::max(0, corner_mid.first - (int)(lk_xsarea / 2)), std::min(dimx - 1, corner_mid.first + (int)(lk_xsarea / 2)));

                int range = 1;
                double G[2][2] = {0, 0, 0, 0};

                // IxIt and IyIt
                double b[2] = {0, 0};

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

                auto cur_flow = std::make_pair(V[0], V[1]);
                #pragma omp critical
                out.push_back(std::make_pair(cur_corner, cur_flow));
            }
        }
    }
    return out;
}

std::vector<std::pair<std::pair<int, int>, std::pair<float, float>>> BothLk::CalcMoreLkFlow(std::vector<std::vector<float>> &imgA, std::vector<std::vector<float>> &imgB, int lk_xsarea, int lk_ysarea, int lk_xarea, int lk_yarea, int h_xarea, int h_yarea, int h_thres, bool verbose)
{
    std::vector<std::pair<std::pair<int, int>, std::pair<float, float>>> out;

    int dimx = imgA[0].size(), dimy = imgA.size();

    std::vector<std::pair<int, int>> dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    #pragma omp parallel for schedule(dynamic, 1)
    for (int startx_dummy = 0; (startx_dummy) < dimx / h_xarea - 1; startx_dummy += 1)
        for (int starty = 0; (starty + h_yarea) < dimy; starty += h_yarea)
        {
            int startx = startx_dummy * h_xarea;

            double results[4] = {0, 0, 0, 0};
            // for all four directions calculate the change

            for (int currdir = 0; currdir < 4; currdir++)
            {
                std::pair<int, int> offset = dir[currdir];
                int newsx = startx + offset.first * h_xarea;
                int newsy = starty + offset.second * h_yarea;

                if (newsx < 0 || newsy < 0 || newsx + h_xarea >= dimx || newsy + h_yarea >= dimy)
                {
                    if (verbose)
                    {
                        // printf("Skipping due to boundary issues");
                    }
                    continue;
                }

                // get E value
                for (int j = 0; j < h_yarea; ++j)
                {
                    for (int i = 0; i < h_xarea; ++i)
                    {
                        float oldPixel = imgA[starty + j][startx + i];
                        float newPixel = imgA[newsy + j][newsx + i];
                        float result = newPixel - oldPixel;
                        results[currdir] += result * result;
                    }
                }
            }

            // get the smallest value from the directions
            float minval = std::min(results[0], std::min(results[1], std::min(results[2], results[3])));

            // thresholding for the min value obtained per direction: local maxima means corner :)
            if (minval > h_thres)
            {
                auto cur_corner = std::make_pair(startx, starty);
                auto corner_mid = std::make_pair((int)(cur_corner.first + (int)(lk_xarea / 2)), (int)(cur_corner.second + (int)(lk_yarea / 2)));

                auto sry = std::make_pair(std::max(0, corner_mid.second - (int)(lk_ysarea / 2)), std::min(dimy - 1, corner_mid.second + (int)(lk_ysarea / 2)));
                auto srx = std::make_pair(std::max(0, corner_mid.first - (int)(lk_xsarea / 2)), std::min(dimx - 1, corner_mid.first + (int)(lk_xsarea / 2)));

                int range = 1;
                double G[2][2] = {0, 0, 0, 0};

                // IxIt and IyIt
                double b[2] = {0, 0};

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

                auto cur_flow = std::make_pair(V[0], V[1]);
                #pragma omp critical
                out.push_back(std::make_pair(cur_corner, cur_flow));
            }
        }

    return out;
}
