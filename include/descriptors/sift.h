#ifndef SIFT_H
#define SIFT_H

#include "../detectors/core.h"

void siftDescriptor( std::vector<KeyPoints> kp, cv::Mat& img, cv::Mat& imgGray,
                     int mGauss = DESC_GAUSS_WINDOW, float sigma = DESC_GAUSS_SIGMA );

#endif