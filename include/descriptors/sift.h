#ifndef SIFT_H
#define SIFT_H

#include "../detectors/core.h"
#include "../detectors/keypoint.h" // can be removed later. used only to print keypoints

void siftDescriptor( std::vector<KeyPoints> kp, cv::Mat& img, cv::Mat& imgGray,
                     int mGauss = DESC_GAUSS_WINDOW, float sigma = DESC_GAUSS_SIGMA );

#endif