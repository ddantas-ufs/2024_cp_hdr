#ifndef SIFT_H
#define SIFT_H

#include "../detectors/core.h"
#include "../cphdr.h"

void dogKp(cv::Mat img, std::vector<KeyPoints> &kp, bool refine_px = true,
           int mgauss = GAUSS_SIZE, int maxsup_size = MAXSUP_SIZE,
           float contrast_th = CONTRAST_TH, float curv_th = CURV_TH);

void siftDescriptor(KeyPoints kp, cv::Mat img, cv::Mat imgGray, 
                    int mGauss = DESC_GAUSS_WINDOW,
                    float sigma = DESC_GAUSS_SIGMA );
