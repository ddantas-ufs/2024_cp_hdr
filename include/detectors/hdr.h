#ifndef HDR_H
#define HDR_H

#include "core.h"

void coefVar(cv::Mat img, cv::Mat &img_cv, int mask_size = CV_SIZE, bool gauss = true,
             float sigma = SIGMA_X);

void logTransform(cv::Mat img, cv::Mat &img_log);

void applyCVMask( cv::Mat img, cv::Mat &res );

void applyCVMask( cv::Mat &img );

#endif
