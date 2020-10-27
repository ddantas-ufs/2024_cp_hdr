#ifndef HARRIS_H
#define HARRIS_H

#include "core.h"

void harrisKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoints> &kp,
              int msobel = SOBEL_SIZE, int mgauss = GAUSS_SIZE, int k = K,
              float min_quality = MIN_QUALITY, int msize = MAXSUP_SIZE);

#endif