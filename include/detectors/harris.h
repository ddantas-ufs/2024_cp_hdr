#ifndef HARRIS_H
#define HARRIS_H

#include "core.h"

void harrisKp(cv::Mat img, std::vector<KeyPoints> &kp, bool is_hdr = false,
              int msobel = SOBEL_SIZE, int mgauss = GAUSS_SIZE,
              float sigma_x = SIGMA_X, float sigma_y = SIGMA_Y, float k = K,
              float min_quality = MIN_QUALITY, int msize = MAXSUP_SIZE,
              int cv_size = CV_SIZE);

void harrisKp( cv::Mat img, std::vector< std::vector<KeyPoints> > &kpList,
               std::vector<cv::Mat> lRoi, bool is_hdr );
#endif
