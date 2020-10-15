#ifndef DOG_H
#define DOG_H

#include "core.h"

void dogKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoints> &kp,
           int mgauss = GAUSS_SIZE, int maxsup_size = MAXSUP_SIZE,
           float contrast_th = CONTRAST_TH, float curv_th = CURV_TH);

#endif