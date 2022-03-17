#ifndef METRICS_H
#define METRICS_H

#include "../detectors/core.h"

void calculateRRUsingOpenCV( cv::Mat img1, cv::Mat img2, cv::Mat H, float &rr, int &cc,
                             std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2 );

#endif