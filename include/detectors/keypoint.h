#ifndef KEYPOINT_H
#define KEYPOINT_H

#include "core.h"

bool outOfBounds(int i, int j, cv::Size size_img);

void plotKeyPoints(cv::Mat &img, std::vector<KeyPoints> kp);

void saveKeypoints(std::vector<KeyPoints> &kp, cv::Mat roi[], std::string out_path, int max_kp = MAX_KP);

#endif