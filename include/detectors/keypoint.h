#ifndef KEYPOINT_H
#define KEYPOINT_H

#include "core.h"

std::string keypointToString( KeyPoints &kp );

void printKeypoint( KeyPoints &kp );

bool outOfBounds(int i, int j, cv::Size size_img);

void plotKeyPoints(cv::Mat img, std::vector<KeyPoints> kp, std::string out_path, int max_kp = MAX_KP);

void saveKeypoints(std::vector<KeyPoints> &kp, std::string out_path, int max_kp = MAX_KP,
                   bool transformCoordinate = true, bool descriptorToBundler = false);

std::vector<KeyPoints> loadKeypoints(std::string arqPath);

std::vector<KeyPoints> loadLoweKeypoints(std::string arqPath);

#endif
