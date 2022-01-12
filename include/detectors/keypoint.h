#ifndef KEYPOINT_H
#define KEYPOINT_H

#include "core.h"

std::string keypointToString( KeyPoints &kp );

void printKeypoint( KeyPoints &kp );

bool outOfBounds(int i, int j, cv::Size size_img);

float distanceBetwenTwoKeyPoints( KeyPoints p1, KeyPoints p2 );

void sortKeypoints( std::vector<KeyPoints> &vec );

std::vector<KeyPoints> vectorSlice(std::vector<KeyPoints> const &v, int m, int n);

void plotKeyPoints(cv::Mat img, std::vector<KeyPoints> kp, std::string out_path, int max_kp = MAX_KP);

void saveKeypoints(std::vector<KeyPoints> &kp, std::string out_path, int max_kp = MAX_KP,
                   bool transformCoordinate = true, bool descriptorToBundler = false);

std::vector<KeyPoints> loadKeypoints(std::string arqPath);

std::vector<KeyPoints> loadLoweKeypoints(std::string arqPath);

void loadOpenCVKeyPoints( std::vector<cv::KeyPoint> &ocv_kp, cv::Mat &descriptor,
                          std::vector<KeyPoints> &kpList );

void loadOpenCVKeyPoints( std::vector<cv::KeyPoint> &ocv_kp, std::vector<KeyPoints> &kpList );

/*
void loadOpenCVKeyPoints( std::vector<cv::KeyPoint> &ocv_kp, cv::Mat &descriptor,
                          std::vector<KeyPoints> &kpList, bool withDescription = false );
*/
#endif
