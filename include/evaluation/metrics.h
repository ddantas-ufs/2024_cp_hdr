#ifndef METRICS_H
#define METRICS_H

#include "../detectors/core.h"
#include "../detectors/keypoint.h"

void calculateRRUsingOpenCV( cv::Mat img1, cv::Mat img2,
                             cv::Mat H, float &rr, int &cc,
                             std::vector<cv::KeyPoint> kp1,
                             std::vector<cv::KeyPoint> kp2 );

void calculateRR( cv::Mat H, std::vector<KeyPoints> kp1,
                  std::vector<KeyPoints> kp2, int &cc, float &rr );

float calculateUniformity( std::vector<int> qtdKps );

float calculateUniformity( std::vector< std::vector<KeyPoints> > lKps );

float calculateAreaOfIntersection( KeyPoints A, KeyPoints B );

float calculateIoU( KeyPoints kp1, KeyPoints kp2 );

//float calculateIoU( std::vector<MatchedKeyPoints> kpPairs, cv::Mat H );

float calculateAP( std::vector<MatchedKeyPoints> kpPairs, cv::Mat H );

#endif