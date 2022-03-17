#ifndef METRICS_H
#define METRICS_H

#include "../detectors/core.h"
#include "../detectors/keypoint.h"

void calculateRRUsingOpenCV( cv::Mat img1, cv::Mat img2, cv::Mat H, float &rr, int &cc,
                             std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2 );

//void calculateRR( cv::Mat H, std::vector<MatchedKeyPoints> kpList, float &rr );
void calculateRR( cv::Mat H, std::vector<KeyPoints> kp1, std::vector<KeyPoints> kp2,
                  int &cc, float &rr );

#endif