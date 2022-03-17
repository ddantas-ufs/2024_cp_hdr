#include "../include/evaluation/metrics.h"

void calculateRRUsingOpenCV( cv::Mat img1, cv::Mat img2, cv::Mat H, float &rr, int &cc,
                             std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2 )
{
  cv::evaluateFeatureDetector( img1, img2, H, &kp1, &kp2, rr, cc );
}