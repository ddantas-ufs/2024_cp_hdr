#include "keypoint.h"

void harrisCalc(cv::Mat img, cv::Mat &resp_map, cv::Mat roi[], int msobel, int mgauss, int k);

void harrisThreshold(cv::Mat &resp_map, std::vector<KeyPoint> &kp, float min_quality);

void harrisMaxSup(cv::Mat &resp_map, std::vector<KeyPoint> &kp, int msize);