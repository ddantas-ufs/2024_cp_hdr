#define MIN_QUALITY 0.05
#define MASK_NMAXSUP 21
#define MASK_SOBEL 7
#define MASK_GAUSS 9
#define K 0.04

#include "keypoint.h"

void responseCalc(cv::Mat img, cv::Mat &resp_map, cv::Mat roi[], int msobel = MASK_SOBEL, int mgauss = MASK_GAUSS, int k = K);
void threshold(cv::Mat &resp_map, std::vector<KeyPoint> &kp, float min_quality = MIN_QUALITY);
void localMaxSup(cv::Mat &resp_map, std::vector<KeyPoint> &kp, int msize = MASK_NMAXSUP);