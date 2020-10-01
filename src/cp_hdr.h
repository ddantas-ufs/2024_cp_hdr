#include "keypoint.h"
#include "harris.h"
#include "dog.h"

#define CONTRAST_TH 0.03
#define CURV_TH 5
#define MAXSUP_SIZE 21
#define GAUSS_SIZE 9
#define MIN_QUALITY 0.05
#define SOBEL_SIZE 7
#define K 0.04

void dogKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoint> &kp,
           int mgauss = GAUSS_SIZE, int maxsup_size = MAXSUP_SIZE, float contrast_th = CONTRAST_TH, float curv_th = CURV_TH);
void harrisKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoint> &kp,
              int msobel = SOBEL_SIZE, int mgauss = GAUSS_SIZE, int k = K,
              float min_quality = MIN_QUALITY, int msize = MAXSUP_SIZE);