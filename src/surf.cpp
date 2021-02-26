#include "../include/detectors/surf.h"

void surfKp(cv::Mat img, std::vector<KeyPoints> &kp)
{
  cv::Mat img_sum;
  cv::integral(img, img_sum);
}