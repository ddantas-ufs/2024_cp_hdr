#include "../detectors/core.h"
#include "../cphdr.h"

int siftDescriptor( std::vector<KeyPoints> kp, cv::Mat img, cv::Mat imgGray,
                    int mGauss = DESC_GAUSS_WINDOW,
                    float sigma = DESC_GAUSS_SIGMA );