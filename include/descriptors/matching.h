#ifndef MATCHING_H
#define MATCHING_H

#include "../detectors/core.h"

void matchDescriptors(std::vector<KeyPoints> kpListImg1,
                      std::vector<KeyPoints> kpListImg2,
                      std::map<char, char> output,
                      float threshold,
                      int calcDistMode = MATCHING_EUCLIDIAN_DIST_CALC);

float calculateDistance( std::vector<int> vec1, std::vector<int> vec2, 
                         int distanceMethod = MATCHING_EUCLIDIAN_DIST_CALC );

void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList );

#endif

