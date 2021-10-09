#ifndef SIFT_H
#define SIFT_H

#include "../detectors/core.h"

void matchDescriptors(std::vector<KeyPoints> kpListImg1,
                      std::vector<KeyPoints> kpListImg2,
                      std::map<char, char> output,
                      float threshold,
                      int calcDistMode = MATCHING_EUCLIDIAN_DIST_CALC);

#endif