#ifndef MATCHING_H
#define MATCHING_H

#include "../detectors/core.h"
#include "../detectors/keypoint.h"

float calculateDistance( std::vector<int> vec1, std::vector<int> vec2, 
                         int distanceMethod = MATCHING_EUCLIDEAN_DIST_CALC );

void concatenateImages( cv::Mat img1, cv::Mat img2, cv::Mat &out );

void nndr( std::vector<KeyPoints> kpListImg1,
           std::vector<KeyPoints> kpListImg2,
           std::vector<MatchedKeyPoints> &output,
           cv::Mat H, float threshold,
           int calcDistMode = MATCHING_EUCLIDEAN_DIST_CALC );

void printLineOnImages( cv::Mat img1, cv::Mat img2, cv::Mat &out,
                        std::vector<MatchedKeyPoints> matchedDesc );

/**
 * All overloads on method matchFPs
**/
void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList,
               cv::Mat H, std::vector<MatchedKeyPoints> &kpsOut,
               cv::Mat &imgOut );

void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList,
               cv::Mat H, cv::Mat &imgOut );

void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList,
               cv::Mat H );

void matchFPs( std::string img1Path, std::vector<KeyPoints> img1KpList,
               std::string img2Path, std::vector<KeyPoints> img2KpList,
               std::string H );

// matching without Homography
void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList );

void matchFPs( cv::Mat img1, cv::Mat img2, 
               std::vector<KeyPoints> img1KpList,
               std::vector<KeyPoints> img2KpList,
               cv::Mat &imgOut );

#endif

