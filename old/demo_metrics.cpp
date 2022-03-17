#include <iostream>
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>

#include <bits/stdc++.h>
#include <limits> //std::numeric_limits

#include "opencv/cv.h"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  Mat image1;
  image1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); // Read the first file

  Mat image2;
  image2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

  if ((!image1.data) || (!image2.data)) {
    std::cout << "ERROR: Cannot load images in\n" << argv[1] << "\n" << argv[2] << endl;
    return -1;
  }

  vector < KeyPoint > keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;

  /** Construction of the feature detector
   */

  double ExTime = (double) cv::getTickCount();
//  cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create( 800 );

  /** Detection of the features
   */

  // Computing Keypoints using OpenCV SIFT with ROI
  cv::Ptr<cv::SIFT> siftImage1 = cv::SIFT::create();
  siftImage1->detect( image1, keypoints1 );
  siftImage1->compute( image1, keypoints1, descriptors1 );

  cv::Ptr<cv::SIFT> siftImage2 = cv::SIFT::create();
  siftImage2->detect( image2, keypoints2 );
  siftImage2->compute( image2, keypoints2, descriptors2 );

  //Calculate the time needed for code execution
  ExTime = ((double) cv::getTickCount() - ExTime) / cv::getTickFrequency();

  /** Draw the keypoints
   */
  Mat ImageKP1, ImageKP2;

  drawKeypoints(image1, keypoints1, ImageKP1, cv::Scalar(255, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  drawKeypoints(image2, keypoints2, ImageKP2, cv::Scalar(255, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  // Construction of the matcher 
  cv::Mat ImageMatch;
  cv::FlannBasedMatcher matcher;

  // Match the two image descriptors    
  std::vector < DMatch > matches;
  matcher.match(descriptors1, descriptors2, matches);

  double max_dist = 0;
  double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for (int i = 0; i < descriptors1.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  std::vector < DMatch > good_matches;

  for (int i = 0; i < descriptors1.rows; i++) {
    if (matches[i].distance <= 2 * min_dist) {
      good_matches.push_back(matches[i]);
    }
  }

  cout << "Number of good matches: " << good_matches.size() << endl;

  drawMatches(image1, keypoints1, image2, keypoints2, good_matches, ImageMatch, cv::Scalar(255, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  /** Evaluation of detected points
   */
  std::cout << ">" << std::endl;
  cout << "Evaluating feature detector..." << endl;
  float repeatability;
  int corrCounter;
  cv::Mat Homog;

  std::vector < cv::Point2f > srcKey;
  std::vector < cv::Point2f > refKey;

  for (int i = 0; i < matches.size(); i++) {
    srcKey.push_back(keypoints1[matches[i].queryIdx].pt);
    refKey.push_back(keypoints2[matches[i].queryIdx].pt);
  }

//  Homog = cv::findHomography(srcKey, refKey);
  Homog = cv::Mat::zeros( cv::Size(3,3), CV_32F);
  Homog.at<float>(0,0) =  1.0304722108346218e+00f;
  Homog.at<float>(0,1) =  5.4546970313168412e-03f;
  Homog.at<float>(0,2) = -7.2232829462272612e+00f;
  Homog.at<float>(1,0) =  3.2017981137050990e-02f;
  Homog.at<float>(1,1) =  1.0262851078286763e+00f;
  Homog.at<float>(1,2) = -3.4666448767093499e+01f;
  Homog.at<float>(2,0) =  1.6370538625827067e-05f;
  Homog.at<float>(2,1) = -2.6485632884678016e-06f;
  Homog.at<float>(2,2) =  1.0000903223070901e+00f;
  std::cout << "Hom: " << std::endl;
  std::cout << Homog << std::endl;

  cv::Mat imgOut;
  cv::warpPerspective( image1, imgOut, Homog, image1.size() );
  
  cv::imwrite( "image1.jpg", image1 );
  cv::imwrite( "image2.jpg", image2 );
  cv::imwrite( "imgOut.jpg", imgOut );
  cv::imwrite( "matchs.jpg", ImageMatch );
  

  cv::evaluateFeatureDetector(image1, image2, Homog, & keypoints1, & keypoints2, repeatability, corrCounter);

  std::cout << "repeatability = " << repeatability << std::endl;
  std::cout << "correspCount = " << corrCounter << std::endl;
  std::cout << ">" << std::endl;

//  system("pause");

  return 0;
}
