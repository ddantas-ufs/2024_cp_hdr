#ifndef CORE_H
#define CORE_H

#include <bits/stdc++.h>
#include <limits> //std::numeric_limits

#include <opencv2/calib3d.hpp> // OpenCV RepeatbilityRate metric

#include "opencv/cv.h"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define SQRT_2 1.414213562f //1.41421356237; // Square root of two [sqrt(2)]

#define NUM_OCTAVES 1//4 // number of octaves
#define NUM_SCALES 5 // number of scales
#define MAX_INTERP_STEPS 5 // interpolation max steps before failure (OpenSIFT, OpenCV)
#define DOG_BORDER 5 // interpolation border to ignore keypoints
#define CONTRAST_TH 0.03f // prybil set to 8
#define CURV_TH 10.0f // curvature threshold
#define GAUSS_SIZE 9 // mask size of gauss bluring
#define SIGMA_X 1.0 // fix value (ex. 1.0) to keep a standard
#define SIGMA_Y 1.0 // fix value (ex. 1.0) to keep a standard
#define MAXSUP_SIZE 5 // can be 3 (based on Lowe's paper)
#define SOBEL_SIZE 5 // mask size of sobel operator
#define K 0.04 // harris constant
#define MIN_QUALITY 0.01 // quality percentual for the kp response
#define MAX_KP 500 // if zero, it is not used
#define ALL_KP -1  // when all keypoints should be used
#define CV_SIZE 3 // mask size to compute coefficient of variation
#define LOG_TRANSFORM_CONSTANT 2.0f//1.0f//2.302585f
#define LDR_MAX_RANGE 255.0
#define HDR_MAX_RANGE 256.0

// DEFINES IF DOG WILL USE CV FILTER
#define USE_CV_FILTER_FALSE false
#define USE_CV_FILTER_TRUE true

// #define USE_CV_FILTER USE_CV_FILTER_FALSE

// MAPPING INTERVALS
#define MAPPING_INTERVAL_FLOAT_0_1 0
#define MAPPING_INTERVAL_FLOAT_0_255 1

// SIFT CONSTANTS
#define SIFT_DESC_ORIENT_SIGMA 1.5f
#define SIFT_DESC_ORIENT_WINDOW 5
#define SIFT_DESC_ORIENT_HIST_BINS 36

#define SIFT_DESC_SIZE 128 // descriptor size
#define SIFT_DESC_WINDOW 16 // window around keypoint size
#define SIFT_DESC_SW_QTD 4 // SubWindow quantity
#define SIFT_DESC_SW_SIZE 4 // SubWindow size
#define SIFT_DESC_BINS_PER_SW 8 // bins for subwindow histogram 
#define SIFT_DESC_MAG_THR 0.2f // magnitude threshold of descriptor elements
#define SIFT_DESC_INT_FTR 512.0f // factor used to convert float descriptor to int
#define SIFT_DESC_SCL_FTR 3.0f // to determine size of a single descriptor orientation histogram

// MATCHING CONSTANTS
#define MATCHING_HAMMING_DIST_CALC 0
#define MATCHING_EUCLIDEAN_DIST_CALC 1

//#define MATCHING_RATIO_MATCH 122.13f // numero obtido calculando 3% de (largura+altura / 2) das imagens de teste.
#define MATCHING_RATIO_MATCH 15
#define MATCHING_NNDR_THRESHOLD 0.7f

// REPEATABILITY KEYPOINT DISK RATIO
#define REPEATABILITY_RATIO 8
#define REPEATABILITY_MIN_OVERLAP 0.7f

#define MATCHING_INCORRECT 0
#define MATCHING_CORRECT 1

// BUBBLE SIZE = MATCHING_RATIO_MATCH
#define AP_BUBBLE_RATIO MATCHING_RATIO_MATCH

struct KeyPoints
{
  float y;
  float x;
  float resp;
  int octave;
  int scale;
  float direction;

  std::vector<int> descriptor;
};

struct MatchedKeyPoints
{
  KeyPoints kp1, kp2;
  bool isCorrect;
};

#endif
