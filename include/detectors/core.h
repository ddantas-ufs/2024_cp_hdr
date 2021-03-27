#ifndef CORE_H
#define CORE_H

#include <bits/stdc++.h>
#include <limits> //std::numeric_limits

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define NUM_OCTAVES 4 // number of octaves
#define NUM_SCALES 5 // number of scales
#define MAX_INTERP_STEPS 5 // interpolation max steps before failure (OpenSIFT, OpenCV)
#define DOG_BORDER 5 // interpolation border to ignore keypoints
#define CONTRAST_TH 0.03 // prybil set to 8
#define CURV_TH 10.0 // curvature threshold
#define GAUSS_SIZE 9 // mask size of gauss bluring
#define SIGMA_X 1.0 // fix value (ex. 1.0) to keep a standard
#define SIGMA_Y 1.0 // fix value (ex. 1.0) to keep a standard
#define MAXSUP_SIZE 3 // can be 3 (based on Lowe's paper)
#define SOBEL_SIZE 5 // mask size of sobel operator
#define K 0.04 // harris constant
#define MIN_QUALITY 0.01 // quality percentual for the kp response
#define MAX_KP 500 // if zero, it is not used
#define CV_SIZE 5 // mask size to compute coefficient of variation

#define DESC_GAUSS_WINDOW 5
#define DESC_GAUSS_SIGMA 1.5
#define DESC_HIST_BINS 36
#define DESC_BINS 8
#define DESC_RADIUS 3 * DESC_GAUSS_SIGMA

struct KeyPoints
{
  float y;
  float x;
  float resp;
  int octave;
  int scale;
  float direction;
};

#endif
