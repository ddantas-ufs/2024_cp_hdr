#ifndef CORE_H
#define CORE_H

#include <bits/stdc++.h>
#include <limits> //std::numeric_limits

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define NUM_OCTAVES 4
#define NUM_SCALES 5
#define MAX_INTERP_STEPS 5 //interpolation max steps before failure (based on OpenSIFT and OpenCV)
#define DOG_BORDER 5 //interpolation border to ignore keypoints
#define CONTRAST_TH 0.03 //prybil set to 8
#define CURV_TH 5
#define GAUSS_SIZE 9
#define SIGMA_X 0 //fix value (ex. 1.0) to keep a standard
#define SIGMA_Y 0 //fix value (ex. 1.0) to keep a standard
#define MAXSUP_SIZE 21 //can be 3 (based on Lowe's paper)
#define SOBEL_SIZE 7
#define K 0.04
#define MIN_QUALITY 0.05
#define MAX_KP 500 //if zero, it is not used
#define CV_SIZE 3 //can be 5 to harris

#define DESC_GAUSS_WINDOW 5
#define DESC_GAUSS_SIGMA 1.5
#define DESC_HIST_BINS 36

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
