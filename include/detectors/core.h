#ifndef CORE_H
#define CORE_H

#include <bits/stdc++.h>
#include <limits>       // std::numeric_limits

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define DOG_SCL_ROWS 4
#define DOG_SCL_COLS 5
#define DOG_MAX_INTERP_STEPS 5 //interpolation max steps before failure. Based on OpenSIFT and OpenCV
#define SIFT_IMG_BORDER 5 //interpolation border to ignore keypoints
#define CONTRAST_TH 0.03 //prybil set to 8
#define CURV_TH 5
#define GAUSS_SIZE 9
#define SIGMA_X 0 //fix value (ex. 1.0) to keep a standard
#define SIGMA_Y 0 //fix value (ex. 1.0) to keep a standard
#define MAXSUP_SIZE 21
#define SOBEL_SIZE 7
#define K 0.04
#define MIN_QUALITY 0.05
#define MAX_KP 500
#define CV_SIZE 3 //can be 5 to harris

struct KeyPoints
{
    float y; //change to float soon
	float x; //change to float soon
	float resp;
	int scale;
	int level;
};

#define ABS(x) (((x) < 0 ) ? -(x) : (x))

#endif
