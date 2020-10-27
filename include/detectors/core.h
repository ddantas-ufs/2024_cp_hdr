#ifndef CORE_H
#define CORE_H

#include <bits/stdc++.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define DOG_SCL_ROWS 4
#define DOG_SCL_COLS 5
#define CONTRAST_TH 0.03
#define CURV_TH 5
#define GAUSS_SIZE 9
#define MAXSUP_SIZE 21
#define SOBEL_SIZE 7
#define K 0.04
#define MIN_QUALITY 0.05
#define MAX_KP 500

struct KeyPoints
{
    int y;
	int x;
	float resp;
	int scale;
	int level;
};

#endif