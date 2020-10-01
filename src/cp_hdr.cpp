#include <bits/stdc++.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "cp_hdr.h"
#include "harris.h"
#include "dog.h"

void dogKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoint> &kp, int mgauss, int maxsup_size, float contrast_th, float curv_th)
{
	cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS];
	cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1];

	dogInitScales(img, scales, mgauss);
	dogCalc(scales, dog);
	dogMaxSup(dog, roi, kp, maxsup_size);
	dogThreshold(kp, dog, contrast_th, curv_th);
}

void harrisKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoint> &kp, int msobel, int mgauss, int k, float min_quality, int msize)
{
    cv::Mat resp_map;

    cv::GaussianBlur(img, img, cv::Size(mgauss, mgauss), 0, 0, cv::BORDER_DEFAULT);
    
    harrisCalc(img, resp_map, roi, msobel, mgauss, k);
    harrisThreshold(resp_map, kp, min_quality);
    harrisMaxSup(resp_map, kp, msize);
}