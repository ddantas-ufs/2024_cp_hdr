#include <bits/stdc++.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "../src/input.h"
#include "../src/keypoint.h"
#include "../src/dog.h"

int main(int, char** argv)
{
	cv::Mat img_in;
	cv::Mat img_gray;
	cv::Mat roi[4];
	cv::Mat scales[SCALES_ROWS][SCALES_COLS];
	cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1];
	std::vector<KeyPoint> kp;
	std::string img_name = getFileName(std::string(argv[1]));
	std::string out_path = "../out/" + img_name + ".dog";

	readData(argv[1], argv[2], img_in, img_gray, roi);
	initOctaves(img_gray, scales);
	calcDoG(scales, dog);
	localMaxSup(dog, roi, kp);
	contrastThreshold(kp, dog);
	edgeThreshold(kp, dog);
	transformCoord(kp);
	saveKeypoints(kp, roi, "../out/" + img_name + ".dog");
	plotKeyPoints(img_in, kp);
	cv::imwrite(out_path + ".kp.png", img_in);

	return 0;
}