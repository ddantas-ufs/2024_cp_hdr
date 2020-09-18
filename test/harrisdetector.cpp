#include <bits/stdc++.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "../src/input.h"
#include "../src/keypoint.h"
#include "../src/harris.h"

int main(int, char** argv)
{	
	cv::Mat img_in;
	cv::Mat img_gray;
	cv::Mat roi[4];
	cv::Mat resp_map;
	std::vector<KeyPoint> kp;
	std::string img_name = getFileName(std::string(argv[1]));
	std::string out_path = "../out/" + img_name + ".harris";
	
	readData(argv[1], argv[2], img_in, img_gray, roi);
	responseCalc(img_gray, resp_map, roi);
	threshold(resp_map, kp);
	localMaxSup(resp_map, kp);	
	saveKeypoints(kp, roi, "../out/" + img_name + ".harris");
	plotKeyPoints(img_in, kp);
	cv::imwrite(out_path + ".kp.png", img_in);

	return 0;
}