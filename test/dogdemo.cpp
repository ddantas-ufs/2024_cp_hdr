#include <bits/stdc++.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "../src/input.h"
#include "../src/keypoint.h"
#include "../src/cp_hdr.h"

int main(int, char** argv)
{
	cv::Mat img_in;
	cv::Mat img_gray;
	cv::Mat roi[4];
	std::vector<KeyPoint> kp;
	std::string img_name;

	readData(argv[1], argv[2], img_in, img_gray, img_name, roi);
	std::string out_path = "out/" + img_name + ".dog";

	dogKp(img_gray, roi, kp);
	
	transformCoord(kp);
	saveKeypoints(kp, roi, out_path);
	plotKeyPoints(img_in, kp);
	cv::imwrite(out_path + ".kp.png", img_in);

	return 0;
}