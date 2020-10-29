#include "../include/cphdr.h"

int main(int, char** argv)
{
	cv::Mat img_in;
	cv::Mat img_gray;
	cv::Mat roi[4];
	std::vector<KeyPoints> kp;
	std::string img_name;

	readImg(argv[1], img_in, img_gray, img_name);
	readRoi(argv[2], roi, img_gray.size());
	std::string out_path = std::string(argv[3]) + img_name + ".dog";

	dogKp(img_gray, roi, kp);
	
	saveKeypoints(kp, roi, out_path);
	plotKeyPoints(img_in, kp, out_path);

	return 0;
}