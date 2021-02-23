#include "../include/cphdr.h"

int main(int, char** argv)
{
	cv::Mat img_in, img_gray;
	std::vector<KeyPoints> kp;
	std::string img_name, out_path;

	readImg(argv[1], img_in, img_gray, img_name);
	out_path = std::string(argv[2]) + img_name + ".dog";

	dogKp(img_gray, kp);
	
	saveKeypoints(kp, out_path);
	plotKeyPoints(img_in, kp, out_path);

	return 0;
}