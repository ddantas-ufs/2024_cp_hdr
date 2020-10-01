#ifndef KEYPOINT_H
#define KEYPOINT_H

#define MAX_KP 500

struct KeyPoint
{
    int y;
	int x;
	float resp;
	int scale;
	int level;
};

bool outOfBounds(int i, int j, cv::Size size_img);

void plotKeyPoints(cv::Mat &img, std::vector<KeyPoint> kp);

bool compareResponse(KeyPoint a, KeyPoint b);

void transformCoord(std::vector<KeyPoint> &kp);

void saveKeypoints(std::vector<KeyPoint> &kp, cv::Mat roi[], std::string out_path, int max_kp = MAX_KP);

#endif