#include "../include/detectors/input.h"

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name)
{
	img_in = cv::imread(img_path, cv::IMREAD_UNCHANGED);

	if (img_in.channels() != 1)
	{
		cv::cvtColor(img_in, img_gray, cv::COLOR_BGR2GRAY);
	}
	if (img_in.depth() == CV_32F)
	{
		cv::normalize(img_in, img_gray, 0.0, 256.0, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	}
	img_name = getFileName(std::string(img_path));
}

std::string getFileName(std::string file_path)
{
	size_t size = file_path.rfind("/", file_path.length());
	
	if (size != std::string::npos)
	{
		file_path = file_path.substr(size + 1, file_path.length() - size);
	}
	else
	{
		file_path = "";
	}
	size = file_path.rfind(".", file_path.length());
	
	if (size != std::string::npos)
	{
		return file_path.substr(0, size);
	}
	else
	{
		return file_path;
	}
}