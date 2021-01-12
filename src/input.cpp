#include "../include/detectors/input.h"

std::string getFolderName(std::string path)
{
	std::string name = "";
	
	if(path.back() == '/')
		path.pop_back();

	while(true)
	{
		if(path.back() == '/')
			break;

		name += path.back();
		path.pop_back();
	}
	std::reverse(name.begin(), name.end());
	
	return name;
}

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name)
{
	img_in = cv::imread(img_path, cv::IMREAD_UNCHANGED);

	if(img_in.channels() != 1)
		cv::cvtColor(img_in, img_gray, cv::COLOR_BGR2GRAY);

	if(img_in.depth() == CV_32F)
		cv::normalize(img_in, img_gray, 0.0, 256.0, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	img_name = getFileName(std::string(img_path));
}

void readRoi(char *dtset_path, cv::Mat roi[], cv::Size img_size)
{
	if(dtset_path != NULL)
	{
		std::string path(dtset_path);
		std::string folder = getFolderName(path);
		
		roi[0] = imread(path + "ROI." + folder + ".png", cv::IMREAD_UNCHANGED);
		roi[1] = imread(path + "ROIh." + folder + ".png", cv::IMREAD_UNCHANGED);
		roi[2] = imread(path + "ROIm." + folder + ".png", cv::IMREAD_UNCHANGED);
		roi[3] = imread(path + "ROIs." + folder + ".png", cv::IMREAD_UNCHANGED);	
	}
	else
	{
		roi[0] = cv::Mat::ones(img_size, CV_8U);
		roi[1] = cv::Mat::ones(img_size, CV_8U);
		roi[2] = cv::Mat::zeros(img_size, CV_8U);
		roi[3] = cv::Mat::zeros(img_size, CV_8U);
	}
}

std::string getFileName(std::string file_path)
{
	// OBTAINING COMPLETE FILENAME
	size_t size = file_path.rfind("/", file_path.length());
	
	if (size != std::string::npos) {
		file_path = file_path.substr(size+1, file_path.length() - size);
	} else {
		file_path = "";
	}
	
	// REMOVING EXTENSION
	size = file_path.rfind(".", file_path.length());
	
	if (size != std::string::npos) {
		return file_path.substr(0, size);
	} else {
		return file_path;
	}
}

