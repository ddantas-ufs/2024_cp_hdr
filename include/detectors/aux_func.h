#ifndef AUX_FUNC_H
#define AUX_FUNC_H

#include "core.h"

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name);

std::string getFileName(std::string file_path);

int sciToDec(const std::string &str);

std::vector<std::string> split(const std::string& s, char delimiter = '\t');

void readROI(std::string roi_path, std::vector<cv::Point> &verts);

void selectROI(cv::Mat img, cv::Mat &img_roi, cv::Point v1, cv::Point v2);

void gaussKernel(cv::Mat &kernel, int size = 5, float sigma = 1.0);

void imgNormalize(cv::Mat img, cv::Mat &img_norm);

#endif