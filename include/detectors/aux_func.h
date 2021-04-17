#ifndef AUX_FUNC_H
#define AUX_FUNC_H

#include "core.h"

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name);

std::string getFileName(std::string file_path);

void imgNormalize(cv::Mat img, cv::Mat &img_norm);

#endif
