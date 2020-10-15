#ifndef INPUT_H
#define INPUT_H

#include "core.h"

std::string getFolderName(std::string path);

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name);
void readRoi(char *dtset_path, cv::Mat roi[], cv::Size img_size);

std::string getFileName(std::string file_path);

#endif