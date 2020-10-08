std::string getFolderName(std::string path);

void readImgData(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name);
void readRoiData(char *dtset_path, cv::Mat roi[], cv::Size img_size);

std::string getFileName(std::string file_path);