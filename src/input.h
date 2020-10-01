std::string getFolderName(std::string path);

void readData(char *img_path, char *dtset_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, cv::Mat roi[]);

std::string getFileName(std::string file_path);