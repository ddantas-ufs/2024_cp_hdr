#include "../include/cphdr.h"

int main(int, char** argv)
{
  cv::Mat img_in;
  cv::Mat img_gray;
  cv::Mat roi[4];
  std::vector<KeyPoints> kp;
  std::string img_name;
  
  readImg(argv[1], img_in, img_gray, img_name);
  std::string out_path = std::string(argv[2]) + img_name + ".surf";
  
  return 0;
}