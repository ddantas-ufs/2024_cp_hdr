#include "../include/detectors/aux_func.h"

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name)
{
  img_in = cv::imread(img_path, cv::IMREAD_UNCHANGED);

  if (img_in.channels() != 1)
  {
    cv::cvtColor(img_in, img_gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    img_gray = img_in;
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

void imgNormalize(cv::Mat img, cv::Mat &img_norm)
{
  if (img.depth() == 0)
  {
    img.convertTo(img_norm, CV_32FC1);
    img_norm = img_norm / LDR_MAX_RANGE;
  }
  else
  {
    img_norm = img / HDR_MAX_RANGE;
  }
}
