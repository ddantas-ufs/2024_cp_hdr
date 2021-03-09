#include "../include/cphdr.h"

int main(int argv, char** args)
{
  for( int i = 0; i < argv; i++ ) 
  {
    std::cout << "arg[" << i << "]: " << args[i] << std::endl;
  }

  cv::Mat img_in, img_gray;
  std::vector<KeyPoints> kp;
  std::string img_name, out_path;

  readImg(args[1], img_in, img_gray, img_name);
  out_path = std::string(args[2]) + img_name + ".dog";
  std::cout << "out path: " << out_path << std::endl;

  dogKp(img_gray, kp);
  saveKeypoints(kp, out_path);
  plotKeyPoints(img_in, kp, out_path);

  return 0;
}
