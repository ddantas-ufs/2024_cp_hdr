#include "../include/cphdr.h"

int main(int argv, char** args)
{	
  cv::Mat img_in, img_gray;
  std::vector<KeyPoints> kp;
  std::string img_name, out_path;
  
  readImg(args[1], img_in, img_gray, img_name, true);
  std::cout << "Image name: " << img_name << std::endl; 
  out_path = std::string(args[2]) + img_name + ".dog";
  
  dogKp(img_gray, kp);
  std::cout << "Quantidade de KeyPoints lidos:" << kp.size() << std::endl; 
  saveKeypoints(kp, out_path, kp.size()); // saving detected keypoints

  siftDescriptor(kp, img_in, img_gray);
  saveKeypoints(kp, out_path, kp.size()); // saving descripted keypoints
  plotKeyPoints(img_in, kp, out_path, kp.size());
  
  img_gray.release();
  img_in.release();
  return 0;
}
