#include <opencv2/features2d.hpp>
#include "../include/cphdr.h"

int main(int argv, char** args)
{
  std::cout << "OpenCV version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;

  cv::Mat img_in, img_gray, descriptor;
  std::vector<KeyPoints> kpListOcv, kpList2;
  std::string img_name, out_path;

  std::cout << "argv: " << argv << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "i: " << i << " " << args[i] << std::endl;
  
  if(argv > 3 )
  {
    std::cout << " Reading Lowe's Keypoint File... " << std::endl;
    std::vector<KeyPoints> loweKeypoints = loadLoweKeypoints(args[3]);
    for(int i=0; i<loweKeypoints.size(); i++) printKeypoint(loweKeypoints[i]);
  }

  std::vector<cv::KeyPoint> ocv_kp; // OpenCV KeyPoints
  
  readImg(args[1], img_in, img_gray, img_name);
  out_path = std::string(args[2]) + img_name;// + ".dog_opencv";
  
  // nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma
  cv::Ptr<cv::SIFT> siftObj = cv::SIFT::create(0, 3, 0.03, 10, 1.6 );
  siftObj->detect( img_gray, ocv_kp );
  siftObj->compute( img_gray, ocv_kp, descriptor);

  std::cout << "########################################" << std::endl;
  std::cout << "KeyPoints vector size: " << ocv_kp.size() << std::endl;
  std::cout << "descriptor size      : " << descriptor.size() << std::endl;
  std::cout << "########################################" << std::endl;

  importOpenCVKeyPoints(ocv_kp, descriptor, kpListOcv, true);

  // Saving results with OpenCV Descriptor
  cv::Mat img_out;
  cv::drawKeypoints(img_in, ocv_kp, img_out);
  cv::imwrite(out_path+".dog_opencv.png", img_out);
  saveKeypoints(kpListOcv, out_path+".dog_opencv", false);

  //for(int k=0; k < kpList.size(); k++ ) kpList[k].descriptor.clear();
  readImg(args[1], img_in, img_gray, img_name);
  importOpenCVKeyPoints(ocv_kp, descriptor, kpList2, false);

  // Calculating description and saving it with our algorithm
  siftDescriptor(kpList2, img_in, img_gray);
  saveKeypoints(kpList2, out_path, false);

  return 0;
}
