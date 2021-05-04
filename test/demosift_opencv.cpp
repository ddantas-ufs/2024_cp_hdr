#include <opencv2/features2d.hpp>
#include "../include/cphdr.h"

int main(int argv, char** args)
{
  std::cout << "OpenCV version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;

  cv::Mat img_in, img_gray, descriptor;
  std::vector<KeyPoints> kp;
  std::string img_name, out_path;

  std::cout << "argv: " << argv << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "i: " << i << " " << args[i] << std::endl;

  std::vector<cv::KeyPoint> ocv_kp; // OpenCV KeyPoints
  
  readImg(args[1], img_in, img_gray, img_name);
  out_path = std::string(args[2]) + img_name + ".dog_opencv";
  
  //dogKp(img_gray, kp);
  //std::cout << "Quantidade de KeyPoints lidos:" << kp.size() << std::endl;

  // nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma
  cv::Ptr<cv::SIFT> siftObj = cv::SIFT::create(0, 3, 0.03, 10, 1.6 );
  siftObj->detect( img_gray, ocv_kp );
  siftObj->compute( img_gray, ocv_kp, descriptor);

  std::cout << "########################################" << std::endl;
  std::cout << "KeyPoints vector size: " << ocv_kp.size() << std::endl;
  std::cout << "descriptor size      : " << descriptor.size() << std::endl;
  std::cout << "########################################" << std::endl;

  // converting cv::KeyPoint to KeyPoints
  for(int i=0; i<ocv_kp.size(); i++)
  {
    //cv::KeyPoint ckp = ocv_kp[i];
    KeyPoints nkp;

    nkp.x = (float) ocv_kp[i].pt.x;
    nkp.y = (float) ocv_kp[i].pt.y;
    nkp.resp = (float) ocv_kp[i].response;
    nkp.octave = (int) ocv_kp[i].octave;
    nkp.direction = (float) ocv_kp[i].angle;

    //std::cout << "Px: " << ocv_kp[i].pt.x << std::endl;
    //std::cout << "Py: " << ocv_kp[i].pt.y << std::endl;
    //std::cout << "Re: " << ocv_kp[i].response << std::endl;
    //std::cout << "Oc: " << ocv_kp[i].octave << std::endl;
    //std::cout << "An: " << ocv_kp[i].angle << std::endl;
    //std::cout << "De: ";

    for( int j=0; j < 128; j++ )
    {
      float d = descriptor.at<float>(j, i);
      //std::cout << "i: " << i << " j: " << j << std::endl;
      nkp.descriptor.push_back( (int) d );
    }
    kp.push_back(nkp);
    //std::cout << std::endl << "########################################" << std::endl;
  }

  // Add results to image and save.
  cv::Mat img_out;
  cv::drawKeypoints(img_in, ocv_kp, img_out);
  cv::imwrite(out_path+".png", img_out);
  saveKeypoints(kp, out_path, false);

  //cv::
  /*
  //siftDescriptor(kp, img_in, img_gray);
  //saveKeypoints(kp, out_path);
  plotKeyPoints(img_in, kp, out_path);
  
  img_gray.release();
  img_in.release();
  */
  return 0;
}
