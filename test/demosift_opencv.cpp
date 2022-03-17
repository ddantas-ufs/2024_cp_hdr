#include "../include/cphdr.h"

int main(int argv, char** args)
{
  std::cout << "OpenCV version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;

  cv::Mat img_in, img_gray, descriptor, descriptor2;
  std::vector<KeyPoints> kpListOcv, kpList2, kpList3, loweKeypoints;
  std::string img_name, out_path;

  std::cout << "argv: " << argv << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "i: " << i << " " << args[i] << std::endl;
  
  if(argv > 3 )
  {
    std::cout << " Reading Lowe's Keypoint File... " << std::endl;
    loweKeypoints = loadLoweKeypoints(args[3]);
    //for(int i=0; i<loweKeypoints.size(); i++) printKeypoint(loweKeypoints[i]);
  }

  std::vector<cv::KeyPoint> ocv_kp; // OpenCV KeyPoints
  
  readImg(args[1], img_in, img_gray, img_name);
  out_path = std::string(args[2]) + img_name;// + ".dog_opencv";
  
  std::cout << " Computing with OpenCV Detector and Descriptor... " << std::endl;
  // nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma
  cv::Ptr<cv::SIFT> siftObj = cv::SIFT::create(0, 3, 0.03, 10, 1.6 );
  siftObj->detect( img_gray, ocv_kp );
  siftObj->compute( img_gray, ocv_kp, descriptor);

  std::cout << "########################################" << std::endl;
  std::cout << "KeyPoints vector size: " << ocv_kp.size() << std::endl;
  std::cout << "descriptor size      : " << descriptor.size() << std::endl;
  std::cout << "########################################" << std::endl;

//  importOpenCVKeyPoints(ocv_kp, descriptor, kpListOcv, true);
  loadOpenCVKeyPoints( ocv_kp, descriptor, kpListOcv );

  std::cout << " Saving OpenCV Keypoints... " << std::endl;
  // Saving results with OpenCV Descriptor
  cv::Mat img_out;
  cv::drawKeypoints(img_in, ocv_kp, img_out);
  cv::imwrite(out_path+".dog_opencv.png", img_out);
  saveKeypoints(kpListOcv, out_path+".dog_opencv", false);

  //for(int k=0; k < kpList.size(); k++ ) kpList[k].descriptor.clear();
  readImg(args[1], img_in, img_gray, img_name);
//  importOpenCVKeyPoints(ocv_kp, descriptor, kpList2, false);
  loadOpenCVKeyPoints( ocv_kp, kpList2 );

  std::cout << " Computing with our descriptor and OpenCV detector... " << std::endl;
  // Calculating description and saving it with our algorithm
  siftDescriptor(kpList2, img_in, img_gray);
  saveKeypoints(kpList2, out_path, false);

  std::cout << " .................................................... " << std::endl;
  std::cout << " Using Lowe detector and OpenCV descriptor... " << std::endl;
  readImg(args[1], img_in, img_gray, img_name);
  cv::Mat mask = cv::Mat::ones(3,3, img_gray.depth()); // 3x3 mask of ones to not interfere
  std::vector<cv::KeyPoint> ocv_kp2; // OpenCV KeyPoints

  // TRYING TO USE OPENCV DESCRIPTOR
  exportToOpenCVKeyPointsObject( loweKeypoints, ocv_kp2 );
  std::cout << " Keypoints Loaded into OpenCV object... " << std::endl;

  // COMPUTING
  cv::Ptr<cv::SIFT> siftObj2 = cv::SIFT::create(0, 3, 0.03, 10, 1.6 );
  //siftObj2->detectAndCompute(img_gray, mask, ocv_kp2, descriptor2, true );
  siftObj->compute( img_gray, ocv_kp2, descriptor2);

  // SAVING RESULTS
//  importOpenCVKeyPoints(ocv_kp2, descriptor2, kpList3, true);
  loadOpenCVKeyPoints( ocv_kp2, descriptor2, kpList3 );
  saveKeypoints(kpList3, out_path+".dog_loweDetector_ocvDesc", false);
  
  return 0;
}
