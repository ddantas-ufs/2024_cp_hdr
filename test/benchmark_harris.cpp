#include "../include/cphdr.h"

int main(int argv, char** args)
{
  cv::Mat img_in, img_gray;
  cv::Mat img1ROIs, img1ROIm, img1ROIh;

  std::vector<KeyPoints> kp;
  std::vector< std::vector<KeyPoints> > lKps;
  std::vector<cv::Mat> imgAllROIs;

  std::string img_name, out_path, imgPath;
  std::string img1ROIsPath, img1ROImPath, img1ROIhPath;

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

  imgPath = std::string(args[1]);
  img1ROIsPath = std::string(args[2]);
  img1ROImPath = std::string(args[3]);
  img1ROIhPath = std::string(args[4]);

  readImg(imgPath, img_in, img_gray, img_name);
  out_path = std::string(args[5]) + img_name + ".harris";

  readROIFromImage( img1ROIsPath, img1ROIs );
  readROIFromImage( img1ROImPath, img1ROIm );
  readROIFromImage( img1ROIhPath, img1ROIh );

  imgAllROIs.push_back(img1ROIs);
  imgAllROIs.push_back(img1ROIm);
  imgAllROIs.push_back(img1ROIh);

  harrisKp(img_gray, lKps, imgAllROIs, false);

  joinKeypoints( lKps, kp );
  std::cout << " ############ Keypoints detected after ROI filtering:" << kp.size() << std::endl;

  saveKeypoints(kp, out_path, MAX_KP);
  plotKeyPoints(img_in, kp, out_path, MAX_KP);

  //saveKeypoints( kp, out_path, kp.size() );
  //plotKeyPoints( img_in, kp, out_path, kp.size() );

  return 0;
}
