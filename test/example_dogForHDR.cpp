#include "../include/cphdr.h"

int main(int argv, char** args)
{
  cv::Mat img_in, img_gray1;
  std::vector<KeyPoints> kp, kp_cv;
  std::string img_path, out_path, img_name, hdrSuf = ".hdr";
  std::string txt_out, img_out;
  bool isHDR;

  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

  img_path = std::string(args[1]);
  out_path = std::string(args[2]);

  // READING IMAGE
  readImg(img_path, img_in, img_gray1, img_name);

  // EVALUATING IF INPUT IMAGE IS HDR
  if( 0 == img_path.compare(img_path.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
  {
    std::cout << " > Input image is HDR" << std::endl;
    isHDR = true;

    out_path += "HDR_";
    img_out = out_path + img_name + ".dogForHDR.hdr";
  }
  else
  {
    std::cout << " > Input image is LDR" << std::endl;

    out_path += "LDR_";
    img_out = out_path + img_name + ".dogForHDR.jpg";
  }

  // OUTPUT USING SIFT DETECTOR
  txt_out = out_path + img_name + ".dogForHDR";

  std::cout << " > Running DoG for HDR" << std::endl;
  dogKp(img_gray1, kp, true); // SIFT DETECTOR

  // SAVING SIFT OUTPUT RESULTS
  saveKeypoints(kp, txt_out, 2000);
  plotKeyPoints(img_in, kp, img_out, 2000);

  return 0;
}
