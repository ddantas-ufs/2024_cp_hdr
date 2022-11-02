#include "../include/cphdr.h"

int main(int, char** args)
{
  cv::Mat img_in, img_gray;
  std::vector<KeyPoints> kp, kp_cv;
  std::string img_path, out_path, img_name, hdrSuf = ".hdr";
  std::string txt_out, txt_cv_out, img_out, img_cv_out;
  bool isHDR = false;

  img_path = std::string(args[1]);
  out_path = std::string(args[2]);

  // READING IMAGE
  readImg(img_path, img_in, img_gray, img_name);

  // EVALUATING IF INPUT IMAGE IS HDR
  if( 0 == img_path.compare(img_path.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
    isHDR = true;

  // OUTPUT USING SIFT DETECTOR
  txt_out = out_path + img_name + ".dog";
  img_out = out_path + img_name + ".dog.jpg";

  // OUTPUT USING SIFT FOR HDR DETECTOR
  txt_cv_out = out_path + img_name + ".dogForHDR";
  img_cv_out = out_path + img_name + ".dogForHDR.hdr";

  dogKp(img_gray, kp); // SIFT DETECTOR
  dogKp(img_gray, kp_cv, true); // SIFT FOR HDR DETECTOR

  // SAVING SIFT OUTPUT RESULTS
  saveKeypoints(kp, txt_out);
  plotKeyPoints(img_in, kp, img_out);

  // SAVING SIFT FOR HDR RESULTS
  saveKeypoints(kp_cv, txt_cv_out);
  plotKeyPoints(img_in, kp_cv, img_cv_out);

  return 0;
}
