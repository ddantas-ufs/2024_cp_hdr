#include "../include/cphdr.h"

/**
 * Compare CP_HDR with other algorithms
 * 
 * @param arg[1]: Image to matching path
 * @param arg[2]: Keypoints file
 * @param arg[3]: Output file
 * @param arg[4]: output directory path
**/
int main(int argv, char** args)
{
  cv::Mat img, imgGray, imgOut;
  std::vector<KeyPoints> kps;

  std::string imgPath     = std::string(args[1]);
  std::string kpsPath     = std::string(args[2]);
  std::string imgOutPath  = std::string(args[3]);
  std::string outDir      = std::string(args[4]);
  std::string aux         = "";

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

  readImg(imgPath, img, imgGray, aux);
  kps = loadKeypoints( kpsPath );

  std::cout << "Tamanho: " << kps.size() << std::endl;

  plotKeyPoints( img, kps, outDir+imgOutPath, kps.size() );

  return 0;
}