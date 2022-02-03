#include "../include/cphdr.h"

/**
 * @param args[1] path to img1
 * @param args[2] path to img2
 * @param args[3] path to txt file with homography matrix (Pribyl-like expected)
 * @param args[4] output dir
**/
int main(int argv, char** args)
{
  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "Received " << argv << " arguments." << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  args[" << i << "]: " << args[i] << std::endl;
  
  std::string pathH = args[3], img1OutPath, img2OutPath, outImagePath;
  cv::Mat H, img1, img2, img1Gray, img2Gray, imgOut;

  readHomographicMatrix( pathH, H );

  readImg(args[1], img1, img1Gray, img1OutPath);
  readImg(args[2], img2, img2Gray, img2OutPath);
  printMat(H, "Homography Matrix");

  // y, x
  // 929.0000, 1286.0000
  cv::Point2f p1, p2;
  p1.y = 929.0000f;
  p1.x = 1286.0000f;
  getHomographicCorrespondence(p1, p2, H);
  std::cout << "----> P1: " << p1.x << ", " << p1.y << "." << std::endl;
  std::cout << "----> P2: " << p2.x << ", " << p2.y << "." << std::endl;

  outImagePath = args[4] +img1OutPath+"_H04.png";
  std::cout << "----> " << outImagePath << std::endl;

  getHomographicCorrespondence( img1, imgOut, H );
  cv::imwrite(outImagePath, imgOut);

  return 0;
}
