#include "../include/cphdr.h"

/**
 * @brief 
 * 
 * @param args[1]: Path to image
 * @param args[2]: Path to output
 */
int main(int argv, char** args)
{
  cv::Mat img, imgGray, img_cv, img_log, out;
  std::string imgPath, outPath, imgName, hdrSuf = ".hdr";

  bool isHDR = false;

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

  imgPath = std::string(args[1]);
  outPath = std::string(args[2]);
  imgName = getFileName( imgPath, true );

  // Evaluating if image is LDR or HDR
  if( 0 == imgPath.compare(imgPath.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
    isHDR = true;

  // MAKING GRAYSCALE COPY
  readImg( imgPath, img );
  makeGrayscaleCopy( img, imgGray );
  
  mapPixelValues( imgGray, imgGray, MAPPING_INTERVAL_FLOAT_0_1 );
  cv::GaussianBlur( imgGray, imgGray, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);

  // LEGADO
  mapPixelValues( imgGray, imgGray, MAPPING_INTERVAL_FLOAT_0_1 );
  coefficienceOfVariationMask( imgGray, img_cv );
  logTranformUchar( img_cv, 2, img_log );
  mapPixelValues( img_log, out );

  // CP_HDR
  //applyCVMask( imgGray, img_cv );
  //logTransform( img_cv, img_log );
  //mapPixelValues( img_log, out );
  
  if( isHDR )
    cv::imwrite( outPath +imgName + ".jpg", out );
  else
    cv::imwrite( outPath +imgName, out );

  return 0;
}