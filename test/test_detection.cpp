#include "../include/cphdr.h"

/**
 * Compare CP_HDR with other algorithms
 * 
 * @param arg[1]: Image 1 to matching path
 * @param arg[2]: Image 1 ROI [optional]
 * @param arg[3]: Image 2 to matching path
 * @param arg[4]: Image 2 ROI [optional]
 * @param arg[5]: Homography Matrix path
 * @param arg[6]: output directory path
**/
int main(int argv, char** args)
{
  std::vector<KeyPoints> kpDoG, kpHarris;
  std::vector< std::vector<KeyPoints> > lKpDoG, lKpHarris;

  cv::Mat img, imgGray, imgROI, imgOut;
  std::vector<cv::Mat> imgROIs;

  std::string imgOutPath, imgPath, outDir, ROIPath, hdrSuf = ".hdr";

  bool isHDR = false;
  bool considerROI = false;

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

  imgPath = std::string(args[1]);
  ROIPath = std::string(args[2]);
  outDir = std::string(args[3]);


  // Evaluating if image is LDR or HDR
  if( 0 == imgPath.compare(imgPath.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
  {
    isHDR = true;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>> HDR" << std::endl; 
  }
  else
  {
    isHDR = false;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>> LDR" << std::endl; 
  }

  readROIFromImage( ROIPath, imgROI );
  readImg(imgPath, img, imgGray, imgOutPath);
  mapPixelValues( img, img );

  imgROIs.push_back(imgROI);
  lKpDoG.push_back( kpDoG );
  lKpHarris.push_back( kpHarris );

  //cv::imwrite("out/sumROI1.png", imgROI);

  runSift(imgGray, lKpDoG, MAX_KP, imgROIs);
  harrisKp(imgGray, lKpHarris, imgROIs, false);

  joinKeypoints( lKpDoG, kpDoG );
  joinKeypoints( lKpHarris, kpHarris );

  sortKeypoints( kpDoG );
  sortKeypoints( kpHarris );
  kpDoG = vectorSlice( kpDoG, 0, MAX_KP);
  kpHarris = vectorSlice( kpHarris, 0, MAX_KP);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>> Quantidade de KPs SIFT  :" << kpDoG.size() << std::endl; 
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>> Quantidade de KPs Harris:" << kpHarris.size() << std::endl; 

  if( isHDR ) 
  {
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>> HDR" << std::endl; 
    plotKeyPoints( img, kpDoG, outDir+imgOutPath+"_HDR_CPHDR_SIFT.hdr", kpDoG.size() );
    plotKeyPoints( img, kpHarris, outDir+imgOutPath+"_HDR_CPHDR_Harris.hdr", kpHarris.size() );
  }
  else
  {
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>> LDR" << std::endl; 
    plotKeyPoints( img, kpDoG, outDir+imgOutPath+"_LDR_CPHDR_SIFT.png", kpDoG.size() );
    plotKeyPoints( img, kpHarris, outDir+imgOutPath+"_LDR_CPHDR_Harris.png", kpHarris.size() );
  }
  
  return 0;
}