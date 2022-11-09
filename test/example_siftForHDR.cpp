#include "../include/cphdr.h"

/**
 * Compare CP_HDR with other algorithms
 * 
 * @param arg[1]: Image 1 to matching path
 * @param arg[2]: Image 1 ROIs
 * @param arg[3]: Image 1 ROIm
 * @param arg[4]: Image 1 ROIh
 * @param arg[5]]: output directory path
**/
int main(int argv, char** args)
{
  std::vector<MatchedKeyPoints> matchings;
  std::vector< std::vector<KeyPoints> > lKp1; // CP_HDR KeyPoint list
  std::vector<KeyPoints> kps1; // CP_HDR KeyPoint list
  std::vector<cv::Mat> img1AllROIs;

  cv::Mat img1, img1Gray, img1Out;
  cv::Mat img1ROIs, img1ROIm, img1ROIh; // ROI images

  std::string img1Path, img1ROIsPath, img1ROImPath, img1ROIhPath; // image 1 input path strings
  std::string img1OutPath, outDir; // output path strings

  std:: string hdrSuf = ".hdr", pathH;
  bool isHDR = false;

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

  img1Path     = std::string(args[1]);
  img1ROIsPath = std::string(args[2]);
  img1ROImPath = std::string(args[3]);
  img1ROIhPath = std::string(args[4]);
  outDir       = std::string(args[5]);

  // Evaluating if image is LDR or HDR
  if( 0 == img1Path.compare(img1Path.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
    isHDR = true;

  // Reading images and setting output image name
  readImg(img1Path, img1, img1Gray, img1OutPath);

  readROIFromImage( img1ROIsPath, img1ROIs );
  readROIFromImage( img1ROImPath, img1ROIm );
  readROIFromImage( img1ROIhPath, img1ROIh );
  
  img1AllROIs.push_back(img1ROIs);
  img1AllROIs.push_back(img1ROIm);
  img1AllROIs.push_back(img1ROIh);

  // Running CP_HDR
  std::cout << "> Running CP_HDR siftForHDR..." << std::endl;
  runSift(img1Gray, lKp1, MAX_KP, img1AllROIs, true);

  joinKeypoints( lKp1, kps1 );

  // Getting only the MAX_KP strongest keypoints
  sortKeypoints( kps1 );
  kps1 = vectorSlice( kps1, 0, 2000);

  if( isHDR )
  {
    saveKeypoints( kps1, outDir+img1OutPath+"_CPHDR_siftForHDR_HDR.txt", kps1.size() );
    plotKeyPoints( img1, kps1, outDir+img1OutPath+"_CPHDR_siftForHDR_HDR.hdr", kps1.size() );
  }
  else
  {
    saveKeypoints( kps1, outDir+img1OutPath+"_CPHDR_siftForHDR_LDR.txt", kps1.size() );
    plotKeyPoints( img1, kps1, outDir+img1OutPath+"_CPHDR_siftForHDR_LDR.png", kps1.size() );
  }

  // Cleaning objects
  cleanKeyPointVector( kps1 );
  img1.release();
  img1Out.release();
  img1Gray.release();
  img1Out.release();
  img1ROIs.release();
  img1ROIm.release();
  img1ROIh.release();

  return 0;
}