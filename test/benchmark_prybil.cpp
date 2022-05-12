#include "../include/cphdr.h"

/**
 * Compare CP_HDR with other algorithms
 * 
 * @param arg[1]: Image 1 to matching path
 * @param arg[2]: Image 1 ROIs
 * @param arg[3]: Image 1 ROIm
 * @param arg[4]: Image 1 ROIh
 * @param arg[5]: Image 2 to matching path
 * @param arg[6]: Image 2 ROIs
 * @param arg[7]: Image 2 ROIm
 * @param arg[8]: Image 2 ROIh
 * @param arg[9]: Homography Matrix path
 * @param arg[10]: output directory path
**/
int main(int argv, char** args)
{
  std::vector<MatchedKeyPoints> matchings;
  std::vector< std::vector<KeyPoints> > lKp1, lKp2; // CP_HDR KeyPoint list
  std::vector<KeyPoints> kps1, kps2; // CP_HDR KeyPoint list
  std::vector<cv::Mat> img1AllROIs, img2AllROIs; // ROI lists

  cv::Mat img1, img2, img1Gray, img2Gray, img1Out, img2Out, imgMatching; // Images
  cv::Mat img1ROIs, img1ROIm, img1ROIh, img2ROIs, img2ROIm, img2ROIh; // ROI images
  cv::Mat H; // Homography matrix

  std::string img1Path, img1ROIsPath, img1ROImPath, img1ROIhPath; // image 1 input path strings
  std::string img2Path, img2ROIsPath, img2ROImPath, img2ROIhPath; // input 2 input path strings
  std::string img1OutPath, img2OutPath, imgMatchingOutPath, outDir; // output path strings
//  std::string finalOut = "Image 1;Image 2;Keypoints Matched;Repeatability;Uniformity\n";

  std:: string hdrSuf = ".hdr", pathH;
  bool isHDR = false;
//  bool considerROI = false;

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

  if( argv == 11 )
  { // COMPLETE
    img1Path     = std::string(args[1]);
    img1ROIsPath = std::string(args[2]);
    img1ROImPath = std::string(args[3]);
    img1ROIhPath = std::string(args[4]);
    img2Path     = std::string(args[5]);
    img2ROIsPath = std::string(args[6]);
    img2ROImPath = std::string(args[7]);
    img2ROIhPath = std::string(args[8]);
    pathH        = std::string(args[9]);
    outDir       = std::string(args[10]);
  } 
  else if( argv == 10 )
  { // NO HOMOGRAPHY MATRIX
    img1Path     = std::string(args[1]);
    img1ROIsPath = std::string(args[2]);
    img1ROImPath = std::string(args[3]);
    img1ROIhPath = std::string(args[4]);
    img2Path     = std::string(args[5]);
    img2ROIsPath = std::string(args[6]);
    img2ROImPath = std::string(args[7]);
    img2ROIhPath = std::string(args[8]);
    outDir       = std::string(args[9]);
  }
  /*
  KeyPoints A, B;
  A.x = 500.0f;
  A.y = 500.0f;
  B.y = 500.0f;

  B.x = 530.0f;
  std::cout << " IoU A(500,500), B(530,500): " << calculateIoU(A, B) << std::endl;
  B.x = 525.0f;
  std::cout << " IoU A(500,500), B(525,500): " << calculateIoU(A, B) << std::endl;
  B.x = 520.0f;
  std::cout << " IoU A(500,500), B(520,500): " << calculateIoU(A, B) << std::endl;
  B.x = 515.0f;
  std::cout << " IoU A(500,500), B(515,500): " << calculateIoU(A, B) << std::endl;
  B.x = 510.0f;
  std::cout << " IoU A(500,500), B(510,500): " << calculateIoU(A, B) << std::endl;
  B.x = 505.0f;
  std::cout << " IoU A(500,500), B(505,500): " << calculateIoU(A, B) << std::endl;
  B.x = 504.0f;
  std::cout << " IoU A(500,500), B(504,500): " << calculateIoU(A, B) << std::endl;
  B.x = 503.0f;
  std::cout << " IoU A(500,500), B(503,500): " << calculateIoU(A, B) << std::endl;
  B.x = 502.0f;
  std::cout << " IoU A(500,500), B(502,500): " << calculateIoU(A, B) << std::endl;
  B.x = 501.0f;
  std::cout << " IoU A(500,500), B(501,500): " << calculateIoU(A, B) << std::endl;
  B.x = 500.0f;
  std::cout << " IoU A(500,500), B(500,500): " << calculateIoU(A, B) << std::endl;
  return 0;
  */

  // Evaluating if image is LDR or HDR
  if( 0 == img1Path.compare(img1Path.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
    isHDR = true;

  // Reading images and setting output image name
  readImg(img1Path, img1, img1Gray, img1OutPath);
  readImg(img2Path, img2, img2Gray, img2OutPath);

  readROIFromImage( img1ROIsPath, img1ROIs );
  readROIFromImage( img1ROImPath, img1ROIm );
  readROIFromImage( img1ROIhPath, img1ROIh );
  readROIFromImage( img2ROIsPath, img2ROIs );
  readROIFromImage( img2ROImPath, img2ROIm );
  readROIFromImage( img2ROIhPath, img2ROIh );

  if( !pathH.empty() )
  {
    readHomographicMatrix( pathH, H );
  }

  img1AllROIs.push_back(img1ROIs);
  img1AllROIs.push_back(img1ROIm);
  img1AllROIs.push_back(img1ROIh);
  img2AllROIs.push_back(img2ROIs);
  img2AllROIs.push_back(img2ROIm);
  img2AllROIs.push_back(img2ROIh);

  /*
  cv::Mat sumROI1 = cv::Mat::zeros( img1ROIs.size(), img1ROIs.type() );
  cv::Mat sumROI2 = cv::Mat::zeros( img2ROIs.size(), img2ROIs.type() );

  cv::add( sumROI1, img1ROIs, sumROI1 );
  cv::add( sumROI1, img1ROIm, sumROI1 );
  cv::add( sumROI1, img1ROIh, sumROI1 );
  cv::add( sumROI2, img2ROIs, sumROI2 );
  cv::add( sumROI2, img2ROIm, sumROI2 );
  cv::add( sumROI2, img2ROIh, sumROI2 );

  cv::imwrite("out/sumROI1.png", sumROI1);
  cv::imwrite("out/img1ROIs.png", img1ROIs);
  cv::imwrite("out/img1ROIm.png", img1ROIm);
  cv::imwrite("out/img1ROIh.png", img1ROIh);
  cv::imwrite("out/sumROI2.png", sumROI2);
  cv::imwrite("out/img2ROIs.png", img2ROIs);
  cv::imwrite("out/img2ROIm.png", img2ROIm);
  cv::imwrite("out/img2ROIh.png", img2ROIh);

  std::cout << "Homography Matrix: " << std::endl;
  std::cout << H << std::endl;
  std::cout << "Output Directory: " << std::endl;
  std::cout << outDir << std::endl;
  */
//  return 0;/*

  // Normalizing images (mandatory to HDR images).
  if( isHDR )
  {
    mapPixelValues( img1, img1 );
    mapPixelValues( img2, img2 );
  }

  // Running CP_HDR
  std::cout << "> Running CP_HDR SIFT..." << std::endl;
  runSift(img1Gray, lKp1, MAX_KP, img1AllROIs);
  runSift(img2Gray, lKp2, MAX_KP, img2AllROIs);

  joinKeypoints( lKp1, kps1 );
  joinKeypoints( lKp2, kps2 );

  // Getting only the MAX_KP strongest keypoints
  sortKeypoints( kps1 );
  sortKeypoints( kps2 );
  kps1 = vectorSlice( kps1, 0, MAX_KP);
  kps2 = vectorSlice( kps2, 0, MAX_KP);

  if( isHDR )
  {
    saveKeypoints( kps1, outDir+img1OutPath+"_CPHDR_SIFT_HDR.txt", kps1.size() );
    saveKeypoints( kps2, outDir+img2OutPath+"_CPHDR_SIFT_HDR.txt", kps2.size() );
    plotKeyPoints( img1, kps1, outDir+img1OutPath+"_CPHDR_SIFT_HDR.hdr", kps1.size() );
    plotKeyPoints( img2, kps2, outDir+img2OutPath+"_CPHDR_SIFT_HDR.hdr", kps2.size() );
    imgMatchingOutPath = outDir +img1OutPath + "_"+ img2OutPath +"_CPHDR_SIFT.hdr";
  }
  else
  {
    saveKeypoints( kps1, outDir+img1OutPath+"_CPHDR_SIFT_LDR.txt", kps1.size() );
    saveKeypoints( kps2, outDir+img2OutPath+"_CPHDR_SIFT_LDR.txt", kps2.size() );
    plotKeyPoints( img1, kps1, outDir+img1OutPath+"_CPHDR_SIFT_LDR.png", kps1.size() );
    plotKeyPoints( img2, kps2, outDir+img2OutPath+"_CPHDR_SIFT_LDR.png", kps2.size() );
    imgMatchingOutPath = outDir +img1OutPath + "_"+ img2OutPath +"_CPHDR_SIFT.png";
  }
  
  matchFPs(img1, kps1, img2, kps2, H, matchings, imgMatching);
  
  int trueMatches = 0, falseMatches = 0;
  for( int k = 0; k < matchings.size(); k++ )
  {
    if( matchings[k].isCorrect ) trueMatches++;
    else falseMatches++;
  }

  std::cout << "> ## Calculating metrics" << std::endl;
  float rr = 0.0f;
  int cc = 0;
  calculateRR(H, kps1, kps2, cc, rr);
  std::cout << "> ##############################" << std::endl;
  std::cout << "> Repeatability Rate: " << rr << std::endl;
  std::cout << "> Correspondence total: " << cc << std::endl;
  std::cout << "> ##############################" << std::endl;

  float U1 = 0.0f, U2 = 0.0f; 
  
  U1 = calculateUniformity( lKp1 );
  U2 = calculateUniformity( lKp2 );

  std::cout << "> Uniformity img1: " << U1 << std::endl;
  std::cout << "> Uniformity img2: " << U2 << std::endl;
  std::cout << "> ##############################" << std::endl;

  float AP = calculateAP(matchings, H);

  std::cout << "> AP img1-img2: " << AP << std::endl;
  std::cout << "> ##############################" << std::endl;

  // Matching and generating output image with matches
  std::cout << "> Matching CP_HDR FPs and saving resulting image" << std::endl;

  cv::imwrite(imgMatchingOutPath, imgMatching);

  // GENERATING CSV OUTPUT FILE
  std::string finalOut = "Image 1;Image 2;Uniformity 1;Uniformity 2;Correct Matches;Incorrect Matches;% Correct Matches;Repeatability;Average Precision\n";
  finalOut += img1OutPath +";" +img2OutPath +";" +std::to_string(U1) +";" +std::to_string(U2) +";" +std::to_string(trueMatches) +";" +std::to_string(falseMatches) +";" +std::to_string(0) +";" +std::to_string(rr) +";" +std::to_string(AP);

  writeTextFile( outDir + img1OutPath + "_"+ img2OutPath +".csv", finalOut );

  // Cleaning objects
  cleanKeyPointVector( kps1 );
  cleanKeyPointVector( kps2 );
  img1.release();
  img2.release();
  img1Out.release();
  img2Out.release();
  img1Gray.release();
  img2Gray.release();
  img1Out.release();
  img2Out.release();
  imgMatching.release();
  img1ROIs.release();
  img1ROIm.release();
  img1ROIh.release();
  img2ROIs.release();
  img2ROIm.release();
  img2ROIh.release();
  if( !H.empty() ) H.release();

  return 0;
}