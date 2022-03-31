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
  std::vector< std::vector<KeyPoints> > lKp1, lKp2; // CP_HDR KeyPoint list
  std::vector<KeyPoints> kps1, kps2; // CP_HDR KeyPoint list
  std::vector<cv::Mat> img1AllROIs, img2AllROIs; // ROI lists

  cv::Mat img1, img2, img1Gray, img2Gray, img1Out, img2Out, imgMatching; // Images
  cv::Mat img1ROIs, img1ROIm, img1ROIh, img2ROIs, img2ROIm, img2ROIh; // ROI images
  cv::Mat H; // Homography matrix

  std::string img1Path, img1ROIsPath, img1ROImPath, img1ROIhPath; // image 1 input path strings
  std::string img2Path, img2ROIsPath, img2ROImPath, img2ROIhPath; // input 2 input path strings
  std::string img1OutPath, img2OutPath, imgMatchingOutPath, outDir; // output path strings

  std:: string hdrSuf = ".hdr", pathH;
  bool isHDR = false;
//  bool considerROI = false;

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

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

  readHomographicMatrix( pathH, H );

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
  
  matchFPs(img1, img2, kps1, kps2, imgMatching);

  // Matching and generating output image with matches
  std::cout << "> Matching CP_HDR FPs and saving resulting image" << std::endl;

  cv::imwrite(imgMatchingOutPath, imgMatching);

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
  H.release();

  /* OPENCV PART
  // Algorithms that doesn't support HDR images
  if(!isHDR)
  {
    cv::Mat ocvDesc1, ocvDesc2; // OpenCV Descriptors
    std::vector<cv::KeyPoint> ocvKPs1, ocvKPs2;

    // Reading images and setting output image name
    readImg(img1Path, img1, img1Gray, img1OutPath);
    readImg(img2Path, img2, img2Gray, img2OutPath);

    if( considerROI )
    {
      std::cout << "> Running OpenCV SIFT [WITH ROI]..." << std::endl;

      // Computing Keypoints using OpenCV SIFT with ROI
      cv::Ptr<cv::SIFT> siftImage1 = cv::SIFT::create();
      siftImage1->detect( img1Gray, ocvKPs1, img1ROI );
      siftImage1->compute( img1Gray, ocvKPs1, ocvDesc1);

      cv::Ptr<cv::SIFT> siftImage2 = cv::SIFT::create();
      siftImage2->detect( img2Gray, ocvKPs2, img2ROI );
      siftImage2->compute( img2Gray, ocvKPs2, ocvDesc2);
    }
    else
    {
      std::cout << "> Running OpenCV SIFT [WITHOUT ROI]..." << std::endl;

      // Computing Keypoints using OpenCV SIFT without ROI
      cv::Ptr<cv::SIFT> siftImage1 = cv::SIFT::create();
      siftImage1->detect( img1Gray, ocvKPs1 );
      siftImage1->compute( img1Gray, ocvKPs1, ocvDesc1);

      cv::Ptr<cv::SIFT> siftImage2 = cv::SIFT::create();
      siftImage2->detect( img2Gray, ocvKPs2 );
      siftImage2->compute( img2Gray, ocvKPs2, ocvDesc2);      
    }
 
    loadOpenCVKeyPoints( ocvKPs1, ocvDesc1, kp1 );
    loadOpenCVKeyPoints( ocvKPs2, ocvDesc2, kp2 );

    // Getting only the strongest keypoints
    sortKeypoints( kp1 );
    sortKeypoints( kp2 );
    kp1 = vectorSlice( kp1, 0, MAX_KP);
    kp2 = vectorSlice( kp2, 0, MAX_KP);

    saveKeypoints( kp1, outDir+img1OutPath+"_OpenCV_SIFT_LDR.txt", kp1.size() );
    saveKeypoints( kp2, outDir+img2OutPath+"_OpenCV_SIFT_LDR.txt", kp2.size() );
    plotKeyPoints( img1, kp1, outDir+img1OutPath+"_OpenCV_SIFT_LDR.png", kp1.size() );
    plotKeyPoints( img2, kp2, outDir+img2OutPath+"_OpenCV_SIFT_LDR.png", kp2.size() );
  
    matchFPs(img1, img2, kp1, kp2, imgMatching);

    // Matching and generating output image with matches
    std::cout << "> Matching OpenCV FPs and saving resulting image..." << std::endl;
    imgMatchingOutPath = outDir +img1OutPath + "_"+ img2OutPath +"_OpenCV_SIFT.png";

    cv::imwrite(imgMatchingOutPath, imgMatching);

    ocvDesc1.release();
    ocvDesc2.release();
    ocvKPs1.clear();
    ocvKPs2.clear();

    // Cleaning objects
    cleanKeyPointVector( kp1 );
    cleanKeyPointVector( kp2 );
    img1.release();
    img2.release();
    img1Out.release();
    img2Out.release();
    img1Gray.release();
    img2Gray.release();
    imgMatching.release();
  }*/

  return 0;
}