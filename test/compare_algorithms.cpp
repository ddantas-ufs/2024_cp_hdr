#include "../include/cphdr.h"

/**
 * Compare CP_HDR with other algorithms
 * 
 * @param arg[1]: Image 1 to matching
 * @param arg[2]: Image 2 to matching
 * @param arg[3]: output directory
**/
int main(int argv, char** args)
{
  cv::Mat img1, img2, img1Gray, img2Gray, img1Out, img2Out, imgMatching;
  std::vector<KeyPoints> kp1, kp2; // CP_HDR KeyPoint list
  
  std::string img1OutPath, img2OutPath, imgMatchingOutPath;
  
  std::string imgPath = args[1], hdrSuf = ".hdr";
  std::string outDir = std::string(args[3]);

  bool isHDR = false;

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;
  
  // Reading images and setting output image name
  readImg(args[1], img1, img1Gray, img1OutPath);
  readImg(args[2], img2, img2Gray, img2OutPath);

  // Evaluating if image is LDR or HDR
  if( 0 == imgPath.compare(imgPath.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
    isHDR = true;

  // Normalizing images (mandatory to HDR images). 
  mapPixelValues( img1, img1 );
  mapPixelValues( img2, img2 );

  std::cout << "> Running CP_HDR SIFT..." << std::endl;

  // Running CP_HDR
  runSift(img1Gray, kp1);
  runSift(img2Gray, kp2);

  // Getting only the 500 strongest keypoints
  sortKeypoints( kp1 );
  sortKeypoints( kp2 );
  kp1 = vectorSlice( kp1, 0, 500);
  kp2 = vectorSlice( kp2, 0, 500);

  if( isHDR )
  {
    saveKeypoints( kp1, outDir+img1OutPath+"_CPHDR_SIFT_HDR.txt", kp1.size() );
    saveKeypoints( kp2, outDir+img2OutPath+"_CPHDR_SIFT_HDR.txt", kp2.size() );
    plotKeyPoints( img1, kp1, outDir+img1OutPath+"_CPHDR_SIFT_HDR.hdr", kp1.size() );
    plotKeyPoints( img2, kp2, outDir+img2OutPath+"_CPHDR_SIFT_HDR.hdr", kp2.size() );
    imgMatchingOutPath = outDir +img1OutPath + "_"+ img2OutPath +"_CPHDR_SIFT.hdr";
  }
  else
  {
    saveKeypoints( kp1, outDir+img1OutPath+"_CPHDR_SIFT_LDR.txt", kp1.size() );
    saveKeypoints( kp2, outDir+img2OutPath+"_CPHDR_SIFT_LDR.txt", kp2.size() );
    plotKeyPoints( img1, kp1, outDir+img1OutPath+"_CPHDR_SIFT_LDR.png", kp1.size() );
    plotKeyPoints( img2, kp2, outDir+img2OutPath+"_CPHDR_SIFT_LDR.png", kp2.size() );
    imgMatchingOutPath = outDir +img1OutPath + "_"+ img2OutPath +"_CPHDR_SIFT.png";
  }

  // Matching and generating output image with matches
  std::cout << "> Matching CP_HDR FPs and saving resulting image" << std::endl;
  matchFPs(img1, kp1, img2, kp2, imgMatching);
  cv::imwrite(imgMatchingOutPath, imgMatching);

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

  // Algorithms that doesn't support HDR images
  if(!isHDR)
  {
    cv::Mat ocvDesc1, ocvDesc2; // OpenCV Descriptors
    std::vector<cv::KeyPoint> ocvKPs1, ocvKPs2;

    // Reading images and setting output image name
    readImg(args[1], img1, img1Gray, img1OutPath);
    readImg(args[2], img2, img2Gray, img2OutPath);

    std::cout << "> Running OpenCV SIFT..." << std::endl;

    // COMPUTING KEYPOINTS USING SIFT
    cv::Ptr<cv::SIFT> siftImage1 = cv::SIFT::create();
    siftImage1->detect( img1Gray, ocvKPs1 );
    siftImage1->compute( img1Gray, ocvKPs1, ocvDesc1);

    cv::Ptr<cv::SIFT> siftImage2 = cv::SIFT::create();
    siftImage2->detect( img2Gray, ocvKPs2 );
    siftImage2->compute( img2Gray, ocvKPs2, ocvDesc2);

    loadOpenCVKeyPoints( ocvKPs1, ocvDesc1, kp1, true);
    loadOpenCVKeyPoints( ocvKPs2, ocvDesc2, kp2, true);

    // Getting only the 500 strongest keypoints
    sortKeypoints( kp1 );
    sortKeypoints( kp2 );
    kp1 = vectorSlice( kp1, 0, 500);
    kp2 = vectorSlice( kp2, 0, 500);

    saveKeypoints( kp1, outDir+img1OutPath+"_OpenCV_SIFT_LDR.txt", kp1.size() );
    saveKeypoints( kp2, outDir+img2OutPath+"_OpenCV_SIFT_LDR.txt", kp2.size() );
    plotKeyPoints( img1, kp1, outDir+img1OutPath+"_OpenCV_SIFT_LDR.png", kp1.size() );
    plotKeyPoints( img2, kp2, outDir+img2OutPath+"_OpenCV_SIFT_LDR.png", kp2.size() );


    // Matching and generating output image with matches
    std::cout << "> Matching OpenCV FPs and saving resulting image..." << std::endl;
    imgMatchingOutPath = outDir +img1OutPath + "_"+ img2OutPath +"_OpenCV_SIFT.png";
    matchFPs(img1, kp1, img2, kp2, imgMatching);
    cv::imwrite(imgMatchingOutPath, imgMatching);

    ocvDesc1.release();
    ocvDesc2.release();
    ocvKPs1.clear();
    ocvKPs2.clear();
  }

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
  return 0;
}