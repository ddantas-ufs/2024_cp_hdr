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
  std::vector<KeyPoints> kp1, kp2; // CP_HDR KeyPoint list
  cv::Mat img1, img2, img1Gray, img2Gray, img1Out, img2Out, imgMatching, H;
  cv::Mat img1ROI, img2ROI;

  std::string img1OutPath, img2OutPath, imgMatchingOutPath;
  std::string img1Path, img2Path, pathH, outDir, img1ROIPath, img2ROIPath, hdrSuf = ".hdr";

  bool isHDR = false;
  bool considerROI = false;

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "> Received " << argv << " arguments:" << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  > args[" << i << "]: " << args[i] << std::endl;

  img1Path = std::string(args[1]);

  // Evaluating if image is LDR or HDR
  if( 0 == img1Path.compare(img1Path.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
    isHDR = true;

  if( argv > 5 )
  {
    considerROI = true;
    img1Path    = std::string(args[1]);
    img1ROIPath = std::string(args[2]);
    img2Path    = std::string(args[3]);
    img2ROIPath = std::string(args[4]);
    pathH       = std::string(args[5]);
    outDir      = std::string(args[6]);
    std::cout << "  > ### Considering ROI" << std::endl;
  }
  else
  {
    img1Path = std::string(args[1]);
    img2Path = std::string(args[2]);
    pathH    = std::string(args[3]);
    outDir   = std::string(args[4]);
  }
  
  // Reading images and setting output image name
  readImg(img1Path, img1, img1Gray, img1OutPath);
  readImg(img2Path, img2, img2Gray, img2OutPath);

  // Reading ROI as image mask
  //readROIAsImage( img1ROIPath, img1Gray, img1ROI );
  //readROIAsImage( img2ROIPath, img2Gray, img2ROI );
  readROIFromImage( img1ROIPath, img1ROI );
  readROIFromImage( img2ROIPath, img2ROI );

  cv::imwrite("out/img1ROI.png", img1ROI);
  cv::imwrite("out/img2ROI.png", img2ROI);

  // Reading Homography Matrix
  readHomographicMatrix( pathH, H );

  // Normalizing images (mandatory to HDR images).
  if( isHDR )
  {
    mapPixelValues( img1, img1 );
    mapPixelValues( img2, img2 );
  }

  // Running CP_HDR
  std::cout << "> Running CP_HDR SIFT..." << std::endl;
  if( considerROI )
  {
    runSift(img1Gray, kp1, MAX_KP, img1ROI);
    runSift(img2Gray, kp2, MAX_KP, img2ROI);
  }
  else
  {
    runSift(img1Gray, kp1, MAX_KP);
    runSift(img2Gray, kp2, MAX_KP);
  }

  // Getting only the MAX_KP strongest keypoints
  sortKeypoints( kp1 );
  sortKeypoints( kp2 );
  kp1 = vectorSlice( kp1, 0, MAX_KP);
  kp2 = vectorSlice( kp2, 0, MAX_KP);

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
    /*
    cv::Mat teste;
    img2.copyTo( teste );
    for (int i = 0; i < kp1.size(); i++)
      cv::circle(teste, cv::Point(kp2[i].x, kp2[i].y), 4, cv::Scalar(0, 255, 0));

    printMat(H, "Readed Homography Matrix");

    for (int i = 0; i < kp2.size(); i++)
    {
      KeyPoints k;
      getHomographicCorrespondence( kp1[i].x, kp1[i].y, k.x, k.y, H );
      cv::circle(teste, cv::Point( k.x, k.y ), 4, cv::Scalar(255, 0, 0));
    }
    
    cv::imwrite( "_teste.png", teste );
    */
    saveKeypoints( kp1, outDir+img1OutPath+"_CPHDR_SIFT_LDR.txt", kp1.size() );
    saveKeypoints( kp2, outDir+img2OutPath+"_CPHDR_SIFT_LDR.txt", kp2.size() );
    plotKeyPoints( img1, kp1, outDir+img1OutPath+"_CPHDR_SIFT_LDR.png", kp1.size() );
    plotKeyPoints( img2, kp2, outDir+img2OutPath+"_CPHDR_SIFT_LDR.png", kp2.size() );
    imgMatchingOutPath = outDir +img1OutPath + "_"+ img2OutPath +"_CPHDR_SIFT.png";
  }

  // Matching and generating output image with matches
  std::cout << "> Matching CP_HDR FPs and saving resulting image" << std::endl;
  matchFPs(img1, kp1, img2, kp2, H, imgMatching);
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
 
    loadOpenCVKeyPoints( ocvKPs1, ocvDesc1, kp1, true);
    loadOpenCVKeyPoints( ocvKPs2, ocvDesc2, kp2, true);

    // Getting only the strongest keypoints
    sortKeypoints( kp1 );
    sortKeypoints( kp2 );
    kp1 = vectorSlice( kp1, 0, MAX_KP);
    kp2 = vectorSlice( kp2, 0, MAX_KP);

    saveKeypoints( kp1, outDir+img1OutPath+"_OpenCV_SIFT_LDR.txt", kp1.size() );
    saveKeypoints( kp2, outDir+img2OutPath+"_OpenCV_SIFT_LDR.txt", kp2.size() );
    plotKeyPoints( img1, kp1, outDir+img1OutPath+"_OpenCV_SIFT_LDR.png", kp1.size() );
    plotKeyPoints( img2, kp2, outDir+img2OutPath+"_OpenCV_SIFT_LDR.png", kp2.size() );

    // Matching and generating output image with matches
    std::cout << "> Matching OpenCV FPs and saving resulting image..." << std::endl;
    imgMatchingOutPath = outDir +img1OutPath + "_"+ img2OutPath +"_OpenCV_SIFT.png";
    matchFPs(img1, kp1, img2, kp2, H, imgMatching);
    cv::imwrite(imgMatchingOutPath, imgMatching);

    ocvDesc1.release();
    ocvDesc2.release();
    ocvKPs1.clear();
    ocvKPs2.clear();

    // Cleaning objects
    cleanKeyPointVector( kp1 );
    cleanKeyPointVector( kp2 );
    H.release();
    img1.release();
    img2.release();
    img1Out.release();
    img2Out.release();
    img1Gray.release();
    img2Gray.release();
    imgMatching.release();
  }

  return 0;
}