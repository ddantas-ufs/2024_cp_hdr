#include "../include/cphdr.h"

/*
** THIS PROGRAM PERFORM THE MATCHING OF TWO IMAGES,
** USING ENTIRELY OPENCV2 TOOLS.
**
** THIS PROGRAM EXPECTS ARGUMENTS
** 
** args[1]: image1
** args[2]: image2
** args[3]: Homographic matrix
** args[4]: matching output
** 
** @author arturxz, 01/10/2021 (created)
*/
int main(int argv, char** args)
{
  // CREATING OBJECTS
  cv::Mat inputImage1, inputImage2, grayInputImage1, grayInputImage2,
          descriptor1, descriptor2, outputImage, H, Hg;
  std::string outImageName1, outImageName2, outputPath;
  std::vector<cv::KeyPoint> kpsImage1, kpsImage2;

  // SHOWING INPUTS
  std::cout << "received " << argv << " arguments." << std::endl;
  
  for( int i = 0; i < argv; i++ )
    std::cout << "  args[" << i << "]: " << args[i] << std::endl;

  // READING IMAGES AND SETTING OUTPUT IMAGE NAME
  std::cout << "Reading images..." << std::endl;
  readImg(args[1], inputImage1, grayInputImage1, outImageName1);
  outputPath = std::string(args[4]) + "match_" +outImageName1;

  readImg(args[2], inputImage2, grayInputImage2, outImageName2);
  outputPath = outputPath + "_"+ outImageName2 +".png";

  // READING HOMOGRAPHIC MATRIX
  readHomographicMatrix( args[3], H );

  if ( inputImage1.empty() || inputImage2.empty() || grayInputImage1.empty() || grayInputImage2.empty() )
  {
    std::cout << "Could not open or find the image!" << std::endl;

    std::cout << "inputImage1 is empty? " << inputImage1.empty() << std::endl;
    std::cout << "inputImage2 is empty? " << inputImage2.empty() << std::endl;
    std::cout << "grayInputImage1 is empty? " << grayInputImage1.empty() << std::endl;
    std::cout << "grayInputImage2 is empty? " << grayInputImage2.empty() << std::endl;
    return -1;
  }

  // COMPUTING KEYPOINTS USING SIFT
  std::cout << "Computing Keypoints..." << std::endl;
  cv::Ptr<cv::SIFT> siftImage1 = cv::SIFT::create();
  siftImage1->detect( grayInputImage1, kpsImage1 );
  siftImage1->compute( grayInputImage1, kpsImage1, descriptor1);

  cv::Ptr<cv::SIFT> siftImage2 = cv::SIFT::create();
  siftImage2->detect( grayInputImage2, kpsImage2 );
  siftImage2->compute( grayInputImage2, kpsImage2, descriptor2);

  // PERFORMING MATCHING
  std::cout << "Matching..." << std::endl;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<cv::DMatch> > knnMatches;
  matcher->knnMatch( descriptor1, descriptor2, knnMatches, 2 );

  // TESTING MATCH WITH USING RATIO TEST
  std::cout << "Ratio Test..." << std::endl;
  const float thresh = 0.7f;
  std::vector<cv::DMatch> goodMatches;
  for (size_t i = 0; i < knnMatches.size(); i++)
  {
    if (knnMatches[i][0].distance < thresh * knnMatches[i][1].distance)
    {
      goodMatches.push_back(knnMatches[i][0]);
    }
  }

  // DRAWING MATCHES
  std::cout << "Drawing matches..." << std::endl;
  cv::drawMatches(inputImage1, kpsImage1, inputImage2, kpsImage2,
                  goodMatches, outputImage, cv::Scalar::all(-1), cv::Scalar::all(-1), 
                  std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  float rr = 0.0f;
  int cc = 0;
  calculateRRUsingOpenCV( inputImage1, inputImage2, H, rr, cc, kpsImage1, kpsImage2 );

  std::cout << "##########################" << std::endl;
  std::cout << "rr1 = " << rr << std::endl;
  std::cout << "cc1 = " << cc << std::endl;
  std::cout << "##########################" << std::endl;

  rr = 0.0f, cc = 0;
  cv::evaluateFeatureDetector( inputImage1, inputImage2, H, &kpsImage1, &kpsImage2, rr, cc );

  std::cout << "##########################" << std::endl;
  std::cout << "rr2 = " << rr << std::endl;
  std::cout << "cc2 = " << cc << std::endl;
  std::cout << "##########################" << std::endl;

  rr = 0.0f, cc = 0;
  std::vector<KeyPoints> kp1, kp2;
  loadOpenCVKeyPoints(kpsImage1, descriptor1, kp1);
  loadOpenCVKeyPoints(kpsImage2, descriptor2, kp2);
//  loadOpenCVKeyPoints(kpsImage1, kp1);
//  loadOpenCVKeyPoints(kpsImage2, kp2);

  calculateRR(H, kp1, kp2, cc, rr);
  std::cout << "> ##############################" << std::endl;
  std::cout << "> Repeatability Rate: " << rr << std::endl;
  std::cout << "> Correspondence total: " << cc << std::endl;
  std::cout << "> ##############################" << std::endl;

  // SAVING OUTPUT IMAGE
  std::cout << "Saving output image..." << std::endl;
  cv::imwrite(outputPath, outputImage);

  return 0;
}