#include "../include/cphdr.h"

float vectorEuclideadDistance( std::vector<int> vec1, std::vector<int> vec2 )
{
  long sum = 0;

  // the sum of squares of differences between corresponding vector elements
  for( int i = 0; i < vec1.size(); i++ )
    sum  = sum + std::pow( std::abs(vec1[i] - vec2[i]), 2 );

  // Square root or zero
  if( sum == 0 )
    return 0.0f;
  else
    return std::sqrt( sum );
}

float vectorHammingDistance( std::vector<int> vec1, std::vector<int> vec2 )
{
  float ret = 0.0f;
  for( int i = 0; i < vec1.size(); i++ )
    if( vec1[i] == vec2[i] )
      ret = ret + 1.0f;

  return ret;
}

float calculateDistance( std::vector<int> vec1, std::vector<int> vec2, 
                         int distanceMethod )
{
  // Choosing distance calculation method
  if( distanceMethod == MATCHING_EUCLIDIAN_DIST_CALC )
  {
    return vectorEuclideadDistance(vec1, vec2);
  }
  else if( distanceMethod == MATCHING_HAMMING_DIST_CALC )
  {
    return vectorHammingDistance(vec1, vec2);
  }

  return 0.0f;
}

void concatenateImages( cv::Mat img1, cv::Mat img2, cv::Mat &out )
{
  // Concatenating img1 and img2 into out
  cv::hconcat(img1, img2, out);

  // drawing a grayscale line in the middle of the image.
  // just for debugging purposes
  std::cout << " --> writing line" << std::endl;
  if( img1.channels() == 1 )
    cv::line( out, cv::Point(img1.cols, 0), cv::Point(img1.cols, img1.rows), cv::Scalar(125), 1);
  else 
    cv::line( out, cv::Point(img1.cols, 0), cv::Point(img1.cols, img1.rows), cv::Scalar(125,125,125), 1);
  
}

void nndr( std::vector<KeyPoints> kpListImg1,
           std::vector<KeyPoints> kpListImg2,
           std::vector<MatchedKeyPoints> &output,
           float threshold, int calcDistMode )
{

  for( int i = 0; i < kpListImg1.size(); i++ )
  {
    std::vector<float> distList;

    // Calculating all distances to descriptor
    for( int j = 0; j < kpListImg2.size(); j++ )
    {
      distList.push_back( calculateDistance( kpListImg1[i].descriptor,
                                             kpListImg2[j].descriptor,
                                             calcDistMode ) );
    }

    std::cout << " ### Tamanho distList ###" << std::endl;
    std::cout << " distList: " << distList.size() << std::endl;

    // Getting 1st and 2nd smallest distance from kpListImg1 description
    float minVal1 = std::numeric_limits<float>::max();
    float minVal2 = std::numeric_limits<float>::max();
    int minValIdx1 = -1, minValIdx2 = -1;
    for( int j = 0; j < kpListImg2.size(); j++ )
    {
      if( distList[j] < minVal1 )
      {
        minVal2 = minVal1;
        minValIdx2 = minValIdx1;

        minVal1 = distList[j];
        minValIdx1 = j;
      }
    }

    // Granting that the algorithm won't take same distance
    if( minVal1 == minVal2 )
    {
      std::cout << " ## Valores obtidos são iguais" << std::endl;
      std::cout << "minValIdx1: " << minValIdx1 << ", minVal1: " << minVal1 << std::endl;
      std::cout << "minValIdx2: " << minValIdx2 << ", minVal2: " << minVal2 << std::endl;

      minVal2 = std::numeric_limits<float>::max();
      minValIdx2 = -1;

      for( int j = 0; j < kpListImg2.size(); j++ )
      {
        // Minor value different from minVal1
        if( (distList[j] < minVal2) && (distList[j] > minVal1) )
        {
          minVal2 = distList[j];
          minValIdx2 = j;
        }
      }

      std::cout << " ## Depois de garantir diferença..." << std::endl;
      std::cout << "minValIdx1: " << minValIdx1 << ", minVal1: " << minVal1 << std::endl;
      std::cout << "minValIdx2: " << minValIdx2 << ", minVal2: " << minVal2 << std::endl;
    }
    
    // Calculating if ratio respect threshold
    float ratio = (1.0f * minVal1) / std::max(1e-6f, minVal2);

    if( ratio < threshold )
    {
      MatchedKeyPoints kps;
      kps.kp1 = kpListImg1[i];
      kps.kp2 = kpListImg2[minValIdx1];
      output.push_back( kps );
    }
  }
}

void printLineOnImages( cv::Mat img1, cv::Mat img2, cv::Mat &out,
                        std::vector<MatchedKeyPoints> matchedDesc )
{
  concatenateImages(img1, img2, out);

  for( int i = 0; i < matchedDesc.size(); i++ )
  {
    KeyPoints kp1 = matchedDesc.at(i).kp1;
    KeyPoints kp2 = matchedDesc.at(i).kp2;

    cv::Point p1 = cv::Point(kp1.x, kp1.y);
    cv::Point p2 = cv::Point(kp2.x+img1.cols, kp2.y);

    // A RATIO AROUND BOTH KEYPOINTS IS CALCULATE TO CONSIDER
    // IF IS A MATCH OR NOT. MATCHING_RATIO_MATCH DEFAULTS
    if( ( std::abs(p1.x-p2.x) < MATCHING_RATIO_MATCH ) && 
        ( std::abs(p1.y-p2.y) < MATCHING_RATIO_MATCH ) )
      cv::line( out, p1, p2, (0,255,0), 2 );
    else
      cv::line( out, p1, p2, (0,0,255), 2 );
  }
}

void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList )
{
  cv::Mat imgOut;
  std::vector<MatchedKeyPoints> matchings;
  
  nndr(img1KpList, img2KpList, matchings, MATCHING_NNDR_THRESHOLD, MATCHING_RATIO_MATCH);
  printLineOnImages(img1, img2, imgOut, matchings);
}

void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList,
               cv::Mat &imgOut )
{
  std::vector<MatchedKeyPoints> matchings;
  
  nndr(img1KpList, img2KpList, matchings, MATCHING_NNDR_THRESHOLD, MATCHING_RATIO_MATCH);
  printLineOnImages(img1, img2, imgOut, matchings);
}

void matchFPs( std::string img1Path, std::vector<KeyPoints> img1KpList,
               std::string img2Path, std::vector<KeyPoints> img2KpList )
{
  cv::Mat img1, img2, img;
  std::vector<MatchedKeyPoints> matchings;
  std::string str = "";

  readImg(img1Path, img1, img, str);
  readImg(img2Path, img2, img, str);
  
  matchFPs( img1, img1KpList, img2, img2KpList );
}