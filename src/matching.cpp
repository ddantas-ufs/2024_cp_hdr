#include "../include/descriptors/matching.h"

float vectorEuclideadDistance( std::vector<int> vec1, std::vector<int> vec2 )
{
  return cv::norm(vec1, vec2, cv::NORM_L2);
  /*
  float sum = 0;

  // the sum of squares of differences between corresponding vector elements
  for( int i = 0; i < vec1.size(); i++ )
    sum = sum + std::pow( std::abs(vec1[i] - vec2[i]), 2 );

  // Square root or zero
  if( sum == 0 )
    return 0.0f;
  else
    return std::sqrt( sum );*/
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
  if( distanceMethod == MATCHING_EUCLIDEAN_DIST_CALC )
  {
    //std::cout << " Euclidean" << std::endl;
    return vectorEuclideadDistance(vec1, vec2);
  }
  else if( distanceMethod == MATCHING_HAMMING_DIST_CALC )
  {
    //std::cout << " --> Hamming" << std::endl;
    return vectorHammingDistance(vec1, vec2);
  }

  //std::cout << " --> Received distanceMethod: " << distanceMethod << ". Returning 0." << std::endl;
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
           cv::Mat H, float threshold, int calcDistMode )
{
  int countCorrect = 0, countIncorrect = 0;
  for( int i = 0; i < kpListImg1.size(); i++ )
  {
    std::vector<float> distList;
    //std::cout << " ### Calculating distances to description " << i << " ###"  << std::endl;

    // Calculating all distances to descriptor
    for( int j = 0; j < kpListImg2.size(); j++ )
    {
      //std::cout << " --> Calculating between pair kpList1:" << i << " and kpList2: " << j << "." << std::endl;
      distList.push_back( calculateDistance( kpListImg1[i].descriptor,
                                             kpListImg2[j].descriptor,
                                             calcDistMode ) );
    }

    //std::cout << " IMG 2 has " << kpListImg2.size() << " descriptions." << std::endl;

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

        //std::cout << " --> Novo valor menor: " << minVal1 << std::endl;
        //std::cout << " --> Novo indice menor: " << minValIdx1 << std::endl;
      }
    }

    // Granting that the algorithm won't take same distance
    if( minVal1 == minVal2 )
    {
      std::cout << " --> Obtained values are equal." << std::endl;
      std::cout << "     minValIdx1: " << minValIdx1 << ", minVal1: " << minVal1 << std::endl;
      std::cout << "     minValIdx2: " << minValIdx2 << ", minVal2: " << minVal2 << std::endl;

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

      std::cout << " --> Granting that minimal values are different:" << std::endl;
      std::cout << "     minValIdx1: " << minValIdx1 << ", minVal1: " << minVal1 << std::endl;
      std::cout << "     minValIdx2: " << minValIdx2 << ", minVal2: " << minVal2 << std::endl;
    }
    
    // Calculating if ratio respect threshold
    //float ratio = (1.0f * minVal1) / std::max(1e-6f, minVal2);

    // 
    //std::cout << "     minVal1: " << minVal1 << ", minVal2: " << minVal2 << std::endl;
    //std::cout << "        minVal2 threshold: " << MATCHING_NNDR_THRESHOLD * minVal2 << std::endl;
    if( minVal1 < MATCHING_NNDR_THRESHOLD * minVal2 )
    {
      MatchedKeyPoints kps;
      KeyPoints kpAux;
      kps.kp1 = kpListImg1[i];
      kps.kp2 = kpListImg2[minValIdx1];
      float kpsDistance = 0.0f;

      // If there's a homography matrix, mapping is made. 
      // If there's not, calculate distance from kp1 to kp2.
      if( H.empty() ) kpsDistance = distanceBetwenTwoKeyPoints( kps.kp1, kps.kp2 );
      else
      {
        getHomographicCorrespondence( kps.kp1.x, kps.kp1.y, kpAux.x, kpAux.y, H );
        kpsDistance = distanceBetwenTwoKeyPoints( kps.kp2, kpAux );
      }

      //float kpsDistance = distanceBetwenTwoKeyPoints( kps.kp1, kps.kp2 );

      // Verifying if the keypoints are in the defined max range from each other
      if( kpsDistance > MATCHING_RATIO_MATCH )
      {
        kps.isCorrect = false;
        countIncorrect++;
      }
      else
      {
        kps.isCorrect = true;
        countCorrect++;
      }

//      std::cout << " --> Matched? " << kps.isCorrect << " -- Distance: " << kpsDistance << std::endl;
//      std::cout << " --> Tried match: Xa: " << kps.kp1.x << ", Ya:" << kps.kp1.y << std::endl;
//      std::cout << " --> Tried match: Xb: " << kps.kp2.x << ", Yb:" << kps.kp2.y << std::endl;
//      std::cout << " --> Tried match: aux x: " << kpAux.x << ", aux y:" << kpAux.y << std::endl;

      output.push_back( kps );
    }
  }
  std::cout << " --> Keypoints to match: " << kpListImg1.size() << std::endl;
  std::cout << " --> Correct matches:   " << countCorrect << std::endl;
  std::cout << " --> Incorrect matches: " << countIncorrect << std::endl;
}

void printLineOnImages( cv::Mat img1, cv::Mat img2, cv::Mat &out,
                        std::vector<MatchedKeyPoints> matchedDesc )
{
  concatenateImages(img1, img2, out);

  for( int i = 0; i < matchedDesc.size(); i++ )
  {
    MatchedKeyPoints mkps = matchedDesc.at(i);
    KeyPoints kp1 = mkps.kp1;
    KeyPoints kp2 = mkps.kp2;

    cv::Point2f p1 = cv::Point2f(kp1.x, kp1.y);
    cv::Point2f p2 = cv::Point2f(kp2.x+img1.cols, kp2.y);

    // Correct matchings are blue and incorrect matches are red.
    if( mkps.isCorrect )
      cv::line( out, p1, p2, cv::Scalar(255,0,0), 2 );
    else
      cv::line( out, p1, p2, cv::Scalar(0,0,255), 2 );
  }
  std::cout << " --> Finishing drawing lines..." << std::endl;
}

/**
 * Main Method that match FPs.
 * 
 * @param img1: Reference image
 * @param img1KpList: Reference image keypoint vector
 * @param img2: Query image
 * @param img2KpList: Query image keypoint vector
 * @param H: homography matrix to map keypoints from reference to query image
 * @param imgOut: output image
 * @param kpsOut: matched keypoints
**/
void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList,
               cv::Mat H, std::vector<MatchedKeyPoints> &matchings,
               cv::Mat &imgOut )
{  
  // Compute matching using NNDR algorithm 
  nndr(img1KpList, img2KpList, matchings, H, MATCHING_NNDR_THRESHOLD, MATCHING_EUCLIDEAN_DIST_CALC);

  // Creating image with lines indicating the matches founded
  printLineOnImages(img1, img2, imgOut, matchings);
}

void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList,
               cv::Mat H, cv::Mat &imgOut )
{
  std::vector<MatchedKeyPoints> matchings;
  matchFPs(img1, img1KpList, img2, img2KpList, H, matchings, imgOut );
}

void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList,
               cv::Mat H )
{
  cv::Mat aux;
  matchFPs( img1, img1KpList, img2, img2KpList, H, aux );
}

void matchFPs( std::string img1Path, std::vector<KeyPoints> img1KpList,
               std::string img2Path, std::vector<KeyPoints> img2KpList,
               std::string pathH )
{
  cv::Mat img1, img2, aux, H;
  std::string str = "";

  readImg(img1Path, img1, aux, str);
  readImg(img2Path, img2, aux, str);

  readHomographicMatrix( pathH, H );
  
  matchFPs( img1, img1KpList, img2, img2KpList, H );
}

void matchFPs( cv::Mat img1, std::vector<KeyPoints> img1KpList,
               cv::Mat img2, std::vector<KeyPoints> img2KpList )
{
  cv::Mat aux, H;
  matchFPs( img1, img1KpList, img2, img2KpList, H, aux );
}

void matchFPs( cv::Mat img1, cv::Mat img2, 
               std::vector<KeyPoints> img1KpList,
               std::vector<KeyPoints> img2KpList,
               cv::Mat &imgOut )
{
  cv::Mat H;
  matchFPs( img1, img1KpList, img2, img2KpList, H, imgOut );
}