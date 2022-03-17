#include "../include/evaluation/metrics.h"

/**
 * @brief return repeatability rate using OpenCV.
 * 
 * @param img1 First image to be compared
 * @param img2 Second image to be compared
 * @param H Homographic matrix from image 1 to image 2
 * @param rr variable where to return repeatability rate
 * @param cc variable where to return correspondence count
 * @param kp1 list of OpenCV detected keypoints of img1
 * @param kp2 list of OpenCV detected keypoints of img2
 */
void calculateRRUsingOpenCV( cv::Mat img1, cv::Mat img2, cv::Mat H, float &rr, int &cc,
                             std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2 )
{
  cv::evaluateFeatureDetector( img1, img2, H, &kp1, &kp2, rr, cc );
}

//void calculateRR( cv::Mat H, std::vector<MatchedKeyPoints> kpList, float &rr )
void calculateRR( cv::Mat H, std::vector<KeyPoints> kp1, std::vector<KeyPoints> kp2,
                  int &cc, float &rr )
{
  int kp1_size = kp1.size();
  int kp2_size = kp2.size();
  int total = 0;

  cc = 0, rr = 0.0f;

  std::cout << "## METRIC (RR) > KP1 size: " << kp1_size << std::endl;
  std::cout << "## METRIC (RR) > KP2 size: " << kp2_size << std::endl;

  for( int i = 0; i < kp1_size; i++ )
  {
    for( int j = 0; j < kp2_size; j++ )
    {
      int d = 0;
      float overlap = 0.0f;

      if( H.empty() ) d = distanceBetwenTwoKeyPoints( kp1[i], kp2[j] );
      else
      {
        KeyPoints kpAux;
        getHomographicCorrespondence( kp1[i].x, kp1[i].y, kpAux.x, kpAux.y, H );
        d = distanceBetwenTwoKeyPoints( kp2[j], kpAux );
      }

      if( d < REPEATABILITY_RATIO )
      {
        total += 1;
        j = kp2_size+1; // to end 'j for loop'
      }

    }
  }

  cc = total;
  rr = float( total / std::fmin( kp1_size, kp2_size ) );

  std::cout << "## METRIC (RR) > Correspondence: " << cc << std::endl;
  std::cout << "## METRIC (RR) > Repeatability: " << rr << std::endl;
  /*
  int kpListSize = kpList.size();
  int qtdKpsMatched = 0;

  for( int i = 0; i < kpListSize; i++ )
  {
    int kpsDistance = 0;
    float overlap = 0.0f;

    if( H.empty() ) kpsDistance = distanceBetwenTwoKeyPoints( kpList[i].kp1, kpList[i].kp2 );
    else
    {
      KeyPoints kpAux;
      getHomographicCorrespondence( kpList[i].kp1.x, kpList[i].kp1.y, kpAux.x, kpAux.y, H );
      kpsDistance = distanceBetwenTwoKeyPoints( kpList[i].kp2, kpAux );
    }
  
    overlap = kpsDistance / REPEATABILITY_RATIO;
    if( overlap < REPEATABILITY_RATIO ) qtdKpsMatched += 1;
  }

  rr = kpListSize / qtdKpsMatched;
  */
}