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
      float distance = 0.0f;

      if( H.empty() ) distance = distanceBetwenTwoKeyPoints( kp1[i], kp2[j] );
      else
      {
        KeyPoints kpAux;
        getHomographicCorrespondence( kp1[i].x, kp1[i].y, kpAux.x, kpAux.y, H );
        distance = distanceBetwenTwoKeyPoints( kp2[j], kpAux );
      }

      if( distance < REPEATABILITY_RATIO * REPEATABILITY_MIN_OVERLAP )
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
}

/**
 * @brief
 * 
 * T = s + m + h
 * 
 * minFP = min(s/T; m/T; h/T) 
 * maxFP = max(s/T; m/T; h/T)
 * 
 * U = 1 − (maxFP − minFP)
 * 
 * @param qtdKps 
 */
float calculateUniformity( std::vector<int> qtdKps )
{
  float minFP = std::numeric_limits<float>::max();
  float maxFP = std::numeric_limits<float>::min();
  int totalAreas = qtdKps.size();
  int T = 0;

  for( int i = 0; i < totalAreas; i++ )
    T = T + qtdKps[i];

  for( int i = 0; i < totalAreas; i++ )
  {
    minFP = std::fmin( float(qtdKps[i]/T), minFP );
    maxFP = std::fmax( float(qtdKps[i]/T), maxFP );
  }
  
  return float( 1.0f - (maxFP - minFP) );
}
