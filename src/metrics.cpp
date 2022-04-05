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
  std::cout << "Calculating Uniformity..." << std::endl;

  float minFP = std::numeric_limits<float>::max();
  float maxFP = std::numeric_limits<float>::min();
  int totalAreas = qtdKps.size();
  int T = 0;

  for( int i = 0; i < totalAreas; i++ )
    T = T + qtdKps[i];

  for( int i = 0; i < totalAreas; i++ )
  {
    float parcial = float(qtdKps[i])/T;

    std::cout << "parcial = " << qtdKps[i] << "/" << T << "=" << parcial << std::endl;

    minFP = std::fmin( parcial, minFP );
    maxFP = std::fmax( parcial, maxFP );

//    std::cout << "Min parcial: " << minFP << std::endl;
//    std::cout << "Max parcial: " << maxFP << std::endl;
  }

 // std::cout << "Total areas: " << totalAreas << std::endl;
 // std::cout << "T: "   << T << std::endl;
 // std::cout << "Min: " << minFP << std::endl;
 // std::cout << "Max: " << maxFP << std::endl;
  
  return float( 1.0f - float(maxFP - minFP) );
}

float calculateUniformity( std::vector< std::vector<KeyPoints> > lKps )
{
  std::vector<int> qtdKps;

  for(int i = 0; i < lKps.size(); i++)
    qtdKps.push_back( lKps[i].size() );
  
  return calculateUniformity( qtdKps );
}

/**
 * @param A: KeyPoint A
 * @param B: KeyPoint B
 * @return float area of intersection between KeyPoint A and B
 * 
 * Calculing intersection between KeyPoint bubles A and B.
 * References: 
 * https://www.xarg.org/2016/07/calculate-the-intersection-area-of-two-circles/
 * https://www.mathopenref.com/segmentarea.html
 *  
 */
float areaOfIntersection( KeyPoints A, KeyPoints B )
{
  float d = distanceBetwenTwoKeyPoints( A, B );
  float ratioSquare = AP_BUBBLE_RATIO * AP_BUBBLE_RATIO;
  float AoI = 0.0f;

  // SE FOR O CASO, VERIFICAR AQUI SE A e B TEM INTERSEÇÃO
  if( true )
  {
    float x = float(d * d) / float(2.0f * d);
    float z = x * x;
    float y = cv::sqrt(ratioSquare-z);

    AoI = float( ratioSquare * std::asin( y / AP_BUBBLE_RATIO )
        + AP_BUBBLE_RATIO * std::asin( y / AP_BUBBLE_RATIO )
        - y * ( x - cv::sqrt(z) ) );

    return AoI;
  }

  return AoI;
}

/**
 * @brief 
 * 
 * AP = IoU = AO/AU
 * AO = Area of Overlap = Area of Intersection of Circles A and B.
 * AU = Area of Union = Area of Circla A + Area of Circle B - Area of Intersection
 */
void calculateAP()
{

}