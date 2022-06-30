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

void calculateRR( cv::Mat H, std::vector<KeyPoints> kpa, std::vector<KeyPoints> kpb,
                  int &cc, float &rr )
{
  std::vector<KeyPoints> kp1, kp2;
  if( kpa.size() < kpb.size() )
  {
    kp1 = kpa;
    kp2 = kpb;
  }
  else
  {
    kp1 = kpb;
    kp2 = kpa;
  }

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

  if( kp1_size == 0.0f ) kp1_size = kp2_size;
  if( kp2_size == 0.0f ) kp2_size = kp1_size;

  cc = total;

  if( kp1_size == 0.0f && kp2_size == 0.0f )
    rr = 0.0f;
  else
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
float calculateAreaOfIntersection( KeyPoints A, KeyPoints B )
{
  float d = distanceBetwenTwoKeyPoints( A, B );
  //std::cout << "--------Distance : " << d << std::endl;
  float ratioSquare = AP_BUBBLE_RATIO * AP_BUBBLE_RATIO;
  float AoI = 0.0f;

  // SE FOR O CASO, VERIFICAR AQUI SE A e B TEM INTERSEÇÃO
  if( d < (AP_BUBBLE_RATIO * 2) )
  {
    float x = float(d * d) / float(2.0f * d);
    float z = x * x;
    float y = cv::sqrt(ratioSquare-z);

    if( d <= 0.0f )
      return CV_PI * ratioSquare;
    
    AoI = float( ratioSquare * std::asin( y / AP_BUBBLE_RATIO ) 
        + ratioSquare * std::asin( y / AP_BUBBLE_RATIO ) 
        - y * ( x + cv::sqrt(z) ) );

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
float calculateIoU( KeyPoints kp1, KeyPoints kp2 )
{
  float AoI = calculateAreaOfIntersection( kp1, kp2 );
  float sumOfAreas = 2.0f * float( CV_PI * AP_BUBBLE_RATIO * AP_BUBBLE_RATIO );
  float AoU = float( sumOfAreas - AoI );
  float IoU = float( AoI / AoU );

  /*
  std::cout << "#####################################" << std::endl;
  std::cout << "Area of Intersection ----: " << AoI << std::endl;
  std::cout << "Sum of Areas ------------: " << sumOfAreas << std::endl;
  std::cout << "Area of Union -----------: " << AoU << std::endl;
  std::cout << "Intersection Over Union -: " << IoU << std::endl;
  std::cout << "#####################################" << std::endl;
  */

  return IoU;
}

float calculateIoU( KeyPoints kp1, KeyPoints kp2, cv::Mat H )
{
  if( H.empty() )
  {
    return calculateIoU( kp1, kp2 );
  }
  else
  {
    KeyPoints kpAux;
    getHomographicCorrespondence( kp1.x, kp1.y, kpAux.x, kpAux.y, H );
    return calculateIoU( kpAux, kp2 );
  }
}

float calculateAP( std::vector<MatchedKeyPoints> kpPairs, cv::Mat H )
{
  int totalPairs = kpPairs.size();
  float sumIoU = 0.0f, ap = 0.0f;

  if( totalPairs == 0 ) return 0.0f;

  for(int i = 0; i < totalPairs; i++)
  {
    float ciou = calculateIoU( kpPairs[i].kp1, kpPairs[i].kp2, H );
    //std::cout << "Parcial IoU " << i << ": " << ciou << std::endl;
    sumIoU = sumIoU + ciou;
  }

  ap = float(sumIoU / totalPairs);

  std::cout << "Total sum : " << sumIoU << std::endl;
  std::cout << "Total pair: " << totalPairs << std::endl;

  return ap;   
}