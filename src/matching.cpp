#include "../include/descriptors/matching.h"

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

void matchDescriptors(std::vector<KeyPoints> kpListImg1,
                      std::vector<KeyPoints> kpListImg2,
                      std::map<char, char> output,
                      float threshold, int calcDistMode )
{
  std::vector<float> distList;

  for( int i = 0; i < kpListImg1.size(); i++ )
  {
    for( int j = 0; j < kpListImg2.size(); j++ )
    {
      distList.push_back( calculateDistance( kpListImg1[i].descriptor,
                                             kpListImg2[i].descriptor,
                                             calcDistMode ) );
    }

    //
    
  }

}
