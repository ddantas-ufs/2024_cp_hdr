#include "../include/descriptors/matching.hpp"

double distanceCalculate(double x1, double y1, double x2, double y2)
{
  // Calculating number to square in next step
  double x = x1 - x2;
  double y = y1 - y2;
  double dist;

  // Calculating Euclidean distance
  dist = pow(x, 2) + pow(y, 2);
  dist = sqrt(dist);

  return dist;
}

void matchDescriptors(std::vector<KeyPoints> kpListImg1,
                      std::vector<KeyPoints> kpListImg2,
                      std::map<char, char> output,
                      float threshold, int calcDistMode )
{
  int qtdKpsImg1 = kpListImg1.size();
  int qtdKpsImg2 = kpListImg2.size();

  for( int i = 0; i < qtdKpsImg1; i++ )
  {
    for( int j = 0; j < qtdKpsImg2; j++ )
    {

    }
  }

}
