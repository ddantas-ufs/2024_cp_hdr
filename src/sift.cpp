#include "../include/descriptors/sift.h"

/**
 * Create a Gaussian Kernel based on sigma value
 * 
 * @param sigma
 * @return array with Gaussian distribution based on sigma return a 2-dimensional array
 *         with arbitrary sizes is not efficient and can leave us with memory problems. 
 *         Our workaround is return an 1-dimensional array instead.
 *         
 *         In this way. to access the item arr[i][j], we just : arr[i*size+j].
 *         Where size is the 'j' size. 
**/
float* gaussianKernel( float sigma, int size )
{
  int half = size / 2;

  double r, s = 2.0 * sigma * sigma;
  double sum = 0.0; // to normalize

  float *kernel = new float[size*size];

  for( int x = -half; x <= half; x++ )
  {
    for( int y = -half; y <= half; y++ )
    {
        r = std::sqrt(x * x + y * y);
        kernel[(( x + half ) * size)+(y + half)] = (std::exp(-(r * r) / s)) / (M_PI * s);
        sum += kernel[( ( x + half ) * size )+( y + half )];
    }
}

  // performing normalization
  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      kernel[i*size+j] /= sum;
    }
  }
  return kernel;
}

/**
 * Verify if the coorduinates (x, y) outside from img limits
 * 
 * @param img image
 * @param x coordinate x (column, width)
 * @param y coordinate y (row, height)
 * 
 * @return true if is outside from img or false
**/
bool isOut(cv::Mat img, int x, int y)
{
  int rows = img.rows;
  int cols = img.cols;
  return ( x <= 0 or x >= cols-1 or y <= 0 or y >= rows-1 );
}

void cartToPolarGradient( int dx, int dy, double mt[2] )
{
  double mag, theta;

  mag = std::sqrt( (dx+dx)+(dy*dy) );
  theta = (std::atan2( dy, dx ) * 180 / M_PI) - 90;

  mt[0] = mag;
  mt[1] = theta;
}

void getGradient( cv::Mat img, int x, int y, double mt[2] )
{
  int dy, dx;
  double hip;

  dy = ( ( int ) img.at<float>( std::min( img.rows-1, y+1 ), x ) ) 
     - ( ( int ) img.at<float>( std::max( 0, y-1 ) ), x );
  dx = ( ( int ) img.at<float>( y, std::min( img.cols-1, x+1 ) ) ) 
     - ( ( int ) img.at<float>( y, std::max( 0, x-1 ) ) );

  cartToPolarGradient( dx, dy, mt );
}

int quantizeOrientation( double theta, int bins )
{
  float width = std::floor( 360 / bins );
  return ( int ) std::floor( std::floor( theta ) / width );
}

/**
 * Method that calculates the orientation of a given KeyPoint
 * 
 * @param img source image of keypoint being calculated
 * @param kp KeyPoint to be used to calculate orientation
 * 
 * @return the orientation of the keypoint
**/
std::vector<KeyPoints> calcOrientation( cv::Mat img, KeyPoints kp )
{
  std::vector<KeyPoints> auxList;
  float sigma, maxIndex, maxValue, hist[DESC_HIST_BINS], *kernel;
  int radius, sizeKernel;

  sigma = DESC_GAUSS_SIGMA * kp.resp;
  radius = (int) 2 * (std::ceil( sigma ) + 1);
  sizeKernel = (int) 2 * (std::ceil( 3 * sigma ) + 1);
  
  kernel = gaussianKernel( sigma, sizeKernel );

  for( int i = -radius; i < radius; i++ )
  {
    int y, binn;
    double mt[2], weight;

    y = kp.y + i;

    if( isOut(img, 1, y) )
      continue;

    for( int j = -radius; j < radius; j++ )
    {
      int x = kp.x + j;
      if( isOut(img, x, 1) )
        continue;

      // mt[0] = mag and mt[1] = theta
      getGradient( img, x, y, mt );
      
      // array[i][j] -> array[i*size+j]. See gaussianKernel() comment.
      weight = kernel[( (i+radius)*sizeKernel )+(j+radius)] * mt[0];
      binn = quantizeOrientation(mt[1], DESC_HIST_BINS) - 1;
      hist[binn] += weight;
    }
  }

  // Getting max value and respective index 
  maxValue = std::numeric_limits<float>::min();
  maxIndex = std::numeric_limits<float>::min();
  for( int i = 0; i < DESC_HIST_BINS; i++ )
  {
    if( hist[i] > maxValue )
    {
      maxValue = hist[i];
      maxIndex = i;
    }
  }
  
  kp.direction = maxIndex * 10;

  // If there's values above 85% of max value, return them.
  for( int i = 0; i < DESC_HIST_BINS; i++ )
  {
    if( i == maxIndex )
      continue;
    
    if( hist[i] > ( 0.85 * maxValue ) )
    {
      KeyPoints aux;
      aux.x = kp.x;
      aux.y = kp.y;
      aux.resp = kp.resp;
      aux.scale = kp.scale;
      aux.octave = kp.octave;
      aux.direction = i * 10;
      auxList.push_back( aux );
    }
  }

  return auxList;
}

/**
 * @param kp KeyPoints list
**/
void siftKPOrientation( std::vector<KeyPoints> kp, cv::Mat img, int mGauss, float sigma )
{
  std::vector<KeyPoints> auxList, newKp;
  cv::Mat img2blur, img2gray;

  cv::GaussianBlur( img, img2blur, cv::Size(mGauss, mGauss), sigma, sigma, 
                   cv::BORDER_REPLICATE );
  cv::cvtColor( img, img2gray, CV_BGR2GRAY );

  for( int i = 0; i < kp.size(); i++ ) 
  {
    int px, py;

    KeyPoints key = kp.at(i);
    auxList = calcOrientation(img, key);

    // ROUNDING 30 * cos( radians( octave ) ) AND sin( radians( octave ) )
    px = (int) 0.5 + ( 30 * ( std::cos( key.direction * ( M_PI / 180) ) ) );
    py = (int) 0.5 + ( 30 * ( std::sin( key.direction * ( M_PI / 180) ) ) );

    cv::arrowedLine( img2gray, cv::Point( key.x, key.y ),
                     cv::Point( key.x + px, key.y + py ), 
                     cv::Scalar( 0, 0, 255 ), 1 );
    for( int j = 0; j < auxList.size(); j++ )
    {
      int ppx, ppy;
      KeyPoints point = auxList.at(j);

      ppx = (int) 0.5 + ( 30 * ( std::cos( point.direction * ( M_PI / 180) ) ) );
      ppy = (int) 0.5 + ( 30 * ( std::sin( point.direction * ( M_PI / 180) ) ) );
      cv::arrowedLine( img2gray, cv::Point( point.x, point.y ), 
                       cv::Point( point.x+px, point.y+py ), 
                       cv::Scalar( 0, 0, 255 ), 1 );
    }

    newKp.insert( newKp.end(), auxList.begin(), auxList.end() );
  }
  
  kp.insert( kp.end(), newKp.begin(), newKp.end() ); 

  // NO SCRIPT DE WELERSON, AQUI ESCREVERIA UM ARQUIVO COM OS NOVOS KEYPOINTS
}

/**
 * SIFT MAIN METHOD
 * 
 * @param kp KeyPoints detected
 * @param name string with image's name
**/
int siftDescriptor( std::vector<KeyPoints> kp, cv::Mat img, cv::Mat imgGray, int mGauss,
                    float sigma )
{
  siftKPOrientation( kp, img, mGauss, sigma );
//  siftExecuteDescription();
  return 0;
}
