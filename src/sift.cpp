#include "../include/descriptors/sift.h"

/**
 * Return a flattened array of a cv::Mat object
 * 
 * @param mat cv::Mat matrix to be flatenned
 * @return the flatenned Matrix in a vector object
**/
std::vector<double> returnRavel( cv::Mat& mat, std::vector<double> arr )
{
  for (int i = 0; i < mat.rows; i++)
  {
    for( int j = 0; j < mat.cols; j++)
    {
      arr.push_back( mat.at<double>(i, j) );
    }
  }

  return arr;
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

cv::Mat repeatLastRow( cv::Mat& mat )
{
  cv::Mat ret = cv::Mat( mat.rows, mat.cols, mat.type(), cv::Scalar(0) );
  int rows = mat.rows;
  int cols = mat.cols;

  for( int i=1; i<rows; i++ )
  {
    for( int j=0; j<cols; j++ )
    {
      ret.at<float>(i-1, j) = mat.at<float>(i, j);
    }
  }

  for( int j=0; j<cols; j++ )
  {
    ret.at<float>(rows-1, j) = mat.at<float>(rows-1, j);
  }

  return ret;
}

cv::Mat repeatFirstRow( cv::Mat& mat )
{
  cv::Mat ret = cv::Mat( mat.rows, mat.cols, mat.type(), cv::Scalar(0) );
  int rows = mat.rows;
  int cols = mat.cols;

  for( int j=0; j<cols; j++ )
  {
    ret.at<float>(0, j) = mat.at<float>(0, j);
  }

  for( int i=0; i<rows-1; i++ )
  {
    for( int j=0; j<cols; j++ )
    {
      ret.at<float>(i+1, j) = mat.at<float>(i, j);
    }
  }

  return ret;
}

cv::Mat repeatFirstColumn( cv::Mat& mat )
{
  cv::Mat ret = cv::Mat( mat.rows, mat.cols, mat.type(), cv::Scalar(0) );
  int rows = mat.rows;
  int cols = mat.cols;

  for( int i=0; i<rows; i++ )
  {
    ret.at<float>(i, 0) = mat.at<float>(i, 0);
  }

  for( int i=0; i<rows; i++ )
  {
    for( int j=0; j<cols-1; j++ )
    {
      ret.at<float>(i, j+1) = mat.at<float>(i, j);
    }
  }

  return ret;
}

cv::Mat repeatLastColumn( cv::Mat& mat )
{
  cv::Mat ret = cv::Mat( mat.rows, mat.cols, mat.type(), cv::Scalar(0) );
  int rows = mat.rows;
  int cols = mat.cols;

  for( int i=0; i<rows-1; i++ )
  {
    for( int j=1; j<cols; j++ )
    {
      ret.at<float>(i, j-1) = mat.at<float>(i, j);
    }
  }

  for( int i=0; i<rows; i++ )
  {
    ret.at<float>(i, cols-1) = mat.at<float>(i, cols-1);
  }

  return ret;
}

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
void gaussianKernel( float sigma, cv::Mat& kernel )
{
  std::cout << "gaussianKernel" << std::endl;
  int size = (int) 2 * (std::ceil( 3 * sigma ) + 1);
  int half = size / 2;

  double r, s = 2.0 * sigma * sigma;
  double sum = 0.0; // to normalize

  kernel = cv::Mat( cv::Size(size, size), CV_32F, cv::Scalar(0.0) );

  for( int x = -half; x <= half; x++ )
  {
    for( int y = -half; y <= half; y++ )
    {
        r = std::sqrt(x * x + y * y);
        kernel.at<float>(x + half, y+half) = (std::exp(-(r * r) / s)) / (M_PI * s);
        sum += kernel.at<float>(x + half, y+half);
    }
  }

  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      sum += kernel.at<float>(i, j);
    }
  }

  // performing normalization
  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      kernel.at<float>(i, j) = kernel.at<float>(i, j) / sum;
    }
  }
}

/**
 * Cartesian to polar coordinate
 * 
 * @param dx x coordinate
 * @param dy y coordinate
 * @param dm return array with magnitude and theta
**/
void cartToPolarGradient( int dx, int dy, double mt[2] )
{
  double mag, theta;

  mag = std::sqrt( (dx*dx)+(dy*dy) );
  theta = ( (std::atan2( dy, dx )+M_PI) * 180/M_PI) - 90;

  mt[0] = mag;
  mt[1] = theta;
}

/**
 * Cartesian to polar angle
 * 
 * @param dx x coordinate
 * @param dy y coordinate
 * @param m return array with magnitudes
 * @param theta return array with thetas
**/
void cartToPolarGradientMat( cv::Mat& dx, cv::Mat& dy, cv::Mat& m, cv::Mat& theta )
{
  for( int i=0; i<dx.rows; i++ )
  {
    for( int j=0; j<dx.cols; j++ )
    {
      double ret[2];
      cartToPolarGradient( dx.at<int>(i, j), dy.at<int>(i, j), ret );
      m.at<double>(i, j) = ret[0];
      theta.at<double>(i, j) = ret[1];
    }
  }
}

void getGradient( cv::Mat& img, int x, int y, double mt[2] )
{
  int dy, dx;

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
void calcOrientation( cv::Mat& img, KeyPoints kp,
                      std::vector<KeyPoints> auxList )
{
  cv::Mat kernel;//, hist;
  float sigma, maxIndex, maxValue, hist[DESC_HIST_BINS];
  int radius, sizeKernel;

  //hist = cv::Mat( cv::Size(1, DESC_HIST_BINS), CV_32F, cv::Scalar(0.0) );

  sigma = DESC_GAUSS_SIGMA * kp.scale;
  radius = (int) 2 * (std::ceil( sigma ) + 1);

  std::cout << "calcOrientation" << std::endl;
  
  gaussianKernel( sigma, kernel );

  for( int i = -radius; i < radius+1; i++ )
  {
    int y, binn;
    double mt[2], weight;

    y = kp.y + i;

    if( isOut(img, 1, y) )
      continue;

    for( int j = -radius; j < radius+1; j++ )
    {
      int x = kp.x + j;
      if( isOut(img, x, 1) )
        continue;

      getGradient( img, x, y, mt );
      
      weight = kernel.at<float>(i+radius, j+radius) * mt[0];
      binn = quantizeOrientation(mt[1], DESC_HIST_BINS) - 1;
      //hist.at<float>(0, binn) += weight;
      hist[binn] += weight;
    }
  }
  
  maxIndex = std::numeric_limits<float>::min();
  maxValue = std::numeric_limits<float>::min();
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
      std::cout << "i: " << i << " aux.direction: " << i*10 << std::endl;
      KeyPoints aux;
      aux.x = kp.x;
      aux.y = kp.y;
      aux.resp = kp.resp;
      aux.scale = kp.scale;
      aux.octave = kp.octave;
      aux.direction = i * 10;
      auxList.push_back( aux );
    } 
    else 
    {
      std::cout << "i: " << i << " so kp: " << std::endl;
    }
  }
  //std::cout << "Release hist" << std::endl;
  //hist = cv::Mat( cv::Size(1, 1), CV_32F, cv::Scalar(0.0) );
  //hist.release();
  std::cout << "Release kernel" << std::endl;
  kernel = cv::Mat( cv::Size(1, 1), CV_32F, cv::Scalar(0.0) );
  //kernel.release();
  std::cout << "Acabou o calc" << std::endl;
}

/**
 * Method that add keypoints orientation to kp.
 * 
 * @param kp KeyPoints vector
 * @param img image being descripted
 * @param mGauss Gaussian blur window size
 * @param sigma  Gaussian blur sigma value
**/
void siftKPOrientation( std::vector<KeyPoints> kp, cv::Mat& img, int mGauss, float sigma )
{
  std::vector<KeyPoints> auxList, newKp;
  cv::Mat img2blur, img2gray;

  cv::GaussianBlur( img, img2blur, cv::Size(mGauss, mGauss), sigma, sigma, 
                   cv::BORDER_REPLICATE );
  cv::cvtColor( img, img2gray, CV_BGR2GRAY );

  for( int i = 0; i < kp.size(); i++ ) 
  {
    int px, py;

    KeyPoints key = kp[i];
    calcOrientation(img, key, auxList);
    std::cout << "saiu calcOrientation" << std::endl;

    // ROUNDING 30 * cos( radians( octave ) ) AND sin( radians( octave ) )
    px = (int) 0.5 + ( 30 * ( std::cos( key.direction * ( M_PI / 180) ) ) );
    py = (int) 0.5 + ( 30 * ( std::sin( key.direction * ( M_PI / 180) ) ) );
      std::cout << "px: " << px << std::endl;
      std::cout << "py: " << py << std::endl;

    cv::arrowedLine( img2gray, cv::Point( key.x, key.y ),
                     cv::Point( key.x + px, key.y + py ),
                     cv::Scalar( 0, 0, 255 ), 1 );
    for( int j = 0; j < auxList.size(); j++ )
    {
      int ppx, ppy;
      KeyPoints point = auxList[j];

      ppx = (int) 0.5 + ( 30 * ( std::cos( point.direction * ( M_PI / 180) ) ) );
      ppy = (int) 0.5 + ( 30 * ( std::sin( point.direction * ( M_PI / 180) ) ) );
      cv::arrowedLine( img2gray, cv::Point( point.x, point.y ), 
                       cv::Point( point.x+px, point.y+py ), 
                       cv::Scalar( 0, 0, 255 ), 1 );
    }
    newKp.insert( newKp.end(), auxList.begin(), auxList.end() );
  }
  kp.insert( kp.end(), newKp.begin(), newKp.end() ); 
}

void getPatchGrads( cv::Mat& subImage, cv::Mat& retX, cv::Mat& retY )
{
  cv::Mat r1 = repeatLastRow( subImage );
  cv::Mat r2 = repeatFirstRow( subImage );

  cv::subtract( r1, r2, retX );

  r1 = repeatLastColumn( subImage );
  r2 = repeatFirstColumn( subImage );
  
  cv::subtract( r1, r2, retY );
}

/**
 * convert an 1d array coordinate into a 2d array coordinates.
 * 
 * @param index the index of 1d array coordinated
 * @param rows the amount of rows in the 2d array
 * @param cols the amount of columns in the 2d array
 * @param ret array containing the index coordinate mapped into 2d array coordinates
 *            where ret[0]=rows and ret[1]=cols
 * 
 * Similar to numpy.unravel_index() function
 * https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
 * 
**/
void unravelIndex( int index, int rows, int cols, int ret[2] )
{
  for( int i=0; i<rows; i++ )
    for( int j=0; j<cols; j++ )
      if( i+j == index )
      {
        ret[0] = i;
        ret[1] = j;
        break;
      }
}

/**
 * Return the histogram of a given subregion
 * 
 * @param mag the subregion's magnitudes
 * @param theta subregion's angles
 * @param numBin amount of values to be returned
 * @param refAngle angle of reference to calculate the histogram
 * @param binWidth width of the bin
 * @param subW width of the subregion
 * @param hist Matrix where the histogram is being returned
**/
void getHistogramForSubregion( cv::Mat& mag, cv::Mat& theta, int numBin, int refAngle,
                               int binWidth, int subW, cv::Mat& hist )
{
  double minimum = 0.000001;
  float center = (subW/2) - 0.5;
  std::vector<double> arrMag, arrThe;
  hist = cv::Mat( cv::Size(1, numBin), CV_32F, cv::Scalar(0.0) );

  returnRavel( mag, arrMag );
  returnRavel( theta, arrThe );

  for( int i=0; i<arrMag.size(); i++ )
  {
    double mg = arrMag[i];
    int angle = (int) (arrThe[i]-refAngle) % 360;
    int b = quantizeOrientation(angle, numBin);
    double vote = mg;

    // b*binWidth is the start angle of the histogram bin
    // b*binWidth+binWidth/2 is the center of the histogram bin
    // angle -[...] is the distance from the angle to the center of the bin 
    int histInterpWeight = 1 - std::abs(angle-(b*binWidth+binWidth/2))/(binWidth/2);
    vote *= std::max( (double) histInterpWeight, minimum );

    int idx[2], xInterpWeight, yInterpWeight;
    unravelIndex( i, subW, subW, idx );

    xInterpWeight = std::max( (double) 1 - ( std::abs(idx[0]-center)/center ), minimum );
    yInterpWeight = std::max( (double) 1 - ( std::abs(idx[1]-center)/center ), minimum );

    vote *= xInterpWeight * yInterpWeight;
    hist.at<float>(0, b) += vote;
  }
}

/**
 * Calculates de description of the Keypoints
 * 
 * @param kpList list of Keypoints to calculate the description
 * @param img image from wich the keypoints description are being calculated
 * @param bins amount of bins
 * @param rad radius used to calculate the bubble around the kpList KeyPoints
 * @param ret vector where the calculated descriptor will be returned
**/
void siftExecuteDescription( std::vector<KeyPoints> kpList, cv::Mat& img, int bins,
                             float rad, std::vector<cv::Mat> ret )
{
  int binWidth, windowSize, radius;
  float sigma, radVal;
  cv::Mat img2gray;

  cv::cvtColor( img, img2gray, CV_BGR2GRAY );

  windowSize = 16;
  sigma = windowSize/6;
  radius = std::floor(windowSize/2);
  binWidth = std::floor(360/bins);
  radVal = 180.0 / M_PI;

  for( int index=0; index < kpList.size(); index++ )
  {
    cv::Mat patch, dx, dy, mag, the, kernel, featVec;
    KeyPoints kp = kpList.at(index);
    float d, theta;// *auxKernel;
    int i, x, y, s, t, l, b, r, xIni, xFim, yIni, yFim, subW;

    i = 0;
    x = kp.x;
    y = kp.y;
    s = kp.scale;
    d = kp.direction;
    
    theta = M_PI * 180.0;
    gaussianKernel( sigma, kernel );

    t = std::max( 0, y-( (int) windowSize/2) );
    l = std::max( 0, x-( (int) windowSize/2) );

    b = std::max( img.rows, y-( (int) (windowSize/2)+1) );
    r = std::max( img.cols, x-( (int) (windowSize/2)+1) );

    patch = img(cv::Range(t, b), cv::Range(l, r));

    getPatchGrads( patch, dx, dy );

    if( dx.rows < windowSize+1 )
    {
      if( t == 0 )
      {
        xIni = kernel.rows-dx.rows;
        xFim = kernel.rows;
      } else
      {
        xIni = 0;
        xFim = dx.rows;
      }
    }
    if( dx.cols < windowSize+1 )
    {
      if( l == 0 )
      {
        yIni = kernel.cols-dx.cols;
        yFim = kernel.cols;
      } else
      {
        yIni = 0;
        yFim = dx.cols;
      }
    }

    kernel = kernel( cv::Range(xIni, xFim), cv::Range(yIni, yFim) );
    
    if( dy.rows < windowSize+1 )
    {
      if( t == 0 )
      {
        xIni = kernel.rows-dy.rows;
        xFim = kernel.rows;
      } else
      {
        xIni = 0;
        xFim = dy.rows;
      }
    }
    if( dy.cols < windowSize+1 )
    {
      if( l == 0 )
      {
        yIni = kernel.cols-dy.cols;
        yFim = kernel.cols;
      } else
      {
        yIni = 0;
        yFim = dy.cols;
      }
    }

    kernel = kernel( cv::Range(xIni, xFim), cv::Range(yIni, yFim) );

    mag = cv::Mat( cv::Size(dx.cols, dx.cols), CV_64F, cv::Scalar(0.0) ); // magnitude
    the = cv::Mat( cv::Size(dx.cols, dx.cols), CV_64F, cv::Scalar(0.0) ); // theta
    
    cartToPolarGradientMat( dx, dy, mag, the );

    if( ( sizeof(kernel) / sizeof(float) ) == 17 )
    {
      dx = dx.dot( kernel );
      dy = dy.dot( kernel );
    }

    subW = (int) windowSize / 4;
    featVec = cv::Mat( cv::Size(1, bins*windowSize), CV_32F, cv::Scalar(0.0) );

    for(int i=0; i<subW; i++)
    {
      for( int j=0; j<subW; j++ )
      {
        t = i*subW;
        l = j*subW;
        b = std::min( img.rows, (i+1) * subW );
        r = std::min( img.cols, (j+1) * subW );

        cv::Mat subMag = mag( cv::Range(t, b), cv::Range(l, r) );
        cv::Mat subThe = the( cv::Range(t, b), cv::Range(l, r) );

        cv::Mat hist;
        getHistogramForSubregion( subMag, subThe, bins, s, binWidth, subW, hist );

        int indexIni = (i*subW*bins) + (j*bins);
        int indexEnd = (i*subW*bins) + ((j+1)*bins);
        
        for(int idx = indexIni; idx<indexEnd; idx++)
          featVec.at<float>(0, idx) = hist.at<float>(0, idx-indexIni);

        subMag.release();
        subMag.release();
      }
    }

    for(int idx=0; idx<featVec.cols; idx++)
    {
      float norm = cv::norm( featVec, cv::NORM_L2, cv::noArray() );
      featVec.at<float>(0, idx) /= std::max((float)0.001, norm);
    }
    ret.push_back(featVec);
  }

}

/**
 * SIFT MAIN METHOD
 * 
 * @param kp KeyPoints detected
 * @param name string with image's name
**/
void siftDescriptor( std::vector<KeyPoints> kp, cv::Mat& img, cv::Mat& imgGray,
                     int mGauss, float sigma )
{
  std::cout << "Calculando orientações" << std::endl;
  siftKPOrientation( kp, img, mGauss, sigma );
  
  std::cout << "KeyPoints com calculo de orientação:" << kp.size() << std::endl;

  std::cout << "########## KEYPOINTS INICIO ##########" << std::endl;
  for( int i=0; i<kp.size(); i++ )
    std::cout << kp[i].y << "\t" << kp[i].x << "\t" << kp[i].scale << "\t" 
              << kp[i].octave << "\t" << kp[i].resp << "\t" 
              << kp[i].direction << std::endl;
  std::cout << "########## KEYPOINTS FINAL ##########" << std::endl;

  std::vector<cv::Mat> descriptorList;
  siftExecuteDescription( kp, img, DESC_BINS, DESC_RADIUS, descriptorList );
}