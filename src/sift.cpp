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

void repeatLastRow(cv::Mat mat, cv::Mat &ret)
{
  mat.row(mat.rows - 1).copyTo(ret.row(ret.rows - 1));
  mat.rowRange(1, mat.rows).copyTo(ret.rowRange(0, ret.rows - 1));
}

void repeatFirstRow(cv::Mat mat, cv::Mat &ret)
{
  mat.row(0).copyTo(ret.row(0));
  mat.rowRange(0, mat.rows - 1).copyTo(ret.rowRange(1, ret.rows));
}

void repeatLastColumn(cv::Mat mat, cv::Mat &ret)
{
  mat.col(mat.cols - 1).copyTo(ret.col(ret.cols - 1));
  mat.colRange(1, mat.cols).copyTo(ret.colRange(0, ret.cols - 1));
}

void repeatFirstColumn(cv::Mat mat, cv::Mat &ret)
{
  mat.col(0).copyTo(ret.col(0));
  mat.colRange(0, mat.cols - 1).copyTo(ret.colRange(1, ret.cols));
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
  int size = (int) (2 * std::ceil( 3 * sigma )) + 1;
  int half = (int) size / 2;

  double r, s = 2.0 * sigma * sigma;
  double sum = 0.0; // to normalize

  kernel = cv::Mat( cv::Size(size, size), CV_32F, cv::Scalar(0.0) );

  for( int x = -half; x <= half; x++ )
  {
    for( int y = -half; y <= half; y++ )
    {
        r = std::sqrt( (x * x) + (y * y) );
        kernel.at<float>(x + half, y+half) = (float) (std::exp(-(r * r) / s)) / (M_PI * s);
        sum += (double) kernel.at<float>(x + half, y+half);
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
  cv::Mat kernel;
  float sigma, maxIndex, maxValue, hist[DESC_HIST_BINS];
  int radius, sizeKernel;

  auxList.clear();

  for( int i=0; i<DESC_HIST_BINS; i++ )
    hist[i] = ( float ) 0.0;

  sigma = DESC_GAUSS_SIGMA * kp.scale;
  radius = (int) 2 * (std::ceil( sigma ) + 1);

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
  //std::cout << "direction kp = " << maxIndex << "*" << "10 = " << kp.direction << std::endl;

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
    //else 
    //{
    //  std::cout << "i: " << i << " so kp: " << std::endl;
    //}
  }
  //std::cout << "Releasing kernel" << std::endl;
  //kernel = cv::Mat( cv::Size(1, 1), CV_32F, cv::Scalar(0.0) );
}

/**
 * Method that add keypoints orientation to kp.
 * 
 * @param kp KeyPoints vector
 * @param img image being descripted
 * @param mGauss Gaussian blur window size
 * @param sigma  Gaussian blur sigma value
**/
void siftKPOrientation( std::vector<KeyPoints> kp, cv::Mat& img, int mGauss,
                        float sigma )
{
  std::vector<KeyPoints> auxList, newKp;
  cv::Mat imgCopy;
  img.copyTo( imgCopy );

  cv::GaussianBlur( img, img, cv::Size(mGauss, mGauss), sigma, sigma, 
                   cv::BORDER_REPLICATE );

  for( int i = 0; i < kp.size(); i++ ) 
  {
    int px, py;

    KeyPoints key = kp[i];
    calcOrientation(img, key, auxList);
    
    //  30 * cos( radians( degrees ) ) AND sin( radians( degrees ) )
    px = (int) ( 30 * ( std::cos( key.direction * ( M_PI / 180) ) ) );
    py = (int) ( 30 * ( std::sin( key.direction * ( M_PI / 180) ) ) );

    cv::arrowedLine( imgCopy, cv::Point( key.x, key.y ),
                     cv::Point( key.x + px, key.y + py ),
                     cv::Scalar( 0, 0, 255 ), 1 );

    for( int j = 0; j < auxList.size(); j++ )
    {
      KeyPoints point = auxList[j];

      px = (int) ( 30 * ( std::cos( point.direction * ( M_PI / 180) ) ) );
      py = (int) ( 30 * ( std::sin( point.direction * ( M_PI / 180) ) ) );
      cv::arrowedLine( imgCopy, cv::Point( point.x, point.y ), 
                       cv::Point( point.x+px, point.y+py ), 
                       cv::Scalar( 0, 0, 255 ), 1 );
    }
    // APPEND newKp
    for( int k=0; k<auxList.size(); k++ )
    {
      newKp.push_back( auxList[k] );
    }
  }
  //std::cout << " append kp " << std::endl;
  for( int k=0; k<newKp.size(); k++ )
  {
    kp.push_back( newKp[k] );
  }
  
  std::cout << " fim siftKPOrientation " << std::endl;
}

void getPatchGrads( cv::Mat& subImage, cv::Mat& retX, cv::Mat& retY )
{
  cv::Mat r1, r2;

  std::cout << "---> Allocating r1" << std::endl;
  r1 = cv::Mat::zeros( subImage.size(), CV_32F );
  std::cout << "---> Allocating r2" << std::endl;
  r2 = cv::Mat::zeros( subImage.size(), CV_32F );

  std::cout << "---> repeatLastRow" << std::endl;
  repeatLastRow( subImage, r1 );
  std::cout << "---> repeatFirstRow" << std::endl;
  repeatFirstRow( subImage, r2 );

  std::cout << "---> subtract row" << std::endl;
  cv::subtract( r1, r2, retX );

  std::cout << "---> repeatLastColumn" << std::endl;
  repeatLastColumn( subImage, r1 );
  std::cout << "---> repeatFirstColumn" << std::endl;
  repeatFirstColumn( subImage, r2 );
  
  std::cout << "---> subtract Column" << std::endl;
  cv::subtract( r1, r2, retY );
  
//  std::cout << "---> r1.release" << std::endl;
//  r1.release();
//  std::cout << "---> r2.release" << std::endl;
//  r2.release();
//  std::cout << "---> end getPatchGrads" << std::endl;
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

  windowSize = 16;
  sigma = windowSize/6;
  radius = std::floor(windowSize/2);
  binWidth = std::floor(360/bins);
  radVal = 180.0 / M_PI;

  std::cout << "for kpList.size()" << std::endl;
  for( int index=0; index < kpList.size(); index++ )
  {
    cv::Mat patch, dx, dy, mag, the, /*kernel,*/ featVec;
    KeyPoints kp = kpList.at(index);
    float d, theta;
    int i, x, y, s, t, l, b, r,/* xIni, xFim, yIni, yFim,*/ subW;

    i = 0;
    x = kp.x;
    y = kp.y;
    s = kp.scale;
    d = kp.direction;
    
    std::cout << "gaussianKernel" << std::endl;
    theta = M_PI * 180.0;
//    gaussianKernel( sigma, kernel );

    t = std::max( 0, y-( (int) windowSize/2) );
    l = std::max( 0, x-( (int) windowSize/2) );

    b = std::max( img.rows, y-( (int) (windowSize/2)+1) );
    r = std::max( img.cols, x-( (int) (windowSize/2)+1) );

    patch = img(cv::Range(t, b), cv::Range(l, r));
    patch.convertTo(patch, CV_32F);

    std::cout << "getPatchGrads" << std::endl;
    getPatchGrads( patch, dx, dy );
    std::cout << "getPatchGrads ok" << std::endl;
/*
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

    std::cout << "subspace kernel 1" << std::endl;
    std::cout << "xIni: " << xIni << std::endl;
    std::cout << "xFim: " << xFim << std::endl;
    std::cout << "yIni: " << yIni << std::endl;
    std::cout << "yFim: " << yFim << std::endl;
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
*/
//    std::cout << "subspace kernel 2" << std::endl;
//    std::cout << "xIni: " << xIni << std::endl;
//    std::cout << "xFim: " << xFim << std::endl;
//    std::cout << "yIni: " << yIni << std::endl;
//    std::cout << "yFim: " << yFim << std::endl;
//    kernel = kernel( cv::Range(xIni, xFim), cv::Range(yIni, yFim) );

    mag = cv::Mat( cv::Size(dx.cols, dx.cols), CV_64F, cv::Scalar(0.0) ); // magnitude
    the = cv::Mat( cv::Size(dx.cols, dx.cols), CV_64F, cv::Scalar(0.0) ); // theta
    
    std::cout << "cartToPolarGradientMat" << std::endl;
    cartToPolarGradientMat( dx, dy, mag, the );

//    if( ( sizeof(kernel) / sizeof(float) ) == 17 )
//    {
//      dx = dx.dot( kernel );
//      dy = dy.dot( kernel );
//    }

    std::cout << "create featVec" << std::endl;
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

        std::cout << "getHistogramForSubregion" << std::endl;
        cv::Mat hist;
        getHistogramForSubregion( subMag, subThe, bins, s, binWidth, subW, hist );

        int indexIni = (i*subW*bins) + (j*bins);
        int indexEnd = (i*subW*bins) + ((j+1)*bins);
        
        std::cout << "populate featVec" << std::endl;
        for(int idx = indexIni; idx<indexEnd; idx++)
          featVec.at<float>(0, idx) = hist.at<float>(0, idx-indexIni);

        std::cout << "release subMag e subThe" << std::endl;
        //subMag.release();
        //subThe.release();
      }
    }

    std::cout << "normalizando featVec" << std::endl;
    for(int idx=0; idx<featVec.cols; idx++)
    {
      float norm = cv::norm( featVec, cv::NORM_L2, cv::noArray() );
      featVec.at<float>(0, idx) /= std::max((float)0.001, norm);
    }

    std::cout << "push_back no vector de retorno" << std::endl;
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
  siftKPOrientation( kp, imgGray, mGauss, sigma );
  
  std::cout << "KeyPoints com calculo de orientação:" << kp.size() << std::endl;

  //removing blur applied in siftKPOrientation
  cv::cvtColor( img, imgGray, CV_BGR2GRAY ); 

  std::vector<cv::Mat> descriptorList;
  siftExecuteDescription( kp, imgGray, DESC_BINS, DESC_RADIUS, descriptorList );
  std::cout << "Size da lista de Keypoints  :" << kp.size() << std::endl;
  std::cout << "Size da lista de descritores:" << descriptorList.size() << std::endl;
}