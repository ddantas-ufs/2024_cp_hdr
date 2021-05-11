#include "../include/descriptors/sift.h"


void printMat( cv::Mat &m, std::string nm )
{
    std::cout << nm << std::endl;
    for(int i=0; i<m.rows; i++)
    {
      for(int j=0; j<m.cols; j++)
        std::cout << m.at<float>(i,j) << " ";
        
      std::cout << std::endl;
    }
}

/**
 * Return a flattened array of a cv::Mat object
 * 
 * @param mat cv::Mat matrix to be flatenned
 * @return the flatenned Matrix in a vector object
 * 
**/
void returnRavel( cv::Mat &mat, cv::Mat &flat )
{
  for (int i = 0; i < mat.rows; i++)
  {
    for( int j = 0; j < mat.cols; j++)
    {
      float item = mat.at<float>(i, j);
      flat.at<float>(i*mat.cols+j) = item;
    }
  }
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
bool isOut(cv::Mat &img, float x, float y)
{
  int rows = img.rows;
  int cols = img.cols;
  return ( x < 0 || x > cols-1 || y < 0 || y > rows-1 );
}

void repeatLastRow( cv::Mat &mat, cv::Mat &ret )
{
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
    ret.at<float>(rows-1, j) = mat.at<float>(rows-1, j);
}

void repeatFirstRow( cv::Mat &mat, cv::Mat &ret )
{
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
}

void repeatLastColumn( cv::Mat &mat, cv::Mat &ret )
{
  int rows = mat.rows;
  int cols = mat.cols;

  for( int i=0; i<rows; i++ )
  {
    for( int j=1; j<cols; j++ )
    {
      ret.at<float>(i, j-1) = mat.at<float>(i, j);
    }
  }

  for( int i=0; i<rows; i++ )
    ret.at<float>(i, cols-1) = mat.at<float>(i, cols-1);
}

void repeatFirstColumn( cv::Mat &mat, cv::Mat &ret )
{
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
}

void gaussianKernel( int size, float sigma, cv::Mat &kernel )
{
  int half = (int) size / 2;

  float r, s = 2.0 * sigma * sigma;
  float sum = 0.0; // to normalize

  kernel = cv::Mat( cv::Size(size, size), CV_32FC1, cv::Scalar(0.0) );

  for( int x = -half; x <= half; x++ )
  {
    for( int y = -half; y <= half; y++ )
    {
        r = std::sqrt( (x * x) + (y * y) );
        kernel.at<float>(x + half, y+half) = (float) (std::exp(-(r * r) / s)) / (M_PI * s);
        sum += (float) kernel.at<float>(x + half, y+half);
    }
  }

  // performing normalization
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j)
      kernel.at<float>(i, j) = kernel.at<float>(i, j) / sum;
}

/**
 * Cartesian to polar coordinate
 * 
 * @param dx x coordinate
 * @param dy y coordinate
 * @param dm return array with magnitude and theta
**/
void cartToPolarGradient( float dx, float dy, float mt[2] )
{
  mt[0] = std::sqrt( (dx*dx)+(dy*dy) );
  mt[1] = ( (std::atan2(dy, dx) ) * (180.0/M_PI) );

  if( mt[1] < 0 ) mt[1] += 360.0f;
  if( mt[1] >= 360.0f ) mt[1] = 0.0f; 
  //printf( "mt[1] %f ", mt[1] );

  if( std::isnan(mt[0]) || std::isnan(mt[1]) )
    std::cout << "###################################---> NAN!! - dx: " << dx << ", dy: " << dy << std::endl;
}

/**
 * Cartesian to polar angle
 * 
 * @param dx x coordinate
 * @param dy y coordinate
 * @param m return array with magnitudes
 * @param theta return array with thetas
**/
void cartToPolarGradientMat( cv::Mat &dx, cv::Mat &dy, cv::Mat &m, cv::Mat &t )
{
  std::cout << "---> cartToPolarGradientMat" << std::endl;
  for( int i=0; i<dx.rows; i++ )
  {
    for( int j=0; j<dx.cols; j++ )
    {
      float ret[2] = {0,0};
      float x = (float) dx.at<float>(i, j);
      float y = (float) dy.at<float>(i, j);

      cartToPolarGradient( x, y, ret );
      
      m.at<float>(i, j) = ret[0];
      t.at<float>(i, j) = ret[1];
    }
  }
}

void getGradient( cv::Mat& img, int x, int y, float mt[2] )
{
  int xm = x-1, xp = x+1;
  int ym = y-1, yp = y+1;
  float dy, dx;

  // Extrapolating image borders        
  if( xp > img.cols ) xp = img.cols-(xp - img.cols)-1;
  if( yp > img.rows ) yp = img.rows-(yp - img.rows)-1;
  if( xm < 0 ) xm = std::abs( xm );
  if( ym < 0 ) ym = std::abs( ym );

  dx = (img.at<float>( y, xp )) - (img.at<float>( y, xm ));
  dy = (img.at<float>( yp, x )) - (img.at<float>( ym, x ));

  cartToPolarGradient( dx, dy, mt );
}

int quantizeOrientation( float theta, int bins )
{
  if( theta < 0 )
    return quantizeOrientation(theta+360, bins);

  int binSize = std::floor( 360/bins );
  binSize = (int) (std::floor(theta)/binSize);
  return binSize;
}

/**
 * Method that calculates the orientation of a given KeyPoint
 * 
 * @param img source image of keypoint being calculated
 * @param kp KeyPoint to be used to calculate orientation
 * 
 * @return the orientation of the keypoint
**/
void calcOrientation( cv::Mat &img, KeyPoints &kp )
{
  cv::Mat kernel;
  float hist[SIFT_DESC_ORIENT_HIST_BINS], maxIndex, maxValue;
  int radius, size;

  // Initializing hist array with 0.0f
  for(int z = 0; z<SIFT_DESC_ORIENT_HIST_BINS; z++ ) hist[z] = 0.0f;

  size = (int) ( (2 * std::ceil( 3 * SIFT_DESC_ORIENT_SIGMA )) + 1 );
  radius = (int) size / 2;

  gaussianKernel( size, SIFT_DESC_ORIENT_SIGMA*kp.scale, kernel );

  for( int i = -radius; i < radius+1; i++ )
  {
    // stores Magnitude in [0] and Theta in [1] (M and T)
    float mt[2];
    int y = kp.y + i;
    if( !isOut(img, 1, y) )
    {
      for( int j = -radius; j < radius+1; j++ )
      {
        int x = kp.x + j;
        if( !isOut(img, x, 1) )
        {
          getGradient( img, x, y, mt );
          
          float weight = (float) kernel.at<float>(i+radius, j+radius) * mt[0];
          int binn = quantizeOrientation(mt[1], SIFT_DESC_ORIENT_HIST_BINS);

          //std::cout << "---> mag=" << mt[0] << ", theta=" << mt[1] << ", binn=" << binn << ", weight=" << weight << std::endl;
          //std::cout << "---> coords y=" << y << ", x=" << x << ", i+r=" << i+radius << ", j+r=" << j+radius << std::endl;
          
          hist[binn] += weight;
          //std::cout << "---> hist[binn]=" << hist[binn] << std::endl << std::endl;
          //hist.at<float>(binn, 0) = hist.at<float>(binn, 0) + weight;
          //std::cout << "---> hist[binn]=" << hist.at<float>(binn, 0) << std::endl << std::endl;
        }
      }
    }
  }

  //for(int z = 0; z<SIFT_DESC_ORIENT_HIST_BINS; z++ ) 
  //  std::cout << hist[z] << ", ";
  //std::cout << std::endl;
  
  maxIndex = std::numeric_limits<float>::min();
  maxValue = std::numeric_limits<float>::min();
  for( int z = 0; z < SIFT_DESC_ORIENT_HIST_BINS; z++ )
  {
    float val = hist[z];
    if( val > maxValue )
    {
      maxValue = val;
      maxIndex = z;
    }
  }
  
  kp.direction = maxIndex * 10;
  //std::cout << "kpIndex = " << maxIndex << std::endl;
  //std::cout << "kpValue = " << maxValue << std::endl;
  //std::cout << "kp.direction = " << kp.direction << std::endl;
  //printKeypoint( kp );
  //std::cout << "--------------------------------------------------" << std::endl;
}

/**
 * Method that add keypoints orientation to kp.
 * 
 * @param kp KeyPoints vector
 * @param img image being descripted
 * @param mGauss Gaussian blur window size
 * @param sigma  Gaussian blur sigma value
**/
void siftKPOrientation( std::vector<KeyPoints> &kp, cv::Mat &img, int mGauss,
                        float sigma )
{
  cv::GaussianBlur( img, img, cv::Size(mGauss, mGauss), sigma, sigma, 
                    cv::BORDER_REPLICATE );

  for( int i = 0; i < kp.size(); i++ )
    calcOrientation( img, kp[i] );
  
  std::cout << " fim siftKPOrientation " << std::endl;
}

void getPatchGrads( cv::Mat &subImage, cv::Mat &retX, cv::Mat &retY )
{
  std::cout << "getPatches subImage size: " << subImage.size() << std::endl;
  cv::Mat r1 = cv::Mat( subImage.size(), subImage.type() );
  cv::Mat r2 = cv::Mat( subImage.size(), subImage.type() );

  repeatLastRow( subImage, r1 );
  repeatFirstRow( subImage, r2 );

  cv::subtract( r1, r2, retX );

  repeatLastColumn( subImage, r1 );
  repeatFirstColumn( subImage, r2 );
  
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
 * | 01 02 03 04 |
 * | 05 06 07 08 |
 * | 09 10 11 12 |
 * | 13 14 15 16 |
 * 
**/
void unravelIndex( int index, int rows, int cols, int ret[2] )
{
  int cont = 0;
  for( int i=0; i<rows; i++ )
    for( int j=0; j<cols; j++ )
    {
      if( cont == index )
      {
        ret[0] = i;
        ret[1] = j;
        break;
      }
      cont++;
    }
}

/**
 * Return the histogram of a given subregion
 * 
 * @param mag the subregion's magnitudes
 * @param theta subregion's angles
 * @param numBin amount of values to be returned
 * @param refAngle Keypoint reference angle to calculate histogram
 * @param binWidth width of the bin (360/8=45)
 * @param subW width of the subregion (=4)
 * @param hist Matrix where the histogram is being returned
**/
void getHistogramForSubregion_old( cv::Mat &mag, cv::Mat &theta, int numBin, float refAngle,
                               int binWidth, int subW, cv::Mat& hist )
{
  std::cout << "---> getHistogramForSubregion_old" << std::endl;
  float minimum = 0.000001f;
  float center = (subW/2.0f) - 0.5f;

  hist = cv::Mat::zeros( cv::Size(numBin, 1), CV_32FC1 );
  
  cv::Mat arrMag = cv::Mat::zeros( cv::Size(mag.rows*mag.cols, 1), CV_32FC1 );
  cv::Mat arrThe = cv::Mat::zeros( cv::Size(theta.rows*theta.cols, 1), CV_32FC1 );

  returnRavel( mag, arrMag );  
  returnRavel( theta, arrThe );

  for( int i=0; i<arrMag.cols; i++ )
  {
    float mg = arrMag.at<float>(i);
    int angle = (int) (arrThe.at<float>(i)-refAngle) % 360;
    int b = quantizeOrientation(angle, numBin);
    float vote = mg;
    
    // b*binWidth is the start angle of the histogram bin
    // b*binWidth+binWidth/2 is the center of the histogram bin
    // angle -[...] is the distance from the angle to the center of the bin 
    float histInterpWeight = 1-std::abs(angle-( (b*binWidth)+(binWidth/2)) ) / (binWidth/2);
    vote *= std::max( histInterpWeight, minimum );
    std::cout << "mg: " << mg << ", angle: " << angle << ", b: " << b << std::endl;
    std::cout << "histInterpWeight: " << histInterpWeight << ", vote1: " << vote << std::endl;

    // interpolating
    int idx[2];
    float xInterpWeight, yInterpWeight;
    unravelIndex( i, subW, subW, idx );

    xInterpWeight = std::max( (float) 1-(std::abs(idx[0]-center)/center), minimum );
    yInterpWeight = std::max( (float) 1-(std::abs(idx[1]-center)/center), minimum );
    std::cout << "xInterpWeight: " << xInterpWeight << std::endl;
    std::cout << "yInterpWeight: " << yInterpWeight << std::endl;

    vote *= (xInterpWeight * yInterpWeight);
    std::cout << "vote3: " << vote << std::endl;
    hist.at<float>(b) += vote;// * angle;
  }
  std::cout << "--> histSubregion: ";
  //for(int k = 0; k<numBin; k++) std::cout << hist.at<int>(k) << " ";
  for(int k = 0; k<numBin; k++) std::cout << hist.at<float>(k) << " ";
  std::cout << std::endl;
}
/**
 * Return the histogram of a given subregion
 * 
 * @param mag the subregion's magnitudes
 * @param theta subregion's angles
 * @param numBin amount of values to be returned
 * @param refAngle Keypoint reference angle to calculate histogram
 * @param refScale Keypoint reference scale to calculate histogram
 * @param binWidth width of the bin (360/8=45)
 * @param subW width of the subregion (=4)
 * @param hist Matrix where the histogram is being returned
**/
//*
void getHistogramForSubregion( cv::Mat &mag, cv::Mat &theta, int numBin, KeyPoints &kp,
                               int cIni, int rIni, int cFim, int rFim, int binWidth, 
                               int subW, cv::Mat& hist )
{
  //float minimum = 0.000001f;
  float minimum = 0.1f;
  //float center = (subW/2.0f) - 0.5f;

  hist = cv::Mat::zeros( cv::Size(numBin, 1), CV_32FC1 );

  cv::Mat arrMag = cv::Mat::zeros( cv::Size(mag.rows*mag.cols, 1), CV_32FC1 );
  cv::Mat arrThe = cv::Mat::zeros( cv::Size(theta.rows*theta.cols, 1), CV_32FC1 );

  returnRavel( mag, arrMag );  
  returnRavel( theta, arrThe );

  for( int i=0; i<arrMag.cols; i++ )
  {
    float mg = arrMag.at<float>(i);
    int angle = (int) (arrThe.at<float>(i)-kp.direction) % 360;
    int b = quantizeOrientation(angle, numBin);
    float vote = mg;
    
    // b*binWidth is the start angle of the histogram bin
    // b*binWidth+binWidth/2 is the center of the histogram bin
    // angle -[...] is the distance from the angle to the center of the bin 
    float histInterpWeight = 1-std::abs(angle-((b*binWidth)+(binWidth/2)))/(binWidth/2);
    vote *= std::max( histInterpWeight, minimum );
    std::cout << "mg: " << mg << ", angle: " << angle << ", b: " << b << std::endl;
    std::cout << "histInterpWeight: " << histInterpWeight << ", vote1: " << vote << std::endl;

    // interpolating
    int idx[2];
    float xInterpWeight, yInterpWeight;
    unravelIndex( i, subW, subW, idx );

    //xInterpWeight = std::max( (float) 1-(std::abs(idx[0]-center)/center), minimum );
    //yInterpWeight = std::max( (float) 1-(std::abs(idx[1]-center)/center), minimum );
    int xCoord = cIni + idx[0];
    int yCoord = rIni + idx[1];
    if(xCoord > cFim) xCoord = cFim;
    if(yCoord > rFim) yCoord = rFim;
    xInterpWeight = std::max( (float) std::abs((float)xCoord-kp.x), minimum );
    yInterpWeight = std::max( (float) std::abs((float)yCoord-kp.y), minimum );
    
    std::cout << "xInterpWeight: " << xInterpWeight << std::endl;
    std::cout << "yInterpWeight: " << yInterpWeight << std::endl;

    vote *= (xInterpWeight * yInterpWeight);
    std::cout << "vote3: " << vote << std::endl;
    hist.at<float>(b) += vote;// * angle;
  }

  std::cout << "--> histSubregion: ";
  //for(int k = 0; k<numBin; k++) std::cout << hist.at<int>(k) << " ";
  for(int k = 0; k<numBin; k++) std::cout << hist.at<float>(k) << " ";
  std::cout << std::endl;
}

void siftExecuteDescription( std::vector<KeyPoints> &kpList, cv::Mat &img )
{
  cv::Mat siftWindow, kernel;

  // Calculates a 17x17 Gaussian kernel
  gaussianKernel( SIFT_DESC_WINDOW+1, SIFT_DESC_WINDOW/2, kernel );

  for( int i = 0; i < kpList.size(); i++ )
  {
    cv::Mat mag, the, dx, dy;
    std::vector<float> tempDescriptor;
    int swHalf = (int) SIFT_DESC_WINDOW / 2;

    // Generating 17x17 window (16x16 window) + keypoint's row and column
    siftWindow = cv::Mat::zeros( SIFT_DESC_WINDOW+1, SIFT_DESC_WINDOW+1, CV_32FC1 );
    int swRows = 0;
    for( int rows = kpList[i].y-swHalf; rows <= kpList[i].y+swHalf; rows++ )
    {
      int swCols = 0;
      for( int cols = kpList[i].x-swHalf; cols <= kpList[i].x+swHalf; cols++ )
      {
        int x = cols, y = rows;

        // Extrapolating image borders        
        if( x >= img.cols ) x = img.cols-(x - img.cols)-1;
        if( y >= img.rows ) y = img.rows-(y - img.rows)-1;
        if( x < 0 ) x = std::abs( x );
        if( y < 0 ) y = std::abs( y );

        //std::cout << "----> X: " << x << ", Y: " << y << std::endl;
        //std::cout << "----> swRows: " << swRows << ", swCols: " << swCols << std::endl;
        float val = img.at<float>(y, x);
        if( std::isnan(val))
          std::cout << "x: " << x << ", cols: " << cols << ", y: " << y << ", rows: " << rows << std::endl;

        siftWindow.at<float>(swRows, swCols) = val;
        swCols = swCols + 1;
      }
      swRows = swRows + 1;
    }

    //printMat( siftWindow, "---------- siftWindow ----------" );
    //printMat( siftWindow, "---------- antes ----------" );
    //siftWindow = siftWindow.mul(kernel);
    //printMat( siftWindow, "---------- depois ----------" );

    // Calculating magnitude and angle of pixels
    std::cout << "getPatchGrads" << std::endl;
    getPatchGrads( siftWindow, dx, dy );
    std::cout << "getPatchGrads ok" << std::endl;

    mag = cv::Mat::zeros( cv::Size(SIFT_DESC_WINDOW+1, SIFT_DESC_WINDOW+1), CV_32FC1 ); // magnitude
    the = cv::Mat::zeros( cv::Size(SIFT_DESC_WINDOW+1, SIFT_DESC_WINDOW+1), CV_32FC1 ); // theta
    
    std::cout << "cartToPolarGradientMat" << std::endl;
    cartToPolarGradientMat( dx, dy, mag, the );

    //printMat( mag, "---------- mag Original ----------" );
    //mag = mag.mul( kernel );
    //printMat( mag, "---------- mag Gaussian ----------" );

    //printMat( dx, "---------- dx ----------" );
    //printMat( dy, "---------- dy ----------" );
    //printMat( mag, "---------- mag Normalized ----------" );
    //printMat( the, "---------- the ----------" );

    // mags that are closer to keypoint should have stronger values
    //siftWindow = siftWindow * kernel;
    mag = mag * 255;// * kernel;
    mag = mag.mul( kernel );// kernel.mul(mag);// * kernel;
    
    // dividir janela em 16 subjanelas 4x4.
    for( int swRows = 0; swRows < SIFT_DESC_SW_QTD; swRows++ )
    {
      for( int swCols = 0; swCols < SIFT_DESC_SW_QTD; swCols++ )
      {
        int rIni, rFim, cIni, cFim;
        int refIniR, refFimR, refIniC, refFimC;
        cv::Mat hist, subMag, subThe;

        subMag = cv::Mat::zeros( cv::Size(SIFT_DESC_SW_SIZE, SIFT_DESC_SW_SIZE),
                                 CV_32FC1 );
        subThe = cv::Mat::zeros( cv::Size(SIFT_DESC_SW_SIZE, SIFT_DESC_SW_SIZE),
                                 CV_32FC1 );

        rIni = SIFT_DESC_SW_SIZE*swRows;
        cIni = SIFT_DESC_SW_SIZE*swCols;
        rFim = rIni+SIFT_DESC_SW_SIZE;
        cFim = cIni+SIFT_DESC_SW_SIZE;

        subMag = mag( cv::Range(rIni, rFim), cv::Range(cIni, cFim) );
        subThe = the( cv::Range(rIni, rFim), cv::Range(cIni, cFim) );

        // adapting references to img coordinates
        if( rIni < SIFT_DESC_WINDOW/2 ) rIni = kpList[i].y-rIni;
        if( cIni < SIFT_DESC_WINDOW/2 ) cIni = kpList[i].x-cIni;
        if( rIni == SIFT_DESC_WINDOW/2 ) rIni = (int) kpList[i].y;
        if( cIni == SIFT_DESC_WINDOW/2 ) cIni = (int) kpList[i].x;
        if( rFim == SIFT_DESC_WINDOW/2 ) rFim = (int) kpList[i].y;
        if( cFim == SIFT_DESC_WINDOW/2 ) cFim = (int) kpList[i].x;
        if( rFim > SIFT_DESC_WINDOW/2 ) rFim = kpList[i].y+rFim;
        if( cFim > SIFT_DESC_WINDOW/2 ) cFim = kpList[i].x+cFim;
        
        //getHistogramForSubregion_old( subMag, subThe, SIFT_DESC_BINS_PER_SW,
        //                          kpList[i].direction, SIFT_DESC_BINS_PER_SW/360, 
        //                          SIFT_DESC_SW_QTD, hist );
        
        getHistogramForSubregion( subMag, subThe, SIFT_DESC_BINS_PER_SW, kpList[i],
                                  cIni, rIni, cFim, rFim, 360/SIFT_DESC_BINS_PER_SW,
                                  SIFT_DESC_SW_QTD, hist );
        
        // adding each bin value to descriptor
        for(int idx = 0; idx<SIFT_DESC_BINS_PER_SW; idx++)
        {
          tempDescriptor.push_back( hist.at<float>(idx) );
          //int hst = (int) hist.at<float>(idx);
          //kpList[i].descriptor.push_back( hst );
        }
      }
    }

    std::cout << std::endl << "Calculated float Descriptor " << i << std::endl;
    for( int k=0; k<tempDescriptor.size(); k++ ) std::cout << tempDescriptor[k] << ", ";
    std::cout << std::endl;

    // Normalizing resulting descriptor
    float sum_square = 0.0f;
    for (int i = 0; i < tempDescriptor.size(); i++)
      sum_square += tempDescriptor[i] * tempDescriptor[i];
    
    float thr = cv::sqrt( sum_square ) * SIFT_DESC_MAG_THR;
    float tmp = 0.0;

    // removing > 0.2 elements after normalized
    sum_square = 0.0;
    for (int i = 0; i < tempDescriptor.size(); i++) {
      tmp = std::fmin(thr, tempDescriptor[i]);
      tempDescriptor[i] = tmp;
      sum_square += tmp * tmp;
    }

    // re-normalizing to get numbers big enough to be converted to int
    float norm_factor = SIFT_DESC_INT_FTR / cv::sqrt( sum_square );
    for (int i = 0; i < tempDescriptor.size(); i++)
      tempDescriptor[i] = tempDescriptor[i] * norm_factor;

    std::cout << std::endl << "Normalized float Descriptor " << i << std::endl;
    for( int k=0; k<tempDescriptor.size(); k++ ) std::cout << tempDescriptor[k] << ", ";
    std::cout << std::endl;

    // converting descriptor to int numbers
    for(int k=0; k < tempDescriptor.size(); k++)
      kpList[i].descriptor.push_back( (int) tempDescriptor[k] );

    std::cout << std::endl << "Calculated int Descriptor " << i << std::endl;
    for( int k=0; k<kpList[i].descriptor.size(); k++ ) std::cout << kpList[i].descriptor[k] << ", ";
    std::cout << std::endl;
  }
}

//*
//calcSIFTDescriptor(img, kp, angle, size*0.5f, d, n, descriptors, i);
void calcSIFTDescriptor( cv::Mat &img, KeyPoints &ptf, float ori, float scl,
                         int d, int n, float* dst )
{
  cv::Point pt(cvRound(ptf.x), cvRound(ptf.y));
  float cos_t = cosf(ori*(float)(CV_PI/180));
  float sin_t = sinf(ori*(float)(CV_PI/180));
  float bins_per_rad = n / 360.f;
  float exp_scale = -1.f/(d * d * 0.5f);
  float hist_width = SIFT_DESC_SCL_FTR * scl;
  int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
  cos_t /= hist_width;
  sin_t /= hist_width;

  int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
  int rows = img.rows, cols = img.cols;

  cv::AutoBuffer<float> buf(len*6 + histlen);
  float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
  float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;
  //float X, Y, Mag, Ori, W, RBin, CBin, hist;

  for( i = 0; i < d+2; i++ )
  {
    for( j = 0; j < d+2; j++ )
      for( k = 0; k < n+2; k++ )
        hist[(i*(d+2) + j)*(n+2) + k] = 0.;
  }

  for( i = -radius, k = 0; i <= radius; i++ )
    for( j = -radius; j <= radius; j++ )
    {
      // Calculate sample's histogram array coords rotated relative to ori.
      // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
      // r_rot = 1.5) have full weight placed in row 1 after interpolation.
      float c_rot = j * cos_t - i * sin_t;
      float r_rot = j * sin_t + i * cos_t;
      float rbin = r_rot + d/2 - 0.5f;
      float cbin = c_rot + d/2 - 0.5f;
      int r = pt.y + i, c = pt.x + j;

      if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
          r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
      {
        float dx = (float)(img.at<float>(r, c+1) - img.at<float>(r, c-1));
        float dy = (float)(img.at<float>(r-1, c) - img.at<float>(r+1, c));
        X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
        W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
        k++;
      }
    }

  len = k;
  cv::hal::fastAtan2(Y, X, Ori, len, true);
  cv::hal::magnitude(X, Y, Mag, len);
  cv::hal::exp(W, W, len);

  for( k = 0; k < len; k++ )
  {
    float rbin = RBin[k], cbin = CBin[k];
    float obin = (Ori[k] - ori)*bins_per_rad;
    float mag = Mag[k]*W[k];

    int r0 = cvFloor( rbin );
    int c0 = cvFloor( cbin );
    int o0 = cvFloor( obin );
    rbin -= r0;
    cbin -= c0;
    obin -= o0;

    if( o0 < 0 )
        o0 += n;
    if( o0 >= n )
        o0 -= n;

    // histogram update using tri-linear interpolation
    float v_r1 = mag*rbin, v_r0 = mag - v_r1;
    float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
    float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
    float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
    float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
    float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
    float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

    int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
    hist[idx] += v_rco000;
    hist[idx+1] += v_rco001;
    hist[idx+(n+2)] += v_rco010;
    hist[idx+(n+3)] += v_rco011;
    hist[idx+(d+2)*(n+2)] += v_rco100;
    hist[idx+(d+2)*(n+2)+1] += v_rco101;
    hist[idx+(d+3)*(n+2)] += v_rco110;
    hist[idx+(d+3)*(n+2)+1] += v_rco111;
  }

  // finalize histogram, since the orientation histograms are circular
  for( i = 0; i < d; i++ )
    for( j = 0; j < d; j++ )
    {
      int idx = ((i+1)*(d+2) + (j+1))*(n+2);
      hist[idx] += hist[idx+n];
      hist[idx+1] += hist[idx+n+1];
      for( k = 0; k < n; k++ )
        dst[(i*d + j)*n + k] = hist[idx+k];
      }
  // copy histogram to the descriptor,
  // apply hysteresis thresholding
  // and scale the result, so that it can be easily converted
  // to byte array
  float nrm2 = 0;
  len = d*d*n;
  for( k = 0; k < len; k++ )
      nrm2 += dst[k]*dst[k];
  float thr = std::sqrt(nrm2)*SIFT_DESC_MAG_THR;
  for( i = 0, nrm2 = 0; i < k; i++ )
  {
    float val = std::min(dst[i], thr);
    dst[i] = val;
    nrm2 += val*val;
  }
  nrm2 = SIFT_DESC_INT_FTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
  for( k = 0; k < len; k++ )
  {
    dst[k] = uchar( dst[k]*nrm2 );// saturate_cast<uchar>(dst[k]*nrm2);
  }
#else
  float nrm1 = 0;
  for( k = 0; k < len; k++ )
  {
    dst[k] *= nrm2;
    nrm1 += dst[k];
  }
  nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
  for( k = 0; k < len; k++ )
  {
    dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
  }
#endif
}

void unpackOpenCVOctave( cv::KeyPoint& kp, int& octave, int& layer, float& scale)
{
  octave = kp.octave & 255;
  layer = (kp.octave >> 8) & 255;
  octave = octave < 128 ? octave : (-128 | octave);
  scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}


//void calcDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
//                          Mat& descriptors, int nOctaveLayers, int firstOctave )
void calcDescriptors(std::vector<KeyPoints> &kpl, cv::Mat &img )
{
  int d = SIFT_DESC_SW_QTD, n = SIFT_DESC_BINS_PER_SW;

  //std::vector<float> descriptors;
  //descriptors.resize(SIFT_DESC_SIZE);
  //float descriptor[SIFT_DESC_SIZE];
  //descriptors.resize(SIFT_DESC_SIZE);

  for ( int i=0; i<kpl.size(); i++ )
  {
    float descriptor[SIFT_DESC_SIZE];
    //KeyPoints kp = kpl[i];
    kpl[i].descriptor.resize(SIFT_DESC_SIZE);
    //for( int k = 0; k < SIFT_DESC_SIZE; k++ )
    //  descriptor[k] = 0.0f;

    //int octave, layer;
    //float scale;
    //unpackOpenCVOctave(kpt, octave, layer, scale);
    //CV_Assert(octave >= firstOctave && layer <= NUM_SCALES+2);
    float size = DOG_BORDER*kpl[i].scale;
    //cv::Point2f ptf(kpt.x*kpt.scale, kpt.y*kpt.scale);
    //const Mat& img = gpyr[(octave - firstOctave)*(NUM_OCTAVES + 3) + layer];

    float angle = 360.f - kpl[i].direction;
    if(std::abs(angle - 360.f) < FLT_EPSILON)
        angle = 0.f;
    calcSIFTDescriptor(img, kpl[i], angle, size*0.5f, d, n, descriptor);

    std::cout << "Keypoint: " << i << std::endl;
    for( int k = 0; k < SIFT_DESC_SIZE; k++ )
    {
      kpl[i].descriptor[k] = descriptor[k];
      std::cout << descriptor[k] << " ";
    }
    std::cout << std::endl;
  }
}
//*/

/**
 * SIFT MAIN METHOD
 * 
 * @param kp KeyPoints detected
 * @param name string with image's name
**/
void siftDescriptor( std::vector<KeyPoints> &kpl, cv::Mat& img_in, cv::Mat& img_gray,
                     int mGauss, float sigma )
{
  cv::Mat img_norm;
  if (img_in.depth() == 0)
  {
    img_in.convertTo(img_norm, CV_32FC1);
    img_norm = img_norm / 255.0f;
  }
  else
  {
    img_in = img_norm / 256.0f;
  }

  //removing blur applied in siftKPOrientation
  cv::cvtColor( img_norm, img_gray, CV_BGR2GRAY ); 

  std::cout << "Calculando orientações" << std::endl;
  siftKPOrientation( kpl, img_gray, mGauss, sigma );
  
  //std::cout << "KeyPoints com calculo de orientação:" << kpl.size() << std::endl;
  //for( int i = 0; i < kpl.size(); i++ )
  //{
  //  std::cout << "kpl[" << i << "].direction: " << kpl[i].direction << std::endl;
  //}

  //removing blur applied in siftKPOrientation
  //cv::cvtColor( img_norm, img_gray, CV_BGR2GRAY ); 

  //calculating keypoints description
  std::cout << "Executando calculo da descrição" << std::endl;
  //siftExecuteDescription( kpl, img_gray );
  calcDescriptors( kpl, img_gray );
  std::cout << "Size da lista de Keypoints  :" << kpl.size() << std::endl;

  //printing keypoints and descriptions
  //for( int i = 0; i < kpl.size(); i++ )
  //  printKeypoint( kpl[i] );
}