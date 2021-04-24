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

void getPatchGrads( cv::Mat& subImage, cv::Mat& retX, cv::Mat& retY )
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
 * @param refAngle angle of reference to calculate the histogram
 * @param binWidth width of the bin
 * @param subW width of the subregion
 * @param hist Matrix where the histogram is being returned
**/
void getHistogramForSubregion( cv::Mat &mag, cv::Mat &theta, int numBin, int refAngle,
                               int binWidth, int subW, cv::Mat& hist )
{
  float minimum = 0.000001f;
  float center = (subW/2.0f) - 0.5f;
  //hist = cv::Mat::zeros( cv::Size(numBin, 1), CV_32SC1 );
  hist = cv::Mat::zeros( cv::Size(numBin, 1), CV_32FC1 );
  
  cv::Mat arrMag = cv::Mat::zeros( cv::Size(mag.rows*mag.cols, 1), CV_32FC1 );
  cv::Mat arrThe = cv::Mat::zeros( cv::Size(theta.rows*theta.cols, 1), CV_32FC1 );

  returnRavel( mag, arrMag );
  printMat( arrMag, "---------- arrMag ----------" );
  
  returnRavel( theta, arrThe );
  printMat( arrThe, "---------- arrThe ----------" );

  for( int i=0; i<arrMag.cols; i++ )
  {
    float mg = arrMag.at<float>(i);
    int angle = (int) (arrThe.at<float>(i)-refAngle) % 360;
    int b = quantizeOrientation(angle, numBin);
    float vote = mg;

    //std::cout << "mg: " << mg << ", angle: " << angle << ", b: " << b << std::endl;
    //std::cout << "vote1: " << vote << std::endl;

    // b*binWidth is the start angle of the histogram bin
    // b*binWidth+binWidth/2 is the center of the histogram bin
    // angle -[...] is the distance from the angle to the center of the bin 
    //float histInterpWeight = 1-std::abs(angle-((b*binWidth)+(binWidth/2)))/(binWidth/2);
    //std::cout << "histInterpWeight: " << histInterpWeight << std::endl;
    //vote = vote * std::max( histInterpWeight, minimum );
    //std::cout << "vote2: " << vote << std::endl;

    //int idx[2];
    //float xInterpWeight, yInterpWeight;
    //unravelIndex( i, subW, subW, idx );

    //xInterpWeight = std::max( (float) 1-(std::abs(idx[0]-center)/center), minimum );
    //yInterpWeight = std::max( (float) 1-(std::abs(idx[1]-center)/center), minimum );
    //std::cout << "xInterpWeight: " << xInterpWeight << std::endl;
    //std::cout << "yInterpWeight: " << yInterpWeight << std::endl;

    //vote = vote * (xInterpWeight * yInterpWeight);
    //std::cout << "vote3: " << vote << std::endl;
    hist.at<float>(b) += /*vote **/ angle;
  }
  std::cout << "--> histSubregion: ";
  //for(int k = 0; k<numBin; k++) std::cout << hist.at<int>(k) << " ";
  for(int k = 0; k<numBin; k++) std::cout << hist.at<float>(k) << " ";
  std::cout << std::endl;
}

void siftExecuteDescription( std::vector<KeyPoints> &kpList, cv::Mat &img )
{
  cv::Mat siftWindow, kernel;
  /*
  double minVal, maxVal;
  cv::Point minLoc, maxLoc;

  cv::minMaxLoc( img, &minVal, &maxVal, &minLoc, &maxLoc );
  std::cout << "minVal: " << minVal << std::endl;
  std::cout << "minLoc: " << minLoc << std::endl;
  std::cout << "maxVal: " << maxVal << std::endl;
  std::cout << "maxLoc: " << maxLoc << std::endl;

  if( maxVal <= 1.0f ) img = img * 255;

  cv::minMaxLoc( img, &minVal, &maxVal, &minLoc, &maxLoc );
  std::cout << "minVal: " << minVal << std::endl;
  std::cout << "minLoc: " << minLoc << std::endl;
  std::cout << "maxVal: " << maxVal << std::endl;
  std::cout << "maxLoc: " << maxLoc << std::endl;

  std::system("read -p \"Pressione enter para sair\" saindo");
  */
  // Calculates a 17x17 Gaussian kernel
  gaussianKernel( SIFT_DESC_WINDOW+1, SIFT_DESC_ORIENT_SIGMA, kernel );

  for( int i = 0; i < kpList.size(); i++ )
  {
    cv::Mat mag, the, dx, dy;
    int swHalf = (int) SIFT_DESC_WINDOW / 2;

    // Generating 17x17 window (16x16 window) + keypoint's row and column
    siftWindow = cv::Mat::zeros( SIFT_DESC_WINDOW+1, SIFT_DESC_WINDOW+1, CV_32FC1 );
    int swRows = 0;
    for( int rows = kpList[i].y-swHalf; rows < kpList[i].y+swHalf; rows++ )
    {
      int swCols = 0;
      for( int cols = kpList[i].x-swHalf; cols < kpList[i].x+swHalf; cols++ )
      {
        int x = cols, y = rows;

        // Extrapolating image borders        
        if( x > img.cols ) x = img.cols-(x - img.cols)-1;
        if( y > img.rows ) y = img.rows-(y - img.rows)-1;
        if( x < 0 ) x = std::abs( x );
        if( y < 0 ) y = std::abs( y );

        //std::cout << "----> X: " << x << ", Y: " << y << std::endl;
        //std::cout << "----> swRows: " << swRows << ", swCols: " << swCols << std::endl;
        siftWindow.at<float>(swRows, swCols) = img.at<float>(y, x);
        swCols = swCols + 1;
      }
      swRows = swRows + 1;
    }

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

    //printMat( dx, "---------- dx ----------" );
    //printMat( dy, "---------- dy ----------" );
    //printMat( mag, "---------- mag ----------" );
    //printMat( the, "---------- the ----------" );

    // mags that are closer to keypoint should have stronger values
    //siftWindow = siftWindow * kernel;
    mag = kernel.mul(mag);// * kernel;
    
    // dividir janela em 16 subjanelas 4x4.
    for( int swRows = 0; swRows < SIFT_DESC_SW_QTD; swRows++ )
    {
      for( int swCols = 0; swCols < SIFT_DESC_SW_QTD; swCols++ )
      {
        cv::Mat hist, subMag, subThe;

        subMag = cv::Mat::zeros( cv::Size(SIFT_DESC_SW_SIZE, SIFT_DESC_SW_SIZE),
                                 CV_32FC1 );
        subThe = cv::Mat::zeros( cv::Size(SIFT_DESC_SW_SIZE, SIFT_DESC_SW_SIZE),
                                 CV_32FC1 );

        int rIni = SIFT_DESC_SW_SIZE*swRows;
        int cIni = SIFT_DESC_SW_SIZE*swCols;
        int a = 0, b = 0;

        std::cout << "rIni: " << rIni << " rFim: " << rIni+SIFT_DESC_SW_SIZE << std::endl;
        std::cout << "cIni: " << cIni << " cFim: " << cIni+SIFT_DESC_SW_SIZE << std::endl;

        //for( int i=rIni; i<rIni+SIFT_DESC_SW_SIZE; i++ )
        //{
        //  for( int j=cIni; j<cIni+SIFT_DESC_SW_SIZE; j++ )
        //  {
        //    std::cout << "i: " << i << " j: " << j << std::endl;
        //    std::cout << "a: " << a << " b: " << b << std::endl;
        //    subMag.at<float>(a, b) = mag.at<float>(i, j);
        //    subThe.at<float>(a, b) = the.at<float>(i, j);
        //    b++;
        //  }
        //  a++;
        //  b = 0;
        //}

        // LEMBRAR DE AJUSTAR PARA NAO PEGAR LINHA/COLUNA DO KP
        subMag = mag( cv::Range(rIni, rIni+SIFT_DESC_SW_SIZE), 
                      cv::Range(cIni, cIni+SIFT_DESC_SW_SIZE) );
        subThe = the( cv::Range(rIni, rIni+SIFT_DESC_SW_SIZE), 
                      cv::Range(cIni, cIni+SIFT_DESC_SW_SIZE) );
        
        printMat( subMag, "---------- subMag ----------" );
        subMag = subMag * 255;
        printMat( subMag, "---------- subMag 2 ----------" );
        printMat( subThe, "---------- subThe ----------" );
        // calculate 8-bin histogram for subwindow
        getHistogramForSubregion( subMag, subThe, SIFT_DESC_BINS_PER_SW, kpList[i].direction,
                                  360/SIFT_DESC_BINS_PER_SW, SIFT_DESC_SW_QTD, hist );
        
        // adding each bin value to descriptor
        for(int idx = 0; idx<SIFT_DESC_BINS_PER_SW; idx++)
        {
          int hst = (int) hist.at<float>(idx);
          kpList[i].descriptor.push_back( hst );
        }
      }
    }
    std::cout << std::endl << "descritor calculado " << i << std::endl;
    for( int k=0; k<kpList[i].descriptor.size(); k++ ) std::cout << kpList[i].descriptor[k] << ", ";
    std::cout << std::endl;
  }
}

/**
 * SIFT MAIN METHOD
 * 
 * @param kp KeyPoints detected
 * @param name string with image's name
**/
void siftDescriptor( std::vector<KeyPoints> &kp, cv::Mat& img_in, cv::Mat& img_gray,
                     int mGauss, float sigma )
{
  cv::Mat img_norm;
  if (img_in.depth() == 0)
  {
    img_in.convertTo(img_norm, CV_32FC1);
    img_norm = img_norm / 255.0;
  }
  else
  {
    img_in = img_norm / 256.0;
  }

  //removing blur applied in siftKPOrientation
  cv::cvtColor( img_norm, img_gray, CV_BGR2GRAY ); 

  std::cout << "Calculando orientações" << std::endl;
  siftKPOrientation( kp, img_gray, mGauss, sigma );
  
  //std::cout << "KeyPoints com calculo de orientação:" << kp.size() << std::endl;
  //for( int i = 0; i < kp.size(); i++ )
  //{
  //  std::cout << "kp[" << i << "].direction: " << kp[i].direction << std::endl;
  //}

  //removing blur applied in siftKPOrientation
  cv::cvtColor( img_norm, img_gray, CV_BGR2GRAY ); 

  //calculating keypoints description
  std::cout << "Executando calculo da descrição" << std::endl;
  siftExecuteDescription( kp, img_gray );
  std::cout << "Size da lista de Keypoints  :" << kp.size() << std::endl;

  //printing keypoints and descriptions
  //for( int i = 0; i < kp.size(); i++ )
  //  printKeypoint( kp[i] );
}