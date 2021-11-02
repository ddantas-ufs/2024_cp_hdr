#include "../include/descriptors/sift.h"
#include "../include/detectors/dog.h"

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

void cartToPolarGradientMat( float dy[], float dx[], float thes[], float exps[], int size )
{ 
  for( int i = 0; i < size; i++ )
  {
    float x = dx[i], y = dy[i];

    float exp = std::exp( exps[i] );
    float the = cv::fastAtan2(y, x);

    float m = (x*x) + (y*y);
    m = cv::sqrt( m );
    
    dy[i] = m;
    thes[i] = the;
    exps[i] = exp;
  }
}

void getGradient( cv::Mat& img, int x, int y, float mt[2] )
{
  int xm = x-1, xp = x+1;
  int ym = y-1, yp = y+1;
  float dy, dx;

  // Extrapolating image borders
  if( xp > img.cols ) xp = img.cols - std::abs(xp-img.cols) - 1;
  if( yp > img.rows ) yp = img.rows - std::abs(yp-img.rows) - 1;
  if( xm < 0 ) xm = std::abs( xm );
  if( ym < 0 ) ym = std::abs( ym );

  float dxa = 0.0f, dxb = 0.0f, dya = 0.0f, dyb = 0.0f;
  dxa = img.at<float>( y, xp );
  dxb = img.at<float>( y, xm );
  dya = img.at<float>( yp, x );
  dyb = img.at<float>( ym, x );

  if( std::isnan( dxa ) ) dxa = 0.0f;
  if( std::isnan( dxb ) ) dxb = 0.0f;
  if( std::isnan( dya ) ) dya = 0.0f;
  if( std::isnan( dyb ) ) dyb = 0.0f;

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
//  cv::Mat auxImg;
//  img.copyTo( auxImg ); // Allocating auxImg to be equal to img

  cv::GaussianBlur( img, img, cv::Size(mGauss, mGauss), sigma, sigma, 
                    cv::BORDER_REPLICATE );

  for( int i = 0; i < kp.size(); i++ )
    calcOrientation( img, kp[i] );
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
**//*
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
*/

/**
 * Receive the histogram array and descriptor array, normalize the values and store
 * the normalized values into descriptor array.
 * 
 * @param hist histogram array with calculated bins
 * @param descriptor descriptor array where normalized values will be stored
 * @param qtdSW the amount of subWindows calculated (default = 4)
 * @param binsPerSW the amount of bins calculated per each subWindow (default = 8)
**/
void normalizeHistToDescriptor(float* hist, float* descriptor, int qtdSW, int binsPerSW )
{
  int k = 0;
  
  // Preparing to normalize histogram
  float nrm2 = 0.0f;
  int len = qtdSW*qtdSW*binsPerSW;
  for( k = 0; k < len; k++ )
    nrm2 += descriptor[k]*descriptor[k];

  float thr = std::sqrt(nrm2)*SIFT_DESC_MAG_THR;
  nrm2 = 0.0f;

  for( int i = 0; i < k; i++ )
  {
    float val = std::min(descriptor[i], thr);
    descriptor[i] = val;
    nrm2 += val*val;
  }

  nrm2 = SIFT_DESC_INT_FTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

  // Copying normalized data to descriptor
  for( k = 0; k < len; k++ )
    descriptor[k] = uchar( descriptor[k]*nrm2 );
}

void calcKeypointDescriptor( cv::Mat &img, KeyPoints &kp, float angle, float scale,
                             float* descriptor )
{
  int qtdSW = SIFT_DESC_SW_QTD, binsPerSW = SIFT_DESC_BINS_PER_SW;
  
  // get sin and cos to rotate angle
  float angleCos = cosf(angle*(float)(M_PI/180.0f));
  float angleSin = sinf(angle*(float)(M_PI/180.0f));
  float bins_per_rad = binsPerSW / 360.0f;
  float exp_scale = -1.0f/( qtdSW*qtdSW*0.5f ); // -1/8

  // radius size according to keypoint's scale
  float hist_width = SIFT_DESC_SCL_FTR * scale;
  int radius = cvRound(hist_width * sqrtf(2) * (qtdSW + 1) * 0.5f);

  // value to interpolate sin and cos with complete histogram width length
  angleCos /= hist_width;
  angleSin /= hist_width;

  // calculating histogram segment and auxiliar arrays length
  int i, j, k;
  int len = (radius*2+1)*(radius*2+1); // size based on gaussian kernel to weight values
  int histlen = (qtdSW+2)*(qtdSW+2)*(binsPerSW+2); // real size is 288 but usable is 128

  // auxiliar arrays and histogram arrays
  float arr_dx[len], arr_dy[len], arr_angles[len], exponencials[len], rowBins[len];
  float colBins[len], hist[histlen];
  float *mag = arr_dy;

  // setting histogram to 0.0
  for( i = 0; i < qtdSW+2; i++ )
    for( j = 0; j < qtdSW+2; j++ )
      for( k = 0; k < binsPerSW+2; k++ )
        hist[(i*(qtdSW+2) + j)*(binsPerSW+2) + k] = 0.0f;

  // populating arrays with their respective values
  for( i = -radius, k = 0; i <= radius; i++ )
    for( j = -radius; j <= radius; j++ )
    {
      // rotate sin and cos relatively to dominant keypoint's angle.
      // if rbin or cbin are half, we need to make it fall into integer, decreasing 0.5. 
      float colRot = j * angleCos - i * angleSin;
      float rowRot = j * angleSin + i * angleCos;
      float rowBin = rowRot + qtdSW/2.0f - 0.5f;
      float colBin = colRot + qtdSW/2.0f - 0.5f;
      
      int r = cvRound(kp.y) + i;
      int c = cvRound(kp.x) + j;

      // execute only if values are inside of image sizes
      if( rowBin > -1 && rowBin < qtdSW && colBin > -1 && colBin < qtdSW &&
          r > 0 && r < img.rows - 1 && c > 0 && c < img.cols - 1 )
      {
        float dx = (float)(img.at<float>(r, c+1) - img.at<float>(r, c-1));
        float dy = (float)(img.at<float>(r-1, c) - img.at<float>(r+1, c));
        
        // exponencial = ( ((j*cos)-(x*sin))² + ((j*cos)+(x*sin))² ) * (-1/8)
        exponencials[k] = (colRot * colRot + rowRot * rowRot) * exp_scale;

        // populating auxiliar arrays
        arr_dx[k] = dx;
        arr_dy[k] = dy;
        rowBins[k] = rowBin;
        colBins[k] = colBin;
        k++;
      }
    }

  len = k;

  // populate auxiliar arr_angles, exponencials and recalculate arr_dy
  cartToPolarGradientMat(arr_dy, arr_dx, arr_angles, exponencials, len);
  
  // each loop calculates an 8-bin histogram
  for( k = 0; k < len; k++ )
  {
    float rbin = rowBins[k];
    float cbin = colBins[k];
    float abin = (arr_angles[k] - angle) * bins_per_rad;
    float magv = mag[k] * exponencials[k];

    // integer part
    int introw = cvFloor( rbin );
    int intcol = cvFloor( cbin );
    int intang = cvFloor( abin );

    // real part
    rbin -= introw;
    cbin -= intcol;
    abin -= intang;

    // [0, 359] range
    if( intang < 0 ) intang += binsPerSW;
    if( intang >= binsPerSW ) intang -= binsPerSW;

    // row interpolation
    float row1 = magv * rbin;
    float row0 = magv - row1;

    // column interpolation
    float row1_col1 = row1 * cbin;
    float row1_col0 = row1 - row1_col1;
    float row0_col1 = row0 * cbin;
    float row0_col0 = row0 - row0_col1;

    // angle interpolation
    float row1_col1_ang1 = row1_col1 * abin;
    float rol1_col1_ang0 = row1_col1 - row1_col1_ang1;
    float row1_col0_ang1 = row1_col0 * abin;
    float row1_col0_ang0 = row1_col0 - row1_col0_ang1;
    float row0_col1_ang1 = row0_col1 * abin;
    float row0_col1_ang0 = row0_col1 - row0_col1_ang1;
    float row0_col0_ang1 = row0_col0 * abin;
    float row0_col0_ang0 = row0_col0 - row0_col0_ang1;

    // calculate index ((row+1) * (4+2) + col+1) * (8+2) + angle
    // 4 + 2 = subWindow + 2
    // 8 + 2 = window + 2
    int idx = ((introw+1)*(qtdSW+2) + intcol+1)*(binsPerSW+2) + intang;

    // incrementing points on histogram
    hist[idx] += row0_col0_ang0;
    hist[idx+1] += row0_col0_ang1;
    hist[idx+(binsPerSW+2)] += row0_col1_ang0;
    hist[idx+(binsPerSW+3)] += row0_col1_ang1;
    hist[idx+(qtdSW+2)*(binsPerSW+2)] += row1_col0_ang0;
    hist[idx+(qtdSW+2)*(binsPerSW+2)+1] += row1_col0_ang1;
    hist[idx+(qtdSW+3)*(binsPerSW+2)] += rol1_col1_ang0;
    hist[idx+(qtdSW+3)*(binsPerSW+2)+1] += row1_col1_ang1;
  }

  // finalize histogram, since the orientation histograms are circular
  for( i = 0; i < qtdSW; i++ )
    for( j = 0; j < qtdSW; j++ )
    {
      int idx = ((i+1)*(qtdSW+2) + (j+1))*(binsPerSW+2);
      hist[idx] += hist[idx+binsPerSW];
      hist[idx+1] += hist[idx+binsPerSW+1];
      for( k = 0; k < binsPerSW; k++ )
        descriptor[(i*qtdSW + j)*binsPerSW + k] = hist[idx+k];
    }
  
  normalizeHistToDescriptor( hist, descriptor, qtdSW, binsPerSW );
}

void calcDescriptors(std::vector<KeyPoints> &kpl, cv::Mat &img )
{
  // FOR EACH KP IN LIST...
  for ( int i=0; i<kpl.size(); i++ )
  {
    float descriptor[SIFT_DESC_SIZE];
    kpl[i].descriptor.resize(SIFT_DESC_SIZE);

    float size = DOG_BORDER*kpl[i].scale;

    float angle;
    if(std::abs(kpl[i].direction) < FLT_EPSILON)
      angle = 0.f;
    else
      angle = angle = 360.f - kpl[i].direction;

    calcKeypointDescriptor( img, kpl[i], angle, size*0.5f, descriptor);

    //std::cout << "Keypoint: " << i << std::endl;
    for( int k = 0; k < SIFT_DESC_SIZE; k++ )
    {
      kpl[i].descriptor[k] = descriptor[k];
      //std::cout << descriptor[k] << " ";
    }
    //std::cout << std::endl;
  }
}

/**
 * SIFT DESCRIPTOR MAIN METHOD
 * 
 * @param kp KeyPoints detected
 * @param name string with image's name
**/
void siftDescriptor( std::vector<KeyPoints> &kpl, cv::Mat& img_in, cv::Mat& img_gray,
                     int mGauss, float sigma )
{
  cv::Mat img_norm;
  mapPixelValues(img_in, img_norm);

  // calculating keypoints orientation
  std::cout << " ## SIFT > > Calculating orientations..." << std::endl;
  siftKPOrientation( kpl, img_gray, mGauss, sigma );
  std::cout << " ## SIFT > > Keypoints orientations computed." << std::endl;

  //Removing blur applied in siftKPOrientation
  //cv::cvtColor( img_norm, img_gray, cv::COLOR_BGR2GRAY ); 
  makeGrayscaleCopy( img_norm, img_gray );
  
  //Calculating keypoints description
  std::cout << " ## SIFT > > Calculating description..." << std::endl;
  calcDescriptors( kpl, img_gray );
  std::cout << " ## SIFT > > descriptions computed." << std::endl;
}

/**
 * This method runs complete sift pipeline, executing dog and sift descriptor.
 * 
 * @param img: image where keypoints will be detected
 * @param kpList: output vector containing detected keypoints and description.
**/
void runSift( cv::Mat img, std::vector<KeyPoints> &kpList )
{
  cv::Mat imgGray;

  makeGrayscaleCopy( img, imgGray );

  std::cout << " ## SIFT > Detecting Keypoints..." << std::endl;
  dogKp(imgGray, kpList);
  std::cout << " ## SIFT > " << kpList.size() << " Keypoints detected." << std::endl;

  std::cout << " ## SIFT > Describing Keypoints..." << std::endl;
  siftDescriptor(kpList, img, imgGray);
  std::cout << " ## SIFT > Keypoints described." << std::endl;

  imgGray.release();
}