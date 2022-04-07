#include "../include/detectors/aux_func.h"

/**
 * @brief Prints an cv::Mat matrix
 * 
 * @param m Matrix to be printed
 * @param mat_name Matrix name
 */
void printMat( cv::Mat &m, std::string mat_name )
{
    std::cout << mat_name << std::endl;
    for(int i=0; i<m.rows; i++)
    {
      for(int j=0; j<m.cols; j++)
        std::cout << m.at<float>(i,j) << " ";
        
      std::cout << std::endl;
    }
}

/**
 * @brief Clear KeyPoints struct
 * 
 * @param kp struct to be cleared
 */
void cleanKeyPointVector( std::vector<KeyPoints> &kp )
{
  for(int i=0; i < kp.size(); i++)
    if( kp[i].descriptor.size() > 0 )
      kp[i].descriptor.clear();

  kp.clear();
}

/**
 * @brief Textually returns the OpenCV datatype
 * 
 * @param type OpenCV type integer
 * @return std::string string containing datatype
 */
std::string returnOpenCVArrayType(int type) 
{
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

/**
 * If image is grayscale, make a copy;
 * If image is RGB, parse it to grayscale.
 * 
 * @param img: image to be copied to grayscale
 * @param imgOut: output image
**/
void makeGrayscaleCopy( cv::Mat img, cv::Mat &imgOut )
{
  if (img.channels() != 1){ cv::cvtColor(img, imgOut, cv::COLOR_BGR2GRAY); }
  else { img.copyTo(imgOut); }
}

/**
* Reads image in img_path and stores a RGB, Grayscale and image name. 
* In case img_path points to a grayscale image, img_in and img_gray will point to same object 
* img_name will store the image name according to withExtension parameter 
*
* @param img_path: string object path where image is in the disk
* @param img_in: cv::Mat object where RGB version will be stored
* @param img_gray: cv::Mat object where grayscale version will be stored
* @param img_name: string object where the image name will be stored
* @param withExtension: defines if img_name will be store with extension. Default is false.
**/
void readImg( std::string img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, bool withExtension)
{
  img_in = cv::imread(img_path, cv::IMREAD_UNCHANGED);
  std::cout << "reading image " << img_path << std::endl;

  makeGrayscaleCopy( img_in, img_gray );
  //if (img_in.channels() != 1){ cv::cvtColor(img_in, img_gray, cv::COLOR_BGR2GRAY); }
  //else { img_gray = img_in; }
  /*
  double inMax, inMin, grayMax, grayMin, normMin, normMax;
  cv::minMaxLoc( img_in, &inMin, &inMax );
  cv::minMaxLoc( img_gray, &grayMin, &grayMax );
  std::cout << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ " << std::endl;
  std::cout << "img type: " << img_in.depth() << std::endl;
  std::cout << "in_min: " << inMin << ", in_max: " << inMax << std::endl;
  std::cout << "gray_min: " << grayMin << ", gray_max: " << grayMax << std::endl;
  std::cout << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ " << std::endl;
  */
  img_name = getFileName(img_path, withExtension);
}

/**
 * @brief Reads image 
 * 
 * @param img_path path to image
 * @param img_in where original image will be returned
 */
void readImg( std::string img_path, cv::Mat &img_in )
{
  cv::Mat gray;
  std::string imgName;

  readImg(img_path, img_in, gray, imgName);
}

/**
 * @brief Reads image
 * 
 * @param img_path path to image
 * @param img_in original image
 * @param img_gray grayscale image from img_in
 * @param img_name image name
 * @param withExtension flag to tell if img_name must be returned with or without extension
 */
void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, bool withExtension)
{
  std::string strImgPath = std::string(img_path);
  readImg(strImgPath, img_in, img_gray, img_name, withExtension);
}

/**
 * @brief Reads the Homography matrix
 * 
 * @param path path to homography matrix
 * @param H where homography matrix will be returned
 */
void readHomographicMatrix( std::string path, cv::Mat &H )
{
  if( !path.empty() )
  {
    H = cv::Mat( cv::Size(3,3), CV_32F );
    std::fstream arch( path, std::ios_base::in );

    float aux;
    for( int i = 0; i < 3; i++ )
      for( int j = 0; j < 3; j++ )
      {
        arch >> aux;
        H.at<float>(i,j) = aux;
      }

    arch.close();
  }
}

/**
 * @brief Reads camera matrix 
 * 
 * @param path path to camera path
 * @param cam where the camera matrix will be returned
 */
void readCameraMatrix( std::string path, cv::Mat &cam )
{
  cam = cv::Mat( cv::Size(3,3), CV_32F );
  std::fstream arch( path, std::ios_base::in );

  float aux;
  for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ )
    {
      arch >> aux;
      cam.at<float>(i,j) = aux;
    }

  arch.close();
}

/**
 * @brief Get the File Name object
 * 
 * @param file_path path to file
 * @param withExtension if the filename returned is returned with or without extension
 */
std::string getFileName(std::string file_path, bool withExtension)
{
  size_t size = file_path.rfind("/", file_path.length());

  if (size != std::string::npos)
  {
    file_path = file_path.substr(size + 1, file_path.length() - size);
  }
  else
  {
    file_path = "";
  }

  if( withExtension ) 
  {
    return file_path;
  }
  else
  {
    size = file_path.rfind(".", file_path.length());
    
    if (size != std::string::npos)
    {
      return file_path.substr(0, size);
    }
    else
    {
      return file_path;
    }
  }
}

int sciToDec(const std::string &str)
{
	std::stringstream ss(str);
  double num = 0;

	ss >> num;

  return((int) std::round(num));
}

std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);

	 while (std::getline(tokenStream, token, delimiter))
   {
      token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
			tokens.push_back(token);
   }
   return tokens;
}

/**
 * @brief Return a vector of cv::Point that delimits the polygon of the ROI
 * 
 * @param roi_path path to polygon that delimits ROI
 * @param verts object where return the vector of cv::Point
 */
void readROI(std::string roi_path, std::vector<cv::Point> &verts)
{
  std::fstream arch( roi_path, std::ios_base::in );
  double aux, m[6][2];

  //std::cout << "    >  reading " << roi_path << std::endl;
  
  // Populating matrix
  for( int i = 0; i < 6; i++ )
    for( int j = 0; j < 2; j++ )
    {
      arch >> aux;
      m[i][j] = aux;
    }

  for(int i=0; i<6; i++)
  {
    cv::Point p;
    p.x = std::floor( m[i][0] );
    p.y = std::floor( m[i][1] );
    verts.push_back( p );

    //std::cout << "    >  x: " << p.x << " y: " << p.y << std::endl;
  }

  arch.close();
}

/**
 * @brief apply ROI in image
 * 
 * @param img image to be segmented
 * @param pathROI path to ROI
 */
void applyROI( cv::Mat &img, std::string pathROI )
{
  cv::Mat imgROI, aux;
  std::vector<cv::Point> vecROI, ROIPolygon;

  readROI( pathROI, vecROI );
  imgROI = cv::Mat::zeros( img.size(), CV_8UC1 );

  cv::approxPolyDP( vecROI, ROIPolygon, 1.0, true );
  cv::fillConvexPoly( imgROI, &ROIPolygon[0], ROIPolygon.size(), cv::Scalar::all(255), 8, 0);
  //cv::fillPoly( imgROI, vecROI, cv::Scalar::all(255) );

  img.copyTo( aux, imgROI );
  aux.copyTo( img );
  std::cout << "  > ### Images masked out." << std::endl;

  aux.release();
}

/**
 * @brief Create ROI image object using a polygon of (x, y) points;
 * 
 * @param pathROI path to (x,y) point polygon
 * @param img image to be segmented
 * @param roi ROI image
 */
void readROIAsImage( std::string pathROI, cv::Mat img, cv::Mat &roi )
{
  std::vector<cv::Point> vecROI, ROIPolygon;

  readROI( pathROI, vecROI );
  roi = cv::Mat::zeros( img.size(), CV_8UC1 );

  cv::approxPolyDP( vecROI, ROIPolygon, 1.0, true );
  cv::fillConvexPoly( roi, &ROIPolygon[0], ROIPolygon.size(), cv::Scalar::all(255), 8, 0);
}

/**
 * @brief Create a ROI image object using an existing image.
 * 
 * @param pathROI path to image
 * @param img image where ROI will be returned
 */
void readROIFromImage( std::string pathROI, cv::Mat &img )
{
  readImg( pathROI, img );
}

/**
 * @brief segment image using a rectangle as "Region Of Interest". v1 and v2 are the extrema points of rectangle.
 * 
 * @param img image to be segmented
 * @param img_roi ROI image, created using v1 and v2
 * @param v1 First extrema of the rectangle
 * @param v2 Second extrema of the rectangle
 */
void selectROI(cv::Mat img, cv::Mat &img_roi, cv::Point v1, cv::Point v2)
{
	cv::Rect roi = cv::Rect(v1, v2);
	img_roi = img(roi);
}

/**
 * @brief Create gaussian window based with size and sigma sent in argument.
 * 
 * @param kernel where to return gaussian window
 * @param size window size
 * @param sigma sigma value
 */
void gaussKernel(cv::Mat &kernel, int size, float sigma)
{
	int k_mid = size / 2;
	cv::Size k_size = cv::Size(size, size);
	cv::Mat dirac = cv::Mat::zeros(k_size, CV_32FC1);

	dirac.at<float>(k_mid, k_mid) = 1.0;
	cv::GaussianBlur(dirac, kernel, k_size, sigma, sigma, cv::BORDER_REPLICATE);
}

/**
 * Linearly maps the Matrix values to [0.0, 1.0] interval.
 * 
 * @param mat: original image
 * @param img_out: resulting image, with mapped pixel values
 * 
**/
void mapPixelValues0_1( cv::Mat &img, cv::Mat &img_out )
{
  cv::normalize( img, img_out, 0.0f, 1.0f, cv::NORM_MINMAX, img_out.depth(), cv::noArray() );
}

/**
 * Linearly maps the Matrix values to [0.0, 255.0] interval.
 * 
 * @param mat: original image
 * @param img_out: resulting image, with mapped pixel values
 * 
**/
void mapPixelValues0_255( cv::Mat &img, cv::Mat &img_out )
{
  cv::normalize( img, img_out, 0.0f, 255.0f, cv::NORM_MINMAX, img_out.depth(), cv::noArray() );
}

/**
 * Linearly maps the Matrix values to a interval.
 * 
 * @param mat: original image
 * @param img_out: resulting image, with mapped pixel values
 * 
**/
void mapPixelValues( cv::Mat &img, cv::Mat &img_out, int mapInterval )
{
  // If empty, initialize output image
  if( img_out.empty() )
    if( img.channels() == 1 ) img_out = cv::Mat::zeros( cv::Size( img.rows, img.cols ), CV_32F );
    else img_out = cv::Mat::zeros( cv::Size( img.rows, img.cols ), CV_32FC3 );

  if( mapInterval == MAPPING_INTERVAL_FLOAT_0_1 )
    mapPixelValues0_1( img, img_out ); // maps to [0.0, 1.0] float interval
  else if( mapInterval == MAPPING_INTERVAL_FLOAT_0_255 )
    mapPixelValues0_255(img, img_out); // maps to [0.0, 255.0] float interval
  else
    mapPixelValues0_255(img, img_out); // if mapInterval does not exist, calls method with default mapping interval
}

/**
 * Gets the Image 1 mapped to image 2 position
 * 
 * @param imgIn: The reference image;
 * @param imgOut: Image transformed using homographic matrix;
 * @param pathMatrix: Path to homography matrix archive.
**/
void getHomographicCorrespondence( cv::Mat imgIn, cv::Mat &imgOut, cv::Mat H )
{
  cv::warpPerspective( imgIn, imgOut, H, imgIn.size() );
}

/**
 * Gets the Image 1 mapped to image 2 position
 * 
 * @param imgIn: The reference image;
 * @param imgOut: Image transformed using homographic matrix;
 * @param pathMatrix: Path to homography matrix archive.
**/
void getHomographicCorrespondence( cv::Mat imgIn, cv::Mat &imgOut, std::string pathMatrix )
{
  cv::Mat H;

  readHomographicMatrix( pathMatrix, H );
  getHomographicCorrespondence( imgIn, imgOut, H );
}


/**
 * Gets the p1, from Image 1, mapped in the Image 2 using p2 argument.
 * 
 * @param p1: A point in Image 1;
 * @param p2: Argument where to store the p1 point mapped in image 2;
 * @param pathMatrix: Path to homography matrix archive.
 * 
 * | x1 |   | x2 |   | h1 h2 h3 |
 * | y1 | = | y2 | X | h4 h5 h6 | = 1x3 * 3x3 = 3x3
 * | 01 |   | 01 |   | h7 h8 h9 |
**/
void getHomographicCorrespondence( cv::Point2f p1, cv::Point2f &p2, cv::Mat H )
{
  cv::Mat pta = cv::Mat( 3, 1, CV_32F );//, mp2;

  pta.at<float>(0,0) = p1.x;
  pta.at<float>(1,0) = p1.y;
  pta.at<float>(2,0) = 1.0f;

  cv::Mat ptb = H * pta;
  
  p2.x  = ptb.at<float>(0,0) / ptb.at<float>(2,0);
  p2.y  = ptb.at<float>(1,0) / ptb.at<float>(2,0);

//  std::cout << "Xa: " << p1.x << ", Ya: " << p1.y << std::endl;
//  std::cout << "Xb: " << p2.x << ", Yb: " << p2.y << std::endl;
//  std::cout << "Scale: " << ptb.at<float>(2,0) << std::endl;

//  printMat(pta, "Point 1");
//  printMat(ptb, "Point 2");
//  printMat(H, "Homography Matrix");
}

/**
 * @brief Get the Homographic Correspondence of a point
 * 
 * @param x1: x coordinate of point 1
 * @param y1: y coordinate of point 1
 * @param x2: x coordinate of point 2
 * @param y2: y coordinate of point 2
 * @param H: Homographic matrix
 */
void getHomographicCorrespondence( float x1, float y1, float &x2, float &y2, cv::Mat H )
{
  cv::Point2f p1, p2;
  p1.x = x1;
  p1.y = y1;

  getHomographicCorrespondence( p1, p2, H );

  x2 = p2.x;
  y2 = p2.y;
}

/**
 * @brief create a vector that all KeyPoints inside lKps in kps.
 * 
 * @param lKps vector containing of KeyPoints vectors
 * @param kps vector of KeyPoints
 */
void joinKeypoints( std::vector< std::vector<KeyPoints> > lKps, std::vector<KeyPoints> &kps )
{
  std::cout << "## Joining Vectors of Keypoints into single vector of KeyPoints" << std::endl;

  for(int i = 0; i < lKps.size(); i++)
  {
    std::vector<KeyPoints> kpVec = lKps[i];
    for(int j = 0; j < kpVec.size(); j++)
    {
      KeyPoints kp = kpVec[j];
      kps.push_back( kp );

      //std::cout << "X: " << kp.x << ", " << "Y: " << kp.y << std::endl;
    }
  }
}

/**
 * @brief Metlhod that sums all cv::Mat objects inside matList. Result is stored into result object.
 * 
 * @param matList: vector of Mat objects
 * @param result cv::Mat object that contains the sum of the matList cv::Mat objects
 */
void sumListOfMats( std::vector< cv::Mat > matList, cv::Mat &result )
{
  result = cv::Mat::zeros( matList[0].size(), matList[0].type() );

  for( int i = 0; i < matList.size(); i++ )
    cv::add( result, matList[i], result );
}
