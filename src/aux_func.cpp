#include "../include/detectors/aux_func.h"

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


void cleanKeyPointVector( std::vector<KeyPoints> &kp )
{
  for(int i=0; i < kp.size(); i++)
    if( kp[i].descriptor.size() > 0 )
      kp[i].descriptor.clear();

  kp.clear();
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

void readImg( std::string img_path, cv::Mat &img_in )
{
  cv::Mat gray;
  std::string imgName;

  readImg(img_path, img_in, gray, imgName);
}

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, bool withExtension)
{
  std::string strImgPath = std::string(img_path);
  readImg(strImgPath, img_in, img_gray, img_name, withExtension);
}

void readHomographicMatrix( std::string path, cv::Mat &H )
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

void readROI(std::string roi_path, std::vector<cv::Point> &verts)
{
  std::fstream arch( roi_path, std::ios_base::in );
  double aux, m[6][2];

  std::cout << "    >  reading " << roi_path << std::endl;
  
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

    std::cout << "    >  x: " << p.x << " y: " << p.y << std::endl;
  }

  arch.close();
}

void applyROI( cv::Mat &img, std::string pathROI )
{
  cv::Mat imgROI, aux;
  std::vector<cv::Point> vecROI, ROIPolygon;

  readROI( pathROI, vecROI );

  imgROI = cv::Mat::zeros( img.size(), img.type() );

  std::cout << "  > ### img size: " << img.size() << std::endl;
  std::cout << "  > ### imgROI size: " << imgROI.size() << std::endl;

  cv::approxPolyDP( vecROI, ROIPolygon, 1.0, true );
  cv::fillConvexPoly( imgROI, &ROIPolygon[0], ROIPolygon.size(), cv::Scalar::all(255), 8, 0);
  //cv::fillPoly( imgROI, vecROI, cv::Scalar::all(255) );
  std::cout << "  > ### imgROI filled up" << std::endl;

  img.copyTo( aux, imgROI );
  aux.copyTo( img );
  std::cout << "  > ### Images masked out." << std::endl;

  aux.release();
}

void selectROI(cv::Mat img, cv::Mat &img_roi, cv::Point v1, cv::Point v2)
{
	cv::Rect roi = cv::Rect(v1, v2);
	img_roi = img(roi);
}

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
  cv::Mat mp1 = cv::Mat( 1, 3, CV_32F );//, mp2;

  mp1.at<float>(0,0) = p1.x;
  mp1.at<float>(0,1) = p1.y;
  mp1.at<float>(0,2) = 1.0f;

//  std::cout << "---> mp1: " << mp1.size() << std::endl;
//  std::cout << "---> mp1: " << H.size() << std::endl;

  cv::Mat mp2 = mp1 * H;
  
  p2.x = mp2.at<float>(0,0);
  p2.y = mp2.at<float>(0,1);

//  printMat(mp1, "Point 1");
//  printMat(mp2, "Point 2");
//  printMat(H, "Homography Matrix");
}

void getHomographicCorrespondence( float x1, float y1, float &x2, float &y2, cv::Mat H )
{
  cv::Point2f p1, p2;
  p1.x = x1;
  p1.y = y1;

  getHomographicCorrespondence( p1, p2, H );

  x2 = p2.x;
  y2 = p2.y;
}