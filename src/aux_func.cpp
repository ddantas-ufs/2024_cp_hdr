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
void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, bool withExtension)
{
  img_in = cv::imread(img_path, cv::IMREAD_UNCHANGED);
  std::cout << "reading image " << img_path << std::endl;

  if (img_in.channels() != 1){ cv::cvtColor(img_in, img_gray, cv::COLOR_BGR2GRAY); }
  else { img_gray = img_in; }
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
  img_name = getFileName(std::string(img_path), withExtension);
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
	std::fstream file;
	std::string line;
	std::vector<std::string> tokens;
	int x, y;

	file.open(roi_path, std::ios::in);

	while (std::getline(file, line))
	{
		tokens = split(line, '\t');
		x = sciToDec(tokens[0]);
		y = sciToDec(tokens[1]);
		verts.push_back(cv::Point(x, y));
	}
	file.close();
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

void imgNormalize(cv::Mat img, cv::Mat &img_norm)
{
  if (img.depth() == 0)
  {
    img.convertTo(img_norm, CV_32FC1);
    img_norm = img_norm / LDR_MAX_RANGE;
  }
  else
  {
    img_norm = img / HDR_MAX_RANGE;
  }
}

void unpackOpenCVOctave(const cv::KeyPoint &kpt, int &octave, int &layer, float &scale)
{
  octave = kpt.octave & 255;
  layer = (kpt.octave >> 8) & 255;
  octave = octave < 128 ? octave : (-128 | octave);
  scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

void importOpenCVKeyPoints( std::vector<cv::KeyPoint> &ocv_kp, cv::Mat &descriptor,
                            std::vector<KeyPoints> &kpList, bool comDescritor )
{
  std::cout << "Importing " << ocv_kp.size() << " Keypoints ";
  if( comDescritor ) std::cout << "with description" << std::endl;
  else std::cout << "without description" << std::endl;

  for(int i=0; i<ocv_kp.size(); i++)
  {
    KeyPoints nkp;
    int uOctave, uLayer;
    float uScale;

    unpackOpenCVOctave( ocv_kp[i], uOctave, uLayer, uScale );

    nkp.x = (float) ocv_kp[i].pt.x;
    nkp.y = (float) ocv_kp[i].pt.y;
    nkp.resp = (float) ocv_kp[i].response;
    nkp.direction = (float) ocv_kp[i].angle;
    nkp.octave = (int) uOctave;
    nkp.scale = (int) uLayer;

    //std::cout << "X, Y: " << nkp.x << ", " << nkp.y << std::endl;
    //std::cout << "Resp: " << uLayer << std::endl;
    //std::cout << "Octv: " << nkp.octave << std::endl;
    //std::cout << "Angl: " << nkp.direction << std::endl;
    //std::cout << "Scal: " << nkp.scale << std::endl;

    if( comDescritor )
    {
      for( int j=0; j < descriptor.cols; j++ )
      {
        float d = descriptor.at<float>(j, i);
        nkp.descriptor.push_back( (int) d );
      }
    }
    kpList.push_back(nkp);
  }

}

/*
  Pack info into octave variable
  Octave integer variable stores 2 informations 
  that are stored in the first 2 binary octets.
  + xxxxxxxx xxxxxxxx llllllll oooooooo -
  Where:
    x = Unused octets
    l = Layer information octet
    o = Octave information octet (less significant octet)
*/
void packOpenCVOctave(const cv::KeyPoint &kpt, int &octave, int &layer )
{
  int oct = octave;
  int aux = layer;
  aux = aux << 8; // pass first octet to second octet
  aux = aux | ( 15 && oct ); // Octave is the first octet

  octave = oct;
}

void exportToOpenCVKeyPointsObject( std::vector<KeyPoints> &kpList, std::vector<cv::KeyPoint> &ocv_kp )
{
  std::cout << "Exporting " << kpList.size() << " Keypoints " << std::endl;

  for(int i=0; i<kpList.size(); i++)
  {
    //std::cout << "kp " << i << std::endl;
    cv::KeyPoint nkp;
    int oct = kpList[i].octave;
    int lay = kpList[i].scale;

    packOpenCVOctave( ocv_kp[i], oct, lay );

    nkp.pt.x = kpList[i].x;
    nkp.pt.y = kpList[i].y;
    nkp.response = kpList[i].resp;
    nkp.angle = kpList[i].direction;
    nkp.octave = (int) oct;

    /*
    nkp.x = (float) ocv_kp[i].pt.x;
    nkp.y = (float) ocv_kp[i].pt.y;
    nkp.resp = (float) ocv_kp[i].response;
    nkp.direction = (float) ocv_kp[i].angle;
    nkp.octave = (int) uOctave;
    nkp.scale = (int) uLayer;
    */

    //std::cout << "X, Y: " << nkp.x << ", " << nkp.y << std::endl;
    //std::cout << "Resp: " << uLayer << std::endl;
    //std::cout << "Octv: " << nkp.octave << std::endl;
    //std::cout << "Angl: " << nkp.direction << std::endl;
    //std::cout << "Scal: " << nkp.scale << std::endl;

    //std::cout << "push_back 1" << std::endl;;
    ocv_kp.push_back(nkp);
    //std::cout << "push_back 2" << std::endl;;
  }
}

/**
 * Maps the value t from the range [a, b] to [c, d].
 * @param a: minimum value origin range
 * @param b: maximum value origin range
 * @param c: minimum value destination range
 * @param d: maximum value destination range
 * @param t: value to be mapped
 * 
 * @return float value in the range [c, d].
**/
float f_t( float a, float b, float c, float d, float t )
{
  return c + ( ( d-c ) / ( b-a ) ) * ( t-a );
}

/**
 * Maps the Matrix values from the range oriRange to desRange.
 * @param mat: original image
 * @param img_out: resulting image, with mapped pixel values
 * @param oriRange: original range (defaults to 0, 255)
 * @param desRange: destination range (defaults to 0, 1)
 * 
 * @return float value in the range [c, d].
**/
void f_t( float a, float b, float c, float d, cv::Mat &img, cv::Mat &img_out )
{
  float frac = ( d-c ) / ( b-a );
  cv::subtract( img, a, img_out, cv::noArray(), img_out.depth() );
  
  img_out.forEach<float>([&](float& element, const int position[]) -> void
  { element *= frac; });

  cv::add( img_out, c, img_out, cv::noArray(), img_out.depth() );
}

/**
 * Maps the Matrix values from the range oriRange to desRange.
 * @param mat: original image
 * @param img_out: resulting image, with mapped pixel values
 * @param oriRange: original range (defaults to 0, 255)
 * @param desRange: destination range (defaults to 0, 1)
 * 
 * @return float value in the range [c, d].
*
void mapMatValues( cv::Mat &mat, cv::Mat &mat_out, float oriRange[], float desRange[] )
{
  for( int i = 0; i < mat.rows; i++ )
    for( int j = 0; j < mat.cols; i++ )
    {
      if( mat.channels() == 1 )
      {
        mat_out = cv::Mat::zeros( cv::Size( mat.rows, mat.cols ), CV_32F );
        float p = float( mat.at<int>(i,j) + 0.0f );
        mat_out.at<float>(i,j) = f_t( oriRange[0], oriRange[1], desRange[0], desRange[1], p );
      }
      else
      {
        mat_out = cv::Mat::zeros( cv::Size( mat.rows, mat.cols ), CV_32FC3 );

        float b = float( mat.at<cv::Vec3b>(i,j)[0] + 0.0f ); // erro aqui
        float g = float( mat.at<cv::Vec3b>(i,j)[1] + 0.0f );
        float r = float( mat.at<cv::Vec3b>(i,j)[2] + 0.0f );

        mat_out.at<cv::Vec3f>(i,j)[0] = f_t( oriRange[0], oriRange[1], desRange[0], desRange[1], b );
        mat_out.at<cv::Vec3f>(i,j)[1] = f_t( oriRange[0], oriRange[1], desRange[0], desRange[1], g );
        mat_out.at<cv::Vec3f>(i,j)[2] = f_t( oriRange[0], oriRange[1], desRange[0], desRange[1], r );
      }
    }
}
**/

void mapPixelValues01( cv::Mat &img, cv::Mat &img_out )
{
  double imgMin, imgMax;
  cv::minMaxLoc( img, &imgMin, &imgMax );

  // initialize output image
  if( img.channels() == 1 ) img_out = cv::Mat::zeros( cv::Size( img.rows, img.cols ), CV_32F );
  else img_out = cv::Mat::zeros( cv::Size( img.rows, img.cols ), CV_32F );

  // float oriRange[2] = { float(imgMin), float(imgMax)}, desRange[2] = {0.0f, 1.0f};
  // mapMatValues(img, img_out, oriRange, desRange);
  f_t( imgMin, imgMax, 0, 1, img, img_out );
}