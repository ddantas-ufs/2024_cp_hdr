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
void readImg( std::string img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, bool withExtension)
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
  img_name = getFileName(img_path, withExtension);
}

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, bool withExtension)
{
  std::string strImgPath = std::string(img_path);
  readImg(strImgPath, img_in, img_gray, img_name, withExtension);
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
 * Linearly maps the Matrix values to [0.0, 1.0] interval.
 * 
 * @param mat: original image
 * @param img_out: resulting image, with mapped pixel values
 * 
**/
void mapPixelValues01( cv::Mat &img, cv::Mat &img_out )
{
  // If empty, initialize output image
  if( img_out.empty() )
    if( img.channels() == 1 ) img_out = cv::Mat::zeros( cv::Size( img.rows, img.cols ), CV_32F );
    else img_out = cv::Mat::zeros( cv::Size( img.rows, img.cols ), CV_32FC3 );

  cv::normalize( img, img_out, 0.0f, 1.0f, cv::NORM_MINMAX, img_out.depth(), cv::noArray() );
}

/**
 * Linearly maps the Matrix values to [0.0, 255.0] interval.
 * 
 * @param mat: original image
 * @param img_out: resulting image, with mapped pixel values
 * 
**/
void mapPixelValues0255( cv::Mat &img, cv::Mat &img_out )
{
  // If empty, initialize output image
  if( img_out.empty() )
    if( img.channels() == 1 ) img_out = cv::Mat::zeros( cv::Size( img.rows, img.cols ), CV_32F );
    else img_out = cv::Mat::zeros( cv::Size( img.rows, img.cols ), CV_32FC3 );

  cv::normalize( img, img_out, 0.0f, 255.0f, cv::NORM_MINMAX, img_out.depth(), cv::noArray() );
}