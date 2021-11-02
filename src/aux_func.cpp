#include "../include/detectors/aux_func.h"

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name)
{
  img_in = cv::imread(img_path, cv::IMREAD_UNCHANGED);

  if (img_in.channels() != 1)
  {
    cv::cvtColor(img_in, img_gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    img_gray = img_in;
  }
  img_name = getFileName(std::string(img_path));
}

std::string getFileName(std::string file_path)
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