#include "../include/detectors/hdr.h"

void coefVar(cv::Mat img, cv::Mat &img_cv, int mask_size)
{
  cv::Mat cv = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
  cv::Mat img2 = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
  cv::Mat img_aux;
  int mask_mid = mask_size / 2;
  int N = mask_size * mask_size;
  float mask_sum, mean;
  double mask_sum2, var, std_dev, coef_var;

  if (img.depth() == CV_8UC1)
  {
    img.convertTo(img_aux, CV_32FC1);
  }
  else
  {
    img_aux = img;
  }

  for (int y = 0; y < img.rows; y++)
  {
    for (int x = 0; x < img.cols; x++)
    {
      img2.at<double>(y, x) = img_aux.at<float>(y, x) * img_aux.at<float>(y, x);
    }
  }
  for (int y = mask_mid; y < (img.rows - mask_mid); y++)
  {
    for (int x = mask_mid; x < (img.cols - mask_mid); x++)
    {
      mask_sum = 0.0;
      mask_sum2 = 0.0;

      for (int i = y - mask_mid; i < (y + mask_mid + 1); i++)
      {
        for (int j = x - mask_mid; j < (x + mask_mid + 1); j++)
        {
          mask_sum += img_aux.at<float>(i, j);
          mask_sum2 += img2.at<double>(i, j);
        }
      }
      mean = mask_sum / (N * 1.0);
      var = (mask_sum2 / (N * 1.0)) - (mean * mean);
      std_dev = sqrt(var);

      if (mean < 1e-10)
      {
        coef_var = 0.0;
      }
      else
      {
        coef_var = std_dev / mean;
      }
      cv.at<double>(y, x) = coef_var;
    }
  }
  cv::normalize(cv, img_cv, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());
}

void logTransform(cv::Mat img, cv::Mat &img_log)
{
  cv::Mat img_aux;
  float ln10 = 2.302;

  img.convertTo(img_aux, CV_32FC1);
  img_aux = img_aux + 1;
  cv::log(img_aux, img_aux);

  img_log = img_aux * (1.0 / ln10);

  cv::normalize(img_log, img_log, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());
}
