#include "../include/detectors/hdr.h"
#include "../include/detectors/aux_func.h"

void coefVar(cv::Mat img, cv::Mat &img_cv, int mask_size, bool gauss, float sigma)
{
  cv::Mat cv = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
  cv::Mat img2 = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
  cv::Mat g_kernel;
  int mask_mid = mask_size / 2;
  float N = float(mask_size * mask_size);
  int kernel_sum;
  float mask_sum, mean;
  double mask_sum2, var, std_dev, coef_var;

  for (int y = 0; y < img.rows; y++)
  {
    for (int x = 0; x < img.cols; x++)
    {
      img2.at<double>(y, x) = img.at<float>(y, x) * img.at<float>(y, x);
    }
  }

  if (gauss)
  {
    gaussKernel(g_kernel, mask_size, sigma);
  }
  else
  {
    kernel_sum = N;
  }

  for (int y = mask_mid; y < (img.rows - mask_mid); y++)
  {
    for (int x = mask_mid; x < (img.cols - mask_mid); x++)
    {
      mask_sum = 0.0;
      mask_sum2 = 0.0;

      for (int r = y - mask_mid, i = 0; r < (y + mask_mid + 1); r++, i++)
      {
        for (int c = x - mask_mid, j = 0; c < (x + mask_mid + 1); c++, j++)
        {
          mask_sum += img.at<float>(r, c);
          if (gauss)
          {
            mask_sum2 += img2.at<double>(r, c) * g_kernel.at<float>(i, j);
          }
          else
          {
            mask_sum2 += img2.at<double>(r, c);
          }
        }
      }
      mean = mask_sum / N ;
      var = (mask_sum2 / N) - (mean * mean * kernel_sum) / N;
      std_dev = sqrt(var);

      if (mean < 1e-5)
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
  cv.convertTo(img_cv, CV_32FC1);
  cv::normalize(img_cv, img_cv, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
}

void logTransform(cv::Mat img, cv::Mat &img_log10)
{
  cv::Mat img_aux, img_ln;
  float ln10 = 2.302585;
  int rmax = 255;

  img_aux = (img * rmax) + 1;
  cv::log(img_aux, img_ln);
  img_log10 = img_ln / ln10;
  img_log10 = img_log10 / std::log10(rmax + 1);
}
