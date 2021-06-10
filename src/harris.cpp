#include "../include/detectors/harris.h"
#include "../include/detectors/keypoint.h"
#include "../include/detectors/hdr.h"
#include "../include/detectors/aux_func.h"

void harrisCalc(cv::Mat img, cv::Mat &resp_map, int msobel, int mgauss, float sigma_x,
                float sigma_y, float k)
{
  cv::Mat Ix, Iy, Ixx, Iyy, Ixy, resp_aux;

  cv::Sobel(img, Ix, CV_32FC1, 1, 0, msobel);
  cv::Sobel(img, Iy, CV_32FC1, 0, 1, msobel);

  Ixx = Ix.mul(Ix);
  Iyy = Iy.mul(Iy);
  Ixy = Ix.mul(Iy);

  cv::GaussianBlur(Ixx, Ixx, cv::Size(mgauss, mgauss), sigma_x, sigma_y);
  cv::GaussianBlur(Iyy, Iyy, cv::Size(mgauss, mgauss), sigma_x, sigma_y);
  cv::GaussianBlur(Ixy, Ixy, cv::Size(mgauss, mgauss), sigma_x, sigma_y);

  resp_aux = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32FC1);

  for (int y = 0; y < img.rows; y++)
  {
    for (int x = 0; x < img.cols; x++)
    {
      float dxx = Ixx.at<float>(y, x);
      float dyy = Iyy.at<float>(y, x);
      float dxy = Ixy.at<float>(y, x);
      float detH = (dxx * dyy) - (dxy * dxy);
      float traceH = (dxx + dyy);

      resp_aux.at<float>(y, x) = detH - k * (traceH * traceH);
    }
  }
  resp_map = resp_aux;
}

void harrisThreshold(cv::Mat &resp_map, std::vector<KeyPoints> &kp, float min_quality)
{
  double min, max;

  cv::minMaxIdx(resp_map, &min, &max);
  double threshold = max * min_quality;

  for (int y = 0; y < resp_map.rows; y++)
  {
    for (int x = 0; x < resp_map.cols; x++)
    {
      if (resp_map.at<float>(y, x) >= threshold)
      {
        kp.push_back({float(y), float(x), resp_map.at<float>(y, x)});
      }
      else
      {
        resp_map.at<float>(y, x) = 0;
      }
    }
  }
}

void harrisMaxSup(cv::Mat &resp_map, std::vector<KeyPoints> &kp, int msize)
{
  std::vector<KeyPoints> kp_aux;
  cv::Mat resp_aux = cv::Mat::zeros(resp_map.size(), CV_32F);

  for (int k = 0; k < (int)kp.size(); k++)
  {
    bool is_max = true;
    int y = kp[k].y;
    int x = kp[k].x;
    float kp_ref = kp[k].resp;
    int mradius = msize / 2;

    for (int i = y - mradius; i <= y + mradius; i++)
    {
      for (int j = x - mradius; j <= x + mradius; j++)
      {
        if (!outOfBounds(i, j, resp_map.size()))
        {
          if (kp_ref < resp_map.at<float>(i, j))
          {
            is_max = false;
            break;
          }
        }
      }
    }
    if (is_max)
    {
      resp_aux.at<float>(y, x) = kp_ref;
      kp_aux.push_back({float(y), float(x), kp_ref, 0, 0});
    }
  }
  resp_map = resp_aux;
  kp.clear();
  kp = kp_aux;
}

void harrisKp(cv::Mat img, std::vector<KeyPoints> &kp, bool is_hdr, int msobel,
              int mgauss, float sigma_x, float sigma_y, float k, float min_quality,
              int msup_size, int cv_size)
{
  cv::Mat resp_map, img_norm, img_blur, img_aux;

  imgNormalize(img, img_norm);

  cv::GaussianBlur(img_norm, img_blur, cv::Size(mgauss, mgauss), sigma_x, sigma_y,
                   cv::BORDER_REPLICATE);

  if (is_hdr)
  {
    cv::Mat img_cv, img_log;

    coefVar(img_blur, img_cv, cv_size);
    logTransform(img_cv, img_log);

    img_aux = img_log;
  }
  else
  {
    img_aux = img_blur;
  }
  harrisCalc(img_aux, resp_map, msobel, mgauss, sigma_x, sigma_y, k);
  harrisThreshold(resp_map, kp, min_quality);
  harrisMaxSup(resp_map, kp, msup_size);
}
