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
        KeyPoints k;
        k.x = x;
        k.y = y;
        k.scale = 1;
        k.resp = resp_map.at<float>(y,x);
        kp.push_back(k);
      }
      else
      {
        resp_map.at<float>(y, x) = 0.0f;
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
      //kp_aux.push_back({float(y), float(x), kp_ref, 0, 0});

      KeyPoints k;
      k.x = x;
      k.y = y;
      k.scale = 1;
      k.resp = kp_ref;
      kp_aux.push_back(k);
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

  mapPixelValues(img, img_norm);

  cv::GaussianBlur(img_norm, img_blur, cv::Size(mgauss, mgauss), sigma_x, sigma_y,
                   cv::BORDER_REPLICATE);

  if (is_hdr)
  {
    cv::Mat img_cv, img_log;

    coefficienceOfVariationMask( img_blur, img_cv );
    logTranformUchar( img_cv, img_log );

    img_aux = img_log;
  }
  else
  {
    img_aux = img_blur;
  }

  mapPixelValues(img_aux, img_aux, MAPPING_INTERVAL_FLOAT_0_1);

  std::cout << " ## HARRIS > > Calculating Harris Keypoints..." << std::endl;
  harrisCalc(img_aux, resp_map, msobel, mgauss, sigma_x, sigma_y, k);

  std::cout << " ## HARRIS > > Thresolding..." << std::endl;
  harrisThreshold(resp_map, kp, min_quality);
  std::cout << " ## HARRIS > > Total amount of Keypoints Founded: " << kp.size() << "." << std::endl;

  std::cout << " ## HARRIS > > Making Maxima Suppression..." << std::endl;
  harrisMaxSup(resp_map, kp, msup_size);
  std::cout << " ## HARRIS > > Keypoints after Maxima supression: " << kp.size() << "." << std::endl;
}

void harrisKp( cv::Mat img, std::vector< std::vector<KeyPoints> > &kpList, std::vector<cv::Mat> lRoi, bool is_hdr )
{
  //cv::Mat sumROI;
  std::vector<KeyPoints> allKps;
  int cont = 0;

  //sumListOfMats( lRoi, sumROI );
  harrisKp(img, allKps, is_hdr );

  std::cout << " ## HARRIS > Computing Keypoints inside ROI." << std::endl;
  // Separing Keypoints found in each ROI
  for( int i = 0; i < lRoi.size(); i++ )
  {
    std::vector<KeyPoints> kps;

    for( int j = 0; j < allKps.size(); j++ )
    {
      KeyPoints kp = allKps[j];
      int x = (int) std::floor( kp.x );
      int y = (int) std::floor( kp.y );

      uchar pixelValue = lRoi[i].at<uchar>(y, x);
      if( pixelValue > 0 )
        kps.push_back( kp );
    }
    
    // Saving position i ROI keypoints
    kpList.push_back( kps );    
    std::cout << " ## Keypoints inside ROI " << i << ": " << kps.size() << std::endl;
  }
}