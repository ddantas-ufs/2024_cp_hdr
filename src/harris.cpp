#include <bits/stdc++.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "harris.h"

void responseCalc(cv::Mat img, cv::Mat &resp_map, cv::Mat roi[], int msobel, int mgauss, int k)
{
	cv::GaussianBlur(img, img, cv::Size(mgauss, mgauss), 0, 0, cv::BORDER_DEFAULT);
	
	cv::Mat Ix, Iy, Ixx, Iyy, Ixy;

	cv::Sobel(img, Ix, CV_32F, 1, 0, msobel);
	cv::Sobel(img, Iy, CV_32F, 0, 1, msobel);

	Ixx = Ix.mul(Ix);
	Iyy = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);
	
	cv::GaussianBlur(Ixx, Ixx, cv::Size(mgauss, mgauss), 0, 0);
	cv::GaussianBlur(Iyy, Iyy, cv::Size(mgauss, mgauss), 0, 0);
	cv::GaussianBlur(Ixy, Ixy, cv::Size(mgauss, mgauss), 0, 0);
	
	resp_map = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32F);
	
	for(int y = 0; y < img.rows; y++)
	{
		for(int x = 0; x < img.cols; x++)
		{
			if(roi[0].at<uchar>(y, x) != 0)
			{
				float dxx = Ixx.at<float>(y, x);
				float dyy = Iyy.at<float>(y, x);
				float dxy = Ixy.at<float>(y, x);
				float detH = (dxx*dyy) - (dxy*dxy);
				float traceH = (dxx + dyy);
				
				resp_map.at<float>(y, x) = detH - K*(traceH*traceH);
			}
		}
	}
}

void threshold(cv::Mat &resp_map, std::vector<KeyPoint> &kp, float min_quality)
{
	std::vector<KeyPoint> kp_aux;
	double min, max;
	
	cv::minMaxIdx(resp_map, &min, &max);
	double threshold = max*min_quality;

	for(int y = 0; y < resp_map.rows; y++)
		for(int x = 0; x < resp_map.cols; x++)
			if(resp_map.at<float>(y, x) >= threshold)
				kp.push_back({y, x, resp_map.at<float>(y, x)});
			else
				resp_map.at<float>(y, x) = 0;
			
}

void localMaxSup(cv::Mat &resp_map, std::vector<KeyPoint> &kp, int msize)
{	 
	std::vector<KeyPoint> kp_aux;
	cv::Mat resp_aux = cv::Mat::zeros(resp_map.size(), CV_32F);
	
	for(int k = 0; k < (int)kp.size(); k++)
	{
		bool is_max = true;
		int y = kp[k].y;
		int x = kp[k].x;
		float kp_ref = kp[k].resp;
		int mradius = msize/2;

		for(int i = y - mradius; i <= y + mradius; i++)
		{
			for(int j = x - mradius; j <= x + mradius; j++)
			{
				if(!outOfBounds(i, j, resp_map.size()))
				{
					if(kp_ref < resp_map.at<float>(i, j)){
						is_max = false;
						break;
					}
				}
			}
		}		
		if(is_max)
		{
			resp_aux.at<float>(y, x) = kp_ref;
			kp_aux.push_back({y, x, kp_ref, 0, 0});
		}
	}
	resp_map = resp_aux;
	kp.clear();
	kp = kp_aux;
}