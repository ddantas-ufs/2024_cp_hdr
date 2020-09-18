#include <bits/stdc++.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "dog.h"

void initOctaves(cv::Mat img, cv::Mat scales[SCALES_ROWS][SCALES_COLS], int mgauss)
{	
	cv::Mat img_aux;
	float k[] = {0.707107, 1.414214, 2.828428, 5.656856};
	
	img.convertTo(img_aux, CV_32FC1);
	
	for(int i = 0; i < SCALES_ROWS; i++)
	{
		float ko = k[i];
		for(int j = 0; j < SCALES_COLS; j++)
		{			
			GaussianBlur(img_aux, scales[i][j], cv::Size(mgauss, mgauss), ko, ko, cv::BORDER_DEFAULT);
			ko = ko*1.414214;
		}
		cv::resize(img_aux, img_aux, cv::Size(img_aux.cols/2, img_aux.rows/2));
	}
}

void calcDoG(cv::Mat scales[SCALES_ROWS][SCALES_COLS], cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1])
{
	for(int i = 0; i < SCALES_ROWS; i++)
		for(int j = 0; j < SCALES_COLS - 1; j++)
		{
			dog[i][j] = cv::Mat::zeros(scales[i][j].size(), CV_32FC1);
			cv::subtract(scales[i][j], scales[i][j + 1], dog[i][j]);
		}
}

void localMaxSup(cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1], cv::Mat roi[], std::vector<KeyPoint> &kp, int msize)
{
	int mradius = msize/2;

	for(int s = 0; s < SCALES_ROWS; s++)
	{
		for(int l = 1; l < SCALES_COLS - 1; l++)
		{
			cv::Mat middle = dog[s][l];
			cv::Mat down = dog[s][l - 1];
			cv::Mat up = dog[s][l + 1];
			cv::Mat dog_aux = cv::Mat::zeros(middle.size(), CV_32FC1);
			
			for(int y = mradius; y < middle.rows - mradius; y++)
			{
				for(int x = mradius; x < middle.cols - mradius; x++)
				{
					if(roi[0].at<uchar>(y*pow(2, s), x*pow(2, s)) == 0)
						continue;
					
					float curr_px = middle.at<float>(y, x);
					bool is_smaller = true;
					bool is_bigger = true;
					
					for(int i = y - mradius; i <= y + mradius; i++)
					{
						for(int j = x - mradius; j <= x + mradius; j++)
						{
							if(!((curr_px < middle.at<float>(i, j) || (y == i && x == j)) && 
								 (curr_px < down.at<float>(i, j)) &&
								 (curr_px < up.at<float>(i, j))))
							{
								is_smaller = false;
								break;
							}
						}
						if(!is_smaller)
							break;
					}
					for(int i = y - mradius; i <= y + mradius; i++)
					{
						for(int j = x - mradius; j <= x + mradius; j++)
						{
							if(!((curr_px > middle.at<float>(i, j) || (y == i && x == j)) && 
								 (curr_px > down.at<float>(i, j)) && 
								 (curr_px > up.at<float>(i, j))))
							{
								is_bigger = false;
								break;
							}
						}
						if(!is_bigger)
							break;
					}
					if(is_smaller || is_bigger)
						kp.push_back({y, x, curr_px, s, l});
						
				}
			}
		}
	}
}

void contrastThreshold(std::vector<KeyPoint> &kp, cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1], float threshold)
{
	std::vector<KeyPoint> kp_aux;
	
	for(int i = 0; i < kp.size(); i++)
	{
		if(kp[i].resp >= threshold)
			kp_aux.push_back(kp[i]);	
	}
	kp.clear();
	kp = kp_aux;
}

void edgeThreshold(std::vector<KeyPoint> &kp, cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1], float curv_th)
{
	std::vector<KeyPoint> kp_aux;
	curv_th = (curv_th + 1)*(curv_th + 1)/curv_th;
	
	for(int i = 0; i < kp.size(); i++)
	{
		cv::Mat D = dog[kp[i].scale][kp[i].level];
		
		int y = kp[i].y;
		int x = kp[i].x;
		
		float dxx = D.at<float>(y - 1, x) + D.at<float>(y + 1, x) - 2.0*D.at<float>(y, x);
		float dyy = D.at<float>(y, x - 1) + D.at<float>(y, x + 1) - 2.0*D.at<float>(y, x);
		float dxy = 0.25*(D.at<float>(y - 1, x - 1) + D.at<float>(y + 1, x + 1) - D.at<float>(y + 1, x - 1) - D.at<float>(y - 1, x + 1));

		float trH = dxx*dyy;
		float detH = dxx*dyy - dxy*dxy;

		float curv_ratio = trH*trH/detH;
		
		if((detH > 0) && (curv_ratio > curv_th))
			kp_aux.push_back(kp[i]);
	}
	kp.clear();
	kp = kp_aux;
}