#include "../include/detectors/dog.h"

void dogInitScales(cv::Mat img, cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS], int mgauss)
{	
	cv::Mat img_aux;
	float k[] = {0.707107, 1.414214, 2.828428, 5.656856};
	
	img.convertTo(img_aux, CV_32FC1);
	
	for(int i = 0; i < DOG_SCL_ROWS; i++)
	{
		float ko = k[i];
		for(int j = 0; j < DOG_SCL_COLS; j++)
		{			
			GaussianBlur(img_aux, scales[i][j], cv::Size(mgauss, mgauss), ko, ko, cv::BORDER_DEFAULT);
			ko = ko*1.414214;
		}
		cv::resize(img_aux, img_aux, cv::Size(img_aux.cols/2, img_aux.rows/2));
	}
}

void dogCalc(cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS], cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1])
{
	for(int i = 0; i < DOG_SCL_ROWS; i++)
		for(int j = 0; j < DOG_SCL_COLS - 1; j++)
		{
			dog[i][j] = cv::Mat::zeros(scales[i][j].size(), CV_32FC1);
			cv::subtract(scales[i][j], scales[i][j + 1], dog[i][j]);
		}
}

void dogMaxSup(cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], cv::Mat roi[], std::vector<KeyPoints> &kp, int maxsup_size)
{
	int maxsup_rad = maxsup_size/2;

	for(int s = 0; s < DOG_SCL_ROWS; s++)
	{
		for(int l = 1; l < DOG_SCL_COLS - 1; l++)
		{
			cv::Mat middle = dog[s][l];
			cv::Mat down = dog[s][l - 1];
			cv::Mat up = dog[s][l + 1];
			cv::Mat dog_aux = cv::Mat::zeros(middle.size(), CV_32FC1);
			
			for(int y = maxsup_rad; y < middle.rows - maxsup_rad; y++)
			{
				for(int x = maxsup_rad; x < middle.cols - maxsup_rad; x++)
				{
					if(roi[0].at<uchar>(y*pow(2, s), x*pow(2, s)) == 0)
						continue;
					
					float curr_px = middle.at<float>(y, x);
					bool is_smaller = true;
					bool is_bigger = true;
					
					for(int i = y - maxsup_rad; i <= y + maxsup_rad; i++)
					{
						for(int j = x - maxsup_rad; j <= x + maxsup_rad; j++)
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
					for(int i = y - maxsup_rad; i <= y + maxsup_rad; i++)
					{
						for(int j = x - maxsup_rad; j <= x + maxsup_rad; j++)
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

void contrastThreshold(std::vector<KeyPoints> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float contrast_th)
{
	std::vector<KeyPoints> kp_aux;
	
	for(int i = 0; i < kp.size(); i++)
	{
		if(kp[i].resp >= contrast_th)
			kp_aux.push_back(kp[i]);	
	}
	kp.clear();
	kp = kp_aux;
}

void edgeThreshold(std::vector<KeyPoints> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float curv_th)
{
	std::vector<KeyPoints> kp_aux;
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

void dogThreshold(std::vector<KeyPoints> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float contrast_th, float curv_th)
{
	contrastThreshold(kp, dog, contrast_th);
	edgeThreshold(kp, dog, curv_th);
}

void dogKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoints> &kp, int mgauss, int maxsup_size, float contrast_th, float curv_th)
{
	cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS];
	cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1];

	dogInitScales(img, scales, mgauss);
	dogCalc(scales, dog);
	dogMaxSup(dog, roi, kp, maxsup_size);
	dogThreshold(kp, dog, contrast_th, curv_th);
}