#include "../include/detectors/hdr.h"

void coefVar(cv::Mat img, cv::Mat &img_cv, int mask_size)
{
    cv::Mat img2 = img.mul(img);
    int mask_mid = mask_size/2;
    int N = mask_size*mask_size;
    
    if(img_cv.depth() != CV_32FC1)
        img_cv = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

    for(int y = mask_mid; y < (img.rows - mask_mid); y++)
    {
        for(int x = mask_mid; x < (img.cols - mask_mid); x++)
        {
            float mask_sum = 0, mask_sum2 = 0;
            for(int i = y - mask_mid; i < y + mask_mid; i++)
                for(int j = x - mask_mid; j < x + mask_mid; j++)
                {
                    mask_sum += img.at<float>(i, j);
                    mask_sum2 += img2.at<float>(i, j);
                }
            float mask_mean = mask_sum/N;
            float var = (mask_sum2/N) - (mask_mean*mask_mean);
            float std_dev = sqrt(var);
            
            float coef_var;
            if(mask_mean == 0)
                coef_var = 0;
            else
                coef_var = std_dev/N;

            img_cv.at<float>(y, x) = coef_var;
        }
    }
    cv::normalize(img_cv, img_cv, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());
}

void logTransform(cv::Mat img, cv::Mat &img_log)
{
    if(img_log.depth() != CV_8UC1)
        img_log = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    
    cv::Mat img_aux;
    cv::add(img, 1, img_aux);
    cv::log(img_aux, img_log);
}