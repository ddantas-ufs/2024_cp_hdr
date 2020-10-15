#include "surf.h"

void surfKp(cv::Mat img, std::vector<KeyPoints> &kp, cv::Mat roi[])
{
    cv::Mat img_sum;
    cv::integral(img, img_sum);

    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            std::cout << (int)pow(2, (i + 1))*(j + 1) + 1 << "\n";
}