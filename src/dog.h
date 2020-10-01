#include "keypoint.h"

#define DOG_SCL_ROWS 4
#define DOG_SCL_COLS 5

void dogInitScales(cv::Mat img, cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS], int mgauss);
void dogCalc(cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS], cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1]);
void dogMaxSup(cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], cv::Mat roi[], std::vector<KeyPoint> &kp, int maxsup_size);
void dogThreshold(std::vector<KeyPoint> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float contrast_th, float curv_th);
void contrastThreshold(std::vector<KeyPoint> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float contrast_th);
void edgeThreshold(std::vector<KeyPoint> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float curv_th);