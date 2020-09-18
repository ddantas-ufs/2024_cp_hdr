#define CONTRAST_TH 0.03
#define CURV_TH 5
#define MASK_NMAXSUP 3
#define MASK_GAUSSIAN 11
#define SCALES_ROWS 4
#define SCALES_COLS 5

#include "keypoint.h"

void initOctaves(cv::Mat img, cv::Mat scales[SCALES_ROWS][SCALES_COLS], int mgauss = MASK_GAUSSIAN);
void calcDoG(cv::Mat scales[SCALES_ROWS][SCALES_COLS], cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1]);
void localMaxSup(cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1], cv::Mat roi[], std::vector<KeyPoint> &kp, int msize = MASK_NMAXSUP);
void contrastThreshold(std::vector<KeyPoint> &kp, cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1], float threshold = CONTRAST_TH);
void edgeThreshold(std::vector<KeyPoint> &kp, cv::Mat dog[SCALES_ROWS][SCALES_COLS - 1], float curv_th = CURV_TH);