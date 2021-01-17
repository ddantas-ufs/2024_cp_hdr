#ifndef DOG_H
#define DOG_H

#include "core.h"

void dogKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoints> &kp,
           int mgauss = GAUSS_SIZE, int maxsup_size = MAXSUP_SIZE,
           float contrast_th = CONTRAST_TH, float curv_th = CURV_TH);


// METHODS TO SUBPIXEL INTERPOLATION
void dogInterpolatedMaxSup(	cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1],
							cv::Mat roi[], 
							std::vector<KeyPoints> &kp, 
							int maxsup_size, 
							float curv_th);

static KeyPoints interp_extremum( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1], int octv,
								  int intvl, int r, int c, int intvls,
								  double contr_thr );

static void interp_step( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1],
						 int octv, int intvl, int r, int c,
						 double* xi, double* xr, double* xc );

static CvMat* deriv_3D( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1],
						int octv, int intvl, int r, int c );

static CvMat* hessian_3D( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1],
						  int octv, int intvl, int r, int c );

static double interp_contr( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1], 
							int octv, int intvl, int r, int c,
							double xi, double xr, double xc );
							
/**
   A function to get a pixel value from a 32-bit floating-point image.
   
   @param img an image
   @param r row
   @param c column
   @return Returns the value of the pixel at (\a r, \a c) in \a img
*/
static inline float pixval32f( cv::Mat img, int r, int c )
{
	//return ( (float*)(img->imageData + img->widthStep*r) )[c];
	return img.at<float>(r,c);
}

#endif
