#include "../include/detectors/dog.h"

/**
 * dogInitScales : initializes the Scale Scpace to calculate Difference
 * 				   of Gaussian (DoG) images. It consists in DOG_SCL_ROWS
 * 				   octaves in DOG_SCL_COLS sizes. It consists on the first
 * 				   step of Lowe's section 3 paper.
 * 
 * @param img		: Original image which KeyPoints must be extracted;
 * @param scales	: Matrix containing pointers to each image of scale-space;
 * @param mgauss	: Gaussian Filter convolution window size.
**/
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

/**
 * docCalc : Calculates the DoG images using the scale-space. It consists
 * 			 on the second step in Lowe's section 3 paper.
 * @param scales	: Matrix containing the scale-space of the image;
 * @param dog		: Matrix that will store the DoG images.
**/
void dogCalc(cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS], cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1])
{
	for(int i = 0; i < DOG_SCL_ROWS; i++)
		for(int j = 0; j < DOG_SCL_COLS - 1; j++)
		{
			dog[i][j] = cv::Mat::zeros(scales[i][j].size(), CV_32FC1);
			cv::subtract(scales[i][j], scales[i][j + 1], dog[i][j]);
		}
}

/**
 * dogMaxSup : Is the Local Extrema Detection step on Lowe's section 3.1
 * 			   paper. It detects the local Maxima and Minima points in DoG
 * 			   images.
 * @param dog			: Matrix with pointers to DoG images;
 * @param roi			: Array of regions where KeyPoint must be extracted;
 * @param kp			: Vector where KeyPoints are stored;
 * @param maxup_size	: Maximum number of checks when locating Maxima/Minima
 * 					  	  in DoG images.
**/
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
						kp.push_back({float(y), float(x), curr_px, s, l});	
				}
			}
		}
	}
}

void dogInterpolatedMaxSup(cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], cv::Mat roi[], std::vector<KeyPoints> &kp, int maxsup_size, float curv_th)
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
					{
						// MAKE INTERPOLATION
						KeyPoints keypoint = interp_extremum( dog, s, l, y, x, DOG_SCL_COLS - 1, curv_th);
						keypoint.resp = curr_px;
						
						int iMin = std::numeric_limits<int>::min();
						float fMin = std::numeric_limits<float>::min();
						
						// ADD TO KEYPOINT VECTOR
						if( ( keypoint.x == fMin  && keypoint.y == fMin ) && 
							( keypoint.scale == iMin && keypoint.level == iMin ) ) 
						{
							
							kp.push_back({float(y), float(x), curr_px, s, l});
						} 
						else 
						{
							// IF RETURNING SOMETHING, INSERT THE INTERPOLATED KEYPOINT
							kp.push_back( keypoint );
						}
					}
					
				}
			}
		}
	}
}

/**
 * contrastThreshold : applies threshold on minimum contrast value to
 * 					   KeyPoints to be accepted.
 * 
 * @param kp			: KeyPoints stored;
 * @param dog			: Matrix with pointers to DoG images;
 * @param contrast_th	: Threshold value to contrast threshold.
**/
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

/**
 * edgeThreshold : applies threshold on minimum edge value to KeyPoints
 * 				   to be accepted.
 * 
 * @param kp		: KeyPoints stored;
 * @param dog		: Matrix with pointers to DoG images;
 * @param curv_th	: Threshold value to contrast threshold.
**/  
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

/**
 * dogThreshold : performs the Threshold proccess on founded KeyPoints.
 * 				  This is made for reducing the amount of low relevance
 * 				  KeyPoints.
 * 
 * @param kp			: KeyPoints stored;
 * @param dog			: Matrix with pointers to DoG images;
 * @param curv_th		: Threshold value to contrast threshold.
 * @param contrast_th	: Threshold value to contrast threshold.
**/
void dogThreshold(std::vector<KeyPoints> &kp, cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1], float contrast_th, float curv_th)
{
	contrastThreshold(kp, dog, contrast_th);
	edgeThreshold(kp, dog, curv_th);
}

/**
 * dogKp : method that coordenates the DoG detector execution. As described
 * 		   in Lowe, 2004 paper.
 * @param img 			: grayscale image to extract feature points (KeyPoints);
 * @param roi 			: 4 masks used to delimitate the region of the scene that 
 * 			  			  KeyPoints must be extracted. Used in Pribyl et. al.;
 * @param kp			: Vector of KeyPoints founded;
 * @param mgauss		: Gaussian filter size. Defaults to GAUSS_SIZE in
 * 						  core.h header;
 * @param maxup_size	: Maximum number of checks when locating Maxima/Minima
 * 						  in DoG images. Defaults to MAXSUP_SIZE in core.h;
 * @param contrast_th	: Contrast threshold value to threshold proccess. 
 * 						  Defaults to CONTRAST_TH value in core.h;
 * @param curv_th		: Edge threshold value to threshold process. Defaults
 * 						  to CURV_TH value in core.h.
**/
void dogKp(cv::Mat img, cv::Mat roi[], std::vector<KeyPoints> &kp, int mgauss, int maxsup_size, float contrast_th, float curv_th)
{
	cv::Mat scales[DOG_SCL_ROWS][DOG_SCL_COLS];
	cv::Mat dog[DOG_SCL_ROWS][DOG_SCL_COLS - 1];

	dogInitScales(img, scales, mgauss);
	dogCalc(scales, dog);
	//dogMaxSup(dog, roi, kp, maxsup_size);
	dogInterpolatedMaxSup(dog, roi, kp, maxsup_size, curv_th);
	dogThreshold(kp, dog, contrast_th, curv_th);
}

/**
 * Interpolates a scale-space extremum's location and scale to subpixel
 * accuracy to form an image feature.  Rejects features with low contrast.
 * Based on Section 4 of Lowe's paper.  
 * @param dog_pyr DoG scale space pyramid
 * @param octv feature's octave of scale space
 * @param intvl feature's within-octave interval
 * @param r feature's image row
 * @param c feature's image column
 * @param intvls total intervals per octave
 * @param contr_thr threshold on feature contrast
 * @return	Returns the feature resulting from interpolation of the given
 *			parameters or NULL if the given location could not be interpolated or
 *			if contrast at the interpolated loation was too low.  If a feature is
 *			returned, its scale, orientation, and descriptor are yet to be determined.
**/
static KeyPoints interp_extremum( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1],  // dog
								  int octv, // s
								  int intvl,  // l
								  int r, // y
								  int c,  // x
								  int intvls, // DOG_SCL_COLS - 1
								  double contr_thr ) // curv_th
{
	KeyPoints feat;
	double xi, xr, xc, contr;
	int i = 0;
	
	// MINIMUM TYPE VALUES TO KEYPOINT
	feat.x = std::numeric_limits<float>::min();
	feat.y = std::numeric_limits<float>::min();
	feat.scale = std::numeric_limits<int>::min();
	feat.level = std::numeric_limits<int>::min();

	while( i < DOG_MAX_INTERP_STEPS )
	{
		interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
		if( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 )
			break;

		c += cvRound( xc );
		r += cvRound( xr );
		intvl += cvRound( xi );

		if( intvl < 1  ||
			intvl > intvls  ||
			c < SIFT_IMG_BORDER  ||
			r < SIFT_IMG_BORDER  ||
			c >= dog_pyr[octv][0].rows - SIFT_IMG_BORDER  ||
			r >= dog_pyr[octv][0].cols - SIFT_IMG_BORDER )
		{
			return feat;
		}

		i++;
	}

	/* ensure convergence of interpolation */
	if( i >= DOG_MAX_INTERP_STEPS )
		return feat;

	contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
	if( ABS( contr ) < contr_thr / intvls )
		return feat;
	
	// SAVING INTERPOLATED VALUES
	feat.x = ( c + xc ) * pow( 2.0, octv );
	feat.y = ( r + xr ) * pow( 2.0, octv );
	feat.scale = octv;
	feat.level = intvl;
	//feat.resp = float(xi);

	return feat;
}

/**
 * Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
 * paper.
 * @param dog_pyr difference of Gaussians scale space pyramid
 * @param octv octave of scale space
 * @param intvl interval being interpolated
 * @param r row being interpolated
 * @param c column being interpolated
 * @param xi output as interpolated subpixel increment to interval
 * @param xr output as interpolated subpixel increment to row
 * @param xc output as interpolated subpixel increment to col
**/

static void interp_step( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1],
						 int octv, int intvl, int r, int c,
						 double* xi, double* xr, double* xc )
{
	CvMat* dD, * H, * H_inv, X;
	double x[3] = { 0 };

	dD = deriv_3D( dog_pyr, octv, intvl, r, c );
	H = hessian_3D( dog_pyr, octv, intvl, r, c );
	H_inv = cvCreateMat( 3, 3, CV_64FC1 );
	cvInvert( H, H_inv, CV_SVD );
	cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
	cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );

	cvReleaseMat( &dD );
	cvReleaseMat( &H );
	cvReleaseMat( &H_inv );

	*xi = x[2];
	*xr = x[1];
	*xc = x[0];
}

/**
 * Computes the partial derivatives in x, y, and scale of a pixel in the DoG
 * scale space pyramid.
 * @param dog_pyr DoG scale space pyramid
 * @param octv pixel's octave in dog_pyr
 * @param intvl pixel's interval in octv
 * @param r pixel's image row
 * @param c pixel's image col
 * @return Returns the vector of partial derivatives for pixel I
 * { dI/dx, dI/dy, dI/ds }^T as a CvMat*
**/
static CvMat* deriv_3D( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1], int octv, int intvl, int r, int c )
{
  CvMat* dI;
  double dx, dy, ds;

  dx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) -
		 pixval32f( dog_pyr[octv][intvl], r, c-1 ) ) / 2.0;
  dy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) -
		 pixval32f( dog_pyr[octv][intvl], r-1, c ) ) / 2.0;
  ds = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) -
		 pixval32f( dog_pyr[octv][intvl-1], r, c ) ) / 2.0;
  
  dI = cvCreateMat( 3, 1, CV_64FC1 );
  cvmSet( dI, 0, 0, dx );
  cvmSet( dI, 1, 0, dy );
  cvmSet( dI, 2, 0, ds );

  return dI;
}

/**
 * Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.
 * @param dog_pyr DoG scale space pyramid
 * @param octv pixel's octave in dog_pyr
 * @param intvl pixel's interval in octv
 * @param r pixel's image row
 * @param c pixel's image col
 * @return Returns the Hessian matrix (below) for pixel I as a CvMat*
 * / Ixx  Ixy  Ixs \
 * | Ixy  Iyy  Iys |
 * \ Ixs  Iys  Iss /
**/
static CvMat* hessian_3D( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1],
						  int octv, int intvl, int r, int c )
{
	CvMat* H;
	double v, dxx, dyy, dss, dxy, dxs, dys;

	v = pixval32f( dog_pyr[octv][intvl], r, c );
	dxx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) + 
			pixval32f( dog_pyr[octv][intvl], r, c-1 ) - 2 * v );
	dyy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) +
			pixval32f( dog_pyr[octv][intvl], r-1, c ) - 2 * v );
	dss = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) +
			pixval32f( dog_pyr[octv][intvl-1], r, c ) - 2 * v );
	dxy = ( pixval32f( dog_pyr[octv][intvl], r+1, c+1 ) -
			pixval32f( dog_pyr[octv][intvl], r+1, c-1 ) -
			pixval32f( dog_pyr[octv][intvl], r-1, c+1 ) +
			pixval32f( dog_pyr[octv][intvl], r-1, c-1 ) ) / 4.0;
	dxs = ( pixval32f( dog_pyr[octv][intvl+1], r, c+1 ) -
			pixval32f( dog_pyr[octv][intvl+1], r, c-1 ) -
			pixval32f( dog_pyr[octv][intvl-1], r, c+1 ) +
			pixval32f( dog_pyr[octv][intvl-1], r, c-1 ) ) / 4.0;
	dys = ( pixval32f( dog_pyr[octv][intvl+1], r+1, c ) -
			pixval32f( dog_pyr[octv][intvl+1], r-1, c ) -
			pixval32f( dog_pyr[octv][intvl-1], r+1, c ) +
			pixval32f( dog_pyr[octv][intvl-1], r-1, c ) ) / 4.0;

	H = cvCreateMat( 3, 3, CV_64FC1 );
	cvmSet( H, 0, 0, dxx );
	cvmSet( H, 0, 1, dxy );
	cvmSet( H, 0, 2, dxs );
	cvmSet( H, 1, 0, dxy );
	cvmSet( H, 1, 1, dyy );
	cvmSet( H, 1, 2, dys );
	cvmSet( H, 2, 0, dxs );
	cvmSet( H, 2, 1, dys );
	cvmSet( H, 2, 2, dss );

	return H;
}

/**
 * Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's
 * paper.
 * @param dog_pyr difference of Gaussians scale space pyramid
 * @param octv octave of scale space
 * @param intvl within-octave interval
 * @param r pixel row
 * @param c pixel column
 * @param xi interpolated subpixel increment to interval
 * @param xr interpolated subpixel increment to row
 * @param xc interpolated subpixel increment to col
 * @param Returns interpolated contrast.
**/
static double interp_contr( cv::Mat dog_pyr[DOG_SCL_ROWS][DOG_SCL_COLS - 1],
							int octv, int intvl, int r, int c,
							double xi, double xr, double xc )
{
	CvMat* dD, X, T;
	double t[1], x[3] = { xc, xr, xi };

	cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
	cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
	dD = deriv_3D( dog_pyr, octv, intvl, r, c );
	cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
	cvReleaseMat( &dD );

	return pixval32f( dog_pyr[octv][intvl], r, c ) + t[0] * 0.5;
}
