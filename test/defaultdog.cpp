#include "../include/cphdr.h"

int main(int argv, char** args)
{
	for( int i = 0; i < argv; i++ ) 
	{
		printf( "arg %i: %s\n", i, args[i] );
	}
	
	cv::Mat img_in;
	cv::Mat img_gray;
	cv::Mat roi[4];
	std::vector<KeyPoints> kp;
	std::string img_name;
	std::string out_path;

	readImg(args[1], img_in, img_gray, img_name);
	
	if( argv < 4 ) 
	{
		printf( "Sem ROI\n" );
		readRoi(NULL, roi, img_gray.size());
		out_path = std::string(args[2]) + img_name + ".dog";
	} else 
	{
		printf( "Com ROI\n" );
		readRoi(args[2], roi, img_gray.size());
		out_path = std::string(args[3]) + img_name + ".dog";
	}

	dogKp(img_gray, roi, kp);
	
	saveKeypoints(kp, roi, out_path);
	plotKeyPoints(img_in, kp, out_path);

	return 0;
}
