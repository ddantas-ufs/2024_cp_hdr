#ifndef AUX_FUNC_H
#define AUX_FUNC_H

#include "core.h"

void printMat( cv::Mat &m, std::string mat_name );

void cleanKeyPointVector( std::vector<KeyPoints> &kp );

void makeGrayscaleCopy( cv::Mat img, cv::Mat &imgOut );

void readImg(char *img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, bool withExtension = false);

void readImg(std::string img_path, cv::Mat &img_in, cv::Mat &img_gray, std::string &img_name, bool withExtension = false);

void readImg( std::string img_path, cv::Mat &img_in );

void readHomographicMatrix( std::string path, cv::Mat &H );

std::string getFileName(std::string file_path, bool withExtension);

int sciToDec(const std::string &str);

std::vector<std::string> split(const std::string& s, char delimiter = '\t');

void readROI(std::string roi_path, std::vector<cv::Point> &verts);

void applyROI( cv::Mat &img, std::string pathROI, bool isHDR );

void selectROI(cv::Mat img, cv::Mat &img_roi, cv::Point v1, cv::Point v2);

void gaussKernel(cv::Mat &kernel, int size = 5, float sigma = 1.0);

void unpackOpenCVOctave(const cv::KeyPoint &kpt, int &octave, int &layer, float &scale);

void exportToOpenCVKeyPointsObject( std::vector<KeyPoints> &kpList, std::vector<cv::KeyPoint> &ocv_kp );

void mapPixelValues( cv::Mat &img, cv::Mat &img_out, int mapInterval = MAPPING_INTERVAL_FLOAT_0_255 );

void getHomographicCorrespondence( cv::Mat imgIn, cv::Mat &imgOut, std::string pathMatrix );

void getHomographicCorrespondence( cv::Mat imgIn, cv::Mat &imgOut, cv::Mat H );

void getHomographicCorrespondence( cv::Point2f p1, cv::Point2f &p2, cv::Mat H );

void getHomographicCorrespondence( float x1, float y1, float &x2, float &y2, cv::Mat H );

#endif
