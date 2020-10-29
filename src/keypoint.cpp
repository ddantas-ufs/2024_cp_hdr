#include "keypoint.h"

bool outOfBounds(int i, int j, cv::Size size_img)
{
	return ((i < 0) || (j < 0) || (i >= size_img.height) || (j >= size_img.width));
}

void transformCoord(std::vector<KeyPoints> &kp)
{
	for(int i = 0; i < kp.size(); i++)
	{
		if(kp[i].scale > 0)
		{
			kp[i].y = kp[i].y*pow(2, kp[i].scale);
			kp[i].x = kp[i].x*pow(2, kp[i].scale);
		}
	}
}

void plotKeyPoints(cv::Mat &img, std::vector<KeyPoints> kp, std::string out_path)
{
	transformCoord(kp);

	for(int i = 0; i < (int)kp.size(); i++)
		cv::circle(img, cv::Point(kp[i].x, kp[i].y), 4, cv::Scalar(0, 255, 0));

	cv::imwrite(out_path + ".kp.png", img);
}

bool compareResponse(KeyPoints a, KeyPoints b)
{
    return (a.resp > b.resp);
}

void saveKeypoints(std::vector<KeyPoints> &kp, cv::Mat roi[], std::string out_path, int max_kp)
{
	std::vector<KeyPoints> kp_roi1, kp_roi2, kp_roi3;
	std::vector<KeyPoints> kp_aux;
	FILE *file;
	
	transformCoord(kp);

    kp_aux = kp;
	std::sort(kp_aux.begin(), kp_aux.end(), compareResponse);
    
    for(int i = 0; (i < max_kp) && (i < (int)kp_aux.size()); i++)
	{
		int y = kp_aux[i].y;
		int x = kp_aux[i].x;

		if(roi[1].at<uchar>(y, x) != 0)
			kp_roi1.push_back(kp_aux[i]);
	 	else if(roi[2].at<uchar>(y, x) != 0)
			kp_roi2.push_back(kp_aux[i]);
	 	else if(roi[3].at<uchar>(y, x) != 0)
			kp_roi3.push_back(kp_aux[i]);
    }
	kp.clear();

	for(int i = 0; i < (int)kp_roi1.size(); i++)
		kp.push_back(kp_roi1[i]);
	
	for(int i = 0; i < (int)kp_roi2.size(); i++)
		kp.push_back(kp_roi2[i]);
	
	for(int i = 0; i < (int)kp_roi3.size(); i++)
		kp.push_back(kp_roi3[i]);
	
	double total_kp = (int)kp_roi1.size() + (int)kp_roi2.size() + (int)kp_roi3.size();
	double rate_min = std::min((int)kp_roi1.size()/total_kp,
				      std::min((int)kp_roi2.size()/total_kp, (int)kp_roi3.size()/total_kp));
	double rate_max = std::max((int)kp_roi1.size()/total_kp,
				      std::max((int)kp_roi2.size()/total_kp, (int)kp_roi3.size()/total_kp));
	double dist_rate = 1 - (rate_max - rate_min);

	file = fopen((out_path + ".distrate.txt").c_str(), "w+");
	fprintf(file, "%.4f\n", dist_rate);
	fclose(file);
	
	file = fopen((out_path + ".kp.roi1.txt").c_str(), "w+");
	fprintf(file, "%d\n", (int)kp_roi1.size());
	for(int i = 0; i < (int)kp_roi1.size(); i++)
		fprintf(file, "%d %d %.4f\n", kp_roi1[i].y, kp_roi1[i].x, kp_roi1[i].resp);
	fclose(file);

	file = fopen((out_path + ".kp.roi2.txt").c_str(), "w+");
	fprintf(file, "%d\n", (int)kp_roi2.size());
	for(int i = 0; i < (int)kp_roi2.size(); i++)
		fprintf(file, "%d %d %.4f\n", kp_roi2[i].y, kp_roi2[i].x, kp_roi2[i].resp);
	fclose(file);

	file = fopen((out_path + ".kp.roi3.txt").c_str(), "w+");
	fprintf(file, "%d\n", (int)kp_roi3.size());
	for(int i = 0; i < (int)kp_roi3.size(); i++)
		fprintf(file, "%d %d %.4f\n", kp_roi3[i].y, kp_roi3[i].x, kp_roi3[i].resp);	
	fclose(file);
}