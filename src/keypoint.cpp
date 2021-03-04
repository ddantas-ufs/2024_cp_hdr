#include "../include/detectors/keypoint.h"

bool outOfBounds(int i, int j, cv::Size size_img)
{
  return ((i < 0) || (j < 0) || (i >= size_img.height) || (j >= size_img.width));
}

void transformCoord(std::vector<KeyPoints> &kp)
{
  for (int i = 0; i < kp.size(); i++)
  {
    if (kp[i].octave > 0)
    {
      kp[i].y = kp[i].y * pow(2, kp[i].octave);
      kp[i].x = kp[i].x * pow(2, kp[i].octave);
    }
  }
}

void plotKeyPoints(cv::Mat img, std::vector<KeyPoints> kp, std::string out_path)
{
  transformCoord(kp);

  for (int i = 0; i < (int)kp.size(); i++)
  {
    cv::circle(img, cv::Point(kp[i].x, kp[i].y), 4, cv::Scalar(0, 255, 0));
  }
  cv::imwrite(out_path + ".kp.png", img);
}

bool compareResponse(KeyPoints a, KeyPoints b)
{
  return (a.resp > b.resp);
}

void saveKeypoints(std::vector<KeyPoints> kp, std::string out_path, int max_kp)
{
  FILE *file;
  std::vector<KeyPoints> kp_aux = kp;
  int num_kp = 0;

  transformCoord(kp_aux);

  std::sort(kp_aux.begin(), kp_aux.end(), compareResponse);

  if (max_kp == 0)
  {
    num_kp = (int)kp.size();
  }
  else
  {
    num_kp = std::min<int>((int)kp_aux.size(), max_kp);
  }

  file = fopen((out_path + ".kp.txt").c_str(), "w+");
  fprintf(file, "%d\n", num_kp);
  for (int i = 0; i < num_kp; i++)
  {
    fprintf(file, "%.4f \t %.4f \t %d \t %d \t %.4f\n", kp_aux[i].y, kp_aux[i].x, kp_aux[i].octave, kp_aux[i].scale, kp_aux[i].resp);
  }
  fclose(file);
}
