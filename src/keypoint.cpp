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

/**
 * Loads previously saved Keypoints
 * 
 * @param kp KeyPoints vector to store the loaded keypoints
 * @param arqPath path to archive containing the saved keypoints
**/
std::vector<KeyPoints> loadKeypoints( std::string arqPath )
{
  std::fstream arch;
  std::string line, strY, strX, strOctave, strScale, strResp;
  std::vector<KeyPoints> kp;

  arch.open( arqPath, std::ios::in ); 

  if( arch.is_open() )
  {
    while( getline( arch, line ) )
    {
      int i = 0;
      int sz = line.size();
      char* buff;
      char ln[sz+1];

      // strtok WORKS ONLY WITH CHAR ARRAY
      strcpy( ln, line.c_str() );

      buff = strtok( ln, "\t" );
      while( buff != NULL )
      {
        std::string strBuff = buff;
        strBuff.erase(std::remove(strBuff.begin(), strBuff.end(), ' '), strBuff.end());
        if( i == 0 )
        {
          strY = strBuff;
        }
        if( i == 1 )
        {
          strX = strBuff;
        }
        if( i == 2 )
        {
          strOctave = strBuff;
        }
        if( i == 3 )
        {
          strScale = strBuff;
        }
        if( i == 4 )
        {
          strResp = strBuff;
        }
        std::cout << strBuff << "\n";
        buff = strtok( NULL, "\t");
        i = i + 1;
      }

      KeyPoints key;
      key.y = std::stof(strY);
      key.x = std::stof(strX);
      key.octave = std::stoi(strOctave);
      key.scale = std::stoi(strScale);
      key.resp = std::stof(strResp);

      kp.push_back( key );
      i = 0;
      delete buff;
    }
    arch.close();
  }

  return kp;
}