#include "../include/detectors/keypoint.h"

std::string keypointToString( KeyPoints &kp )
{
  return "x: " +std::to_string(kp.x) 
         + ", y:" +std::to_string(kp.y) +"\n"
         + "scale: " +std::to_string(kp.scale) 
         +", octave:" +std::to_string(kp.octave) +"\n"
         + "response: " +std::to_string(kp.resp) 
         +", direction:" +std::to_string(kp.direction);
}

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

void plotKeyPoints(cv::Mat img, std::vector<KeyPoints> kp, std::string out_path, int max_kp)
{
  int num_kp = 0;

  transformCoord(kp);

  if (max_kp == 0)
  {
    num_kp = (int)kp.size();
  }
  else
  {
    num_kp = std::min<int>((int)kp.size(), max_kp);
  }

  for (int i = 0; i < num_kp; i++)
  {
    cv::circle(img, cv::Point(kp[i].x, kp[i].y), 4, cv::Scalar(0, 255, 0));
  }
  cv::imwrite(out_path + ".kp.png", img);
}

bool compareResponse(KeyPoints a, KeyPoints b)
{
  return (a.resp > b.resp);
}

void saveKeypoints(std::vector<KeyPoints> &kp, std::string out_path, int max_kp,
                   bool transformCoordinate, bool descriptorToBundler)
{
  FILE *file;
  std::vector<KeyPoints> kp_aux = kp;
  int num_kp = 0;

  //if( transformCoordinate ) transformCoord(kp_aux);

  std::sort(kp_aux.begin(), kp_aux.end(), compareResponse);

  if (max_kp == 0)
  {
    num_kp = (int)kp_aux.size();
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

    int descSize = kp_aux[i].descriptor.size();
    if( descSize == 0 )
    {
        fprintf(file, "0\n" ); // print 0 as descriptor
    }
    else
    {
      for( int j=0; j<descSize; j++ )
      {
        int desc = (int) kp_aux[i].descriptor[j];
        fprintf(file, "%d ", desc );

        if( descriptorToBundler && (i % 10) == 0 && (i != 0) && j != descSize-1 )
          fprintf(file, "\n" );
      }
      fprintf(file, "\n" );
    }
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
      key.direction = 0.0;

      kp.push_back( key );
      i = 0;
      delete buff;
    }
    arch.close();
  }

  return kp;
}

/**
 * Loads keypoints saved using Lowe implementation,
 * avaliable at https://www.cs.ubc.ca/~lowe/keypoints/
 * 
 * @param kp KeyPoints vector to store the loaded keypoints
 * @param arqPath path to archive containing the saved keypoints
 * 
 * File format:
 * qtdKeypoints 128
 * X Y Scale Orientation (-pi, pi)
 * 128-dimension descriptor
**/
std::vector<KeyPoints> loadLoweKeypoints( std::string arqPath )
{
  std::cout << " 1 " << std::endl;
  std::fstream arch; 
  std::string line, strY, strX, strOctave, strScale, strOrientation;
  std::vector<KeyPoints> kp;

  arch.open( arqPath, std::ios::in ); 
  std::cout << " 2 " << std::endl;

  if( arch.is_open() )
  {
    std::cout << " 2 " << std::endl;
    int qtdKeypoints;
    std::string qtdKeypointsStr;
    int sz = line.size();
    char* buff;
    char ln[sz+1];
    getline( arch, line );

    // READING QUANTITY OF KEYPOINTS TO BE LOADED
    strcpy( ln, line.c_str() );
    buff = strtok( ln, " " );
    std::cout << " 4 " << std::endl;

    std::string strBuff = buff;
    strBuff.erase(std::remove(strBuff.begin(), strBuff.end(), ' '), strBuff.end());
    qtdKeypointsStr = strBuff;
    qtdKeypoints = std::stoi(qtdKeypointsStr);
    //delete buff;

    for(int i=0; i < qtdKeypoints; i++)
    {
      std::cout << " 5 " << std::endl;
      getline( arch, line );

      // READING KEYPOINT POSITION, SCALE AND ORIENTATION
      strcpy( ln, line.c_str() );
      buff = strtok( ln, " " );
      strBuff = buff;
      std::cout << " 6 " << std::endl;

      for( int keypoints = 0; keypoints < 4; keypoints++ )
      {
        std::cout << " 7 " << std::endl;
        strBuff = buff;
        strBuff.erase(std::remove(strBuff.begin(), strBuff.end(), ' '), strBuff.end());
        if( keypoints == 0 )
        {
          strY = strBuff;
        }
        if( keypoints == 1 )
        {
          strX = strBuff;
        }
        if( keypoints == 2 )
        {
          strScale = strBuff;
        }
        if( keypoints == 3 )
        {
          strOrientation = strBuff;
        }
        std::cout << strBuff << "\n";
        buff = strtok( NULL, " ");
        std::cout << " 8 " << std::endl;
      }

      // IGNORING 7 LINES CONTAINING DESCRIPTOR INFORMATION
      for(int desc = 0; desc < 7; desc++) getline( arch, line );
      std::cout << " 9 " << std::endl;

      // SAVE KEYPOINT
      KeyPoints key;
      key.y = std::stof(strY);
      key.x = std::stof(strX);
      key.scale = std::stof(strScale);
      key.direction = (std::stof(strOrientation) + M_PI) * (180.0 / M_PI); // converting to degrees
      key.resp = 0.0;
      key.octave = 0;

      kp.push_back( key );
      //delete buff;
      std::cout << " 10 " << std::endl;
    }
    arch.close();
    std::cout << " 11 " << std::endl;
  }

  return kp;
}

void printKeypoint( KeyPoints &kp )
{
  std::cout << "X: " << kp.x << ", Y: " << kp.y << std::endl;
  std::cout << "Scale: " << kp.scale << ", Octave: " << kp.octave << std::endl;
  std::cout << "Response: " << kp.resp << ", Direction: " << kp.direction << std::endl;
  
  if( kp.descriptor.size() == 0 )
  {
    std::cout << "Descriptor not calculated" << std::endl;
  }
  else
  {
    std::cout << "Descriptor size: " << kp.descriptor.size() << std::endl;
    for( int i=0; i<kp.descriptor.size(); i++ )
    {
      if( i > 0 && ( i % 10 ) == 0 )
        std::cout << std::endl;

      std::cout << kp.descriptor[i] << ", ";
    }
    std::cout << std::endl;
  }
}