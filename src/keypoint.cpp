#include "../include/detectors/keypoint.h"

/**
 * @brief prints a KeyPoints object
 * 
 * @param kp : keypoint to be printed
 * @return std::string returned keypoint image
 */
std::string keypointToString( KeyPoints &kp )
{
  return "x: " +std::to_string(kp.x) 
         + ", y:" +std::to_string(kp.y) +"\n"
         + "scale: " +std::to_string(kp.scale) 
         +", octave:" +std::to_string(kp.octave) +"\n"
         + "response: " +std::to_string(kp.resp) 
         +", direction:" +std::to_string(kp.direction);
}

/**
 * @brief Test if coordinate is out of image
 * 
 * @param i y coordinate
 * @param j x coordinate
 * @param size_img image to test
 * @return true if coordinates are inside image
 * @return false if coordinates are outside image
 */
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

/**
 * @brief Calculates the distance between points A and B, using Pitagoras theorem
 * 
 * @param p1 point A
 * @param p2 point B
 * @return float distance between points
 */
float distanceBetwenTwoKeyPoints( KeyPoints p1, KeyPoints p2 )
{
  float distance = std::sqrt( std::pow( std::abs( p1.x-p2.x ), 2 ) + std::pow( p1.y-p2.y, 2 ) );
  //std::cout << "Calculated distance between P1(" << p1.x << "," << p1.y << ") and P2(" << p2.x << "," << p2.y << "): " << distance << std::endl;
  return distance;
}

/**
 * @brief get slice of vector
 * 
 * @param vtr vector to be 
 * @param beg inicial interval
 * @param end final interval
 * @return std::vector<KeyPoints> 
 */
std::vector<KeyPoints> vectorSlice(std::vector<KeyPoints> const &vtr, int beg, int end)
{
  if( vtr.size() <= end )
    return vtr;

  auto first = vtr.cbegin() + beg;
  auto last = vtr.cbegin() + end;

  std::vector<KeyPoints> vec(first, last);
  return vec;
}

/**
 * Receive the histogram array and descriptor array, normalize the values and store
 * the normalized values into descriptor array.
 * 
 * @param hist histogram array with calculated bins
 * @param descriptor descriptor array where normalized values will be stored
 * @param qtdSW the amount of subWindows calculated (default = 4)
 * @param binsPerSW the amount of bins calculated per each subWindow (default = 8)
**/
void normalizeDescriptor( cv::Mat hist, cv::Mat &descriptor  )
{
  int k = 0;

  int qtdSW = 4;
  int binsPerSW = 8;

  hist.copyTo( descriptor );
  
  // Preparing to normalize histogram
  float nrm2 = 0.0f;
  int len = qtdSW*qtdSW*binsPerSW;
  for( k = 0; k < len; k++ )
    nrm2 += descriptor.at<float>(k)*descriptor.at<float>(k);
//  nrm2 += descriptor[k]*descriptor[k];

  float thr = std::sqrt(nrm2)*SIFT_DESC_MAG_THR;
  nrm2 = 0.0f;

  for( int i = 0; i < k; i++ )
  {
//  float val = std::min(descriptor[i], thr);
//  descriptor[i] = val;
    float val = std::min(descriptor.at<float>(i), thr);
    descriptor.at<float>(i) = val;
    nrm2 += val*val;
  }

  nrm2 = SIFT_DESC_INT_FTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

  // Copying normalized data to descriptor
  for( k = 0; k < len; k++ )
    descriptor.at<float>(k) = uchar( descriptor.at<float>(k)*nrm2 );
}

/**
 * @brief plot bubbles in images using KeyPoints vector
 * 
 * @param img image where bubbles will be ploted
 * @param kp vector of KeyPoints
 * @param out_path output images
 * @param max_kp max amount of KeyPoints
 */
void plotKeyPoints(cv::Mat img, std::vector<KeyPoints> kp, std::string out_path, int max_kp)
{
  int num_kp = 0;
  cv::Mat out;
  img.copyTo( out );

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
    cv::circle(out, cv::Point(kp[i].x, kp[i].y), 6, cv::Scalar(0, 255, 0), 1);
    cv::circle(out, cv::Point(kp[i].x, kp[i].y), 5, cv::Scalar(0, 255, 0), 1);
    cv::circle(out, cv::Point(kp[i].x, kp[i].y), 4, cv::Scalar(0, 0, 255), 1);
    cv::circle(out, cv::Point(kp[i].x, kp[i].y), 3, cv::Scalar(0, 0, 255), 1);
    cv::circle(out, cv::Point(kp[i].x, kp[i].y), 2, cv::Scalar(0, 0, 255), 1);
    cv::circle(out, cv::Point(kp[i].x, kp[i].y), 1, cv::Scalar(0, 0, 255), 1);
  }
  std::cout << "out path: " << out_path << std::endl;
  cv::imwrite(out_path, out);
}

bool compareResponse(KeyPoints a, KeyPoints b)
{
  return (a.resp > b.resp);
}

void sortKeypoints( std::vector<KeyPoints> &vec )
{
  //std::vector<KeyPoints> vec_aux = vec;
  std::sort(vec.begin(), vec.end(), compareResponse);
}

void saveKeypoints(std::vector<KeyPoints> &kp, std::string out_path, int max_kp,
                   bool transformCoordinate, bool descriptorToBundler)
{
  FILE *file;
//  std::vector<KeyPoints> kp_aux = kp;
  int num_kp = 0;

  sortKeypoints( kp );
  //if( transformCoordinate ) transformCoord(kp_aux);

//  std::sort(kp_aux.begin(), kp_aux.end(), compareResponse);

  if (max_kp <= 0)
  {
    num_kp = (int) kp.size();
  }
  else
  {
    num_kp = std::min<int>((int) kp.size(), max_kp);
  }

  file = fopen((out_path + ".kp.txt").c_str(), "w+");
  fprintf(file, "%d\n", num_kp);
  for (int i = 0; i < num_kp; i++)
  {
    fprintf(file, "%.4f \t %.4f \t %d \t %d \t %.4f\n", kp[i].y, kp[i].x, kp[i].octave, kp[i].scale, kp[i].resp);

    int descSize = kp[i].descriptor.size();
    if( descSize == 0 )
    {
        fprintf(file, "0\n" ); // print 0 as descriptor
    }
    else
    {
      for( int j=0; j<descSize; j++ )
      {
        int desc = (int) kp[i].descriptor[j];
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
  std::string line, strY = "", strX = "", strOctave = "", strScale = "", strResp = "";
  std::vector<KeyPoints> kp;
  std::vector<int> desc;

  arch.open( arqPath, std::ios::in ); 

  if( arch.is_open() )
  {
    getline( arch, line );
    std::cout << "> # Keypoints being loaded: " << line << std::endl;

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
        std::cout << strBuff << std::endl;
        buff = strtok( NULL, "\t");
        i = i + 1;
      }

      // READING DESCRIPTION
      std::cout << "Reading description" << std::endl;
      getline( arch, line );
      /*
      std::cout << "Line size: " << sz << ", line: " << line << std::endl;
      char lndesc[128];
      strcpy( lndesc, line.c_str() );
      std::cout << "Line char: " << lndesc << std::endl;

      buff = strtok( lndesc, " " );
      while( buff != NULL )
      {
        std::string strBuff = buff;
        strBuff.erase(std::remove(strBuff.begin(), strBuff.end(), ' '), strBuff.end());

        desc.push_back( std::stoi( strBuff ) );
        
        std::cout << strBuff << std::endl;
        buff = strtok( NULL, " ");
        i = i + 1;
      }
      */

      KeyPoints key;
      key.y      = std::stof(strY);
      key.x      = std::stof(strX);
      key.octave = std::stoi(strOctave);
      key.scale  = std::stoi(strScale);
      key.resp   = std::stof(strResp);
      key.direction = 0.0f;

      kp.push_back( key );

      //strY = "", strX = "", strOctave = "", strScale = "", strResp = "";
      i = 0;
      delete buff;
    }
    arch.close();
  }
  else
  {
    std::cout << "Keypoints file not opened! Path:" << arqPath << std::endl;
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
  std::fstream arch;
  std::string line, strY, strX, strOctave, strScale, strOrientation;
  std::vector<KeyPoints> kp;

  arch.open( arqPath, std::ios::in ); 

  if( arch.is_open() )
  {
    int qtdKeypoints;
    std::string qtdKeypointsStr;
    int sz = line.size();
    char* buff;
    char ln[sz+1];
    getline( arch, line );

    // READING QUANTITY OF KEYPOINTS TO BE LOADED
    strcpy( ln, line.c_str() );
    buff = strtok( ln, " " );

    std::string strBuff = buff;
    strBuff.erase(std::remove(strBuff.begin(), strBuff.end(), ' '), strBuff.end());
    qtdKeypointsStr = strBuff;
    qtdKeypoints = std::stoi(qtdKeypointsStr);
    //delete buff;

    for(int i=0; i < qtdKeypoints; i++)
    {
      getline( arch, line );

      // READING KEYPOINT POSITION, SCALE AND ORIENTATION
      strcpy( ln, line.c_str() );
      buff = strtok( ln, " " );
      strBuff = buff;

      for( int keypoints = 0; keypoints < 4; keypoints++ )
      {
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
        //std::cout << strBuff << "\n";
        buff = strtok( NULL, " ");
      }

      // IGNORING 7 LINES CONTAINING DESCRIPTOR INFORMATION
      for(int desc = 0; desc < 7; desc++) getline( arch, line );

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
    }
    arch.close();
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

void unpackOpenCVOctave(const cv::KeyPoint &kpt, int &octave, int &layer, float &scale)
{
  octave = kpt.octave & 255;
  layer = (kpt.octave >> 8) & 255;
  octave = octave < 128 ? octave : (-128 | octave);
  scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

/**
 * Function that reads OpenCV Keypoints and description objects
 * and transforms it in CP_HDR KeyPoints structure.
 * 
 * @param ocv_kp: OpenCV keypoints vector;
 * @param descriptor: OpenCV descriptor object;
 * @param kpList: CP_HDR vector in with the Keypoints and description will be returned;
 * @param withDescription: boolean that indicates if description needs to be transformed.
**/
void loadOpenCVKeyPoints( std::vector<cv::KeyPoint> &ocv_kp, cv::Mat &descriptor,
                          std::vector<KeyPoints> &kpList, bool withDescription )
{
  std::cout << "Importing " << ocv_kp.size() << " Keypoints ";
  if( withDescription ) std::cout << "with description" << std::endl;
  else std::cout << "without description" << std::endl;

  for(int i=0; i<ocv_kp.size(); i++)
  {
    KeyPoints nkp;
    int uOctave, uLayer;
    float uScale;

    unpackOpenCVOctave( ocv_kp[i], uOctave, uLayer, uScale );

    nkp.x = float(ocv_kp[i].pt.x);
    nkp.y = float(ocv_kp[i].pt.y);
    nkp.resp = float(ocv_kp[i].response);
    nkp.direction = float(ocv_kp[i].angle);
    nkp.octave = int(uOctave);
    nkp.scale = int(uLayer);

    //std::cout << "Coord. OpenCV: X:" << ocv_kp[i].pt.x << ", Y:" << ocv_kp[i].pt.y << std::endl;
    //std::cout << "Coord. CP_HDR: X:" << nkp.x          << ", Y:" << nkp.y          << std::endl;

    //std::cout << "X, Y: " << nkp.x << ", " << nkp.y << std::endl;
    //std::cout << "Resp: " << uLayer << std::endl;
    //std::cout << "Octv: " << nkp.octave << std::endl;
    //std::cout << "Angl: " << nkp.direction << std::endl;
    //std::cout << "Scal: " << nkp.scale << std::endl;

    if( withDescription )
    {
//      cv::Mat normDescriptor;
//      normalizeDescriptor( descriptor, normDescriptor );
      for( int j=0; j < descriptor.cols; j++ )
      {
        float dValue = descriptor.at<float>(j, i);
        unsigned int iValue = static_cast< unsigned int >( static_cast<int>( dValue ) );
        if( iValue > 255 ) iValue = 255;
        if( iValue < 0 ) iValue = 0;

        nkp.descriptor.push_back( iValue );
        
        //std::cout << "OpenCV Desc: " << descriptor.at<float>(j, i) << ". Type: " << returnOpenCVArrayType( descriptor.type() ) << std::endl;
        //std::cout << "CP_HDR Desc: " << dValue << ", Convertido: " << iValue << std::endl;
      }
    }
    kpList.push_back(nkp);
  }

}

void loadOpenCVKeyPoints( std::vector<cv::KeyPoint> &ocv_kp, cv::Mat &descriptor,
                          std::vector<KeyPoints> &kpList )
{
  loadOpenCVKeyPoints( ocv_kp, descriptor, kpList, true );
}

void loadOpenCVKeyPoints( std::vector<cv::KeyPoint> &ocv_kp, std::vector<KeyPoints> &kpList )
{
  cv::Mat descriptor;
  loadOpenCVKeyPoints( ocv_kp, descriptor, kpList, false );
}

/*
  Pack info into octave variable
  Octave integer variable stores 2 informations 
  that are stored in the first 2 binary octets.
  + xxxxxxxx xxxxxxxx llllllll oooooooo -
  Where:
    x = Unused octets
    l = Layer information octet
    o = Octave information octet (less significant octet)
*/
void packOpenCVOctave(const cv::KeyPoint &kpt, int &octave, int &layer )
{
  int oct = octave;
  int aux = layer;
  aux = aux << 8; // pass first octet to second octet
  aux = aux | ( 15 && oct ); // Octave is the first octet

  octave = oct;
}

void exportToOpenCVKeyPointsObject( std::vector<KeyPoints> &kpList, std::vector<cv::KeyPoint> &ocv_kp )
{
  std::cout << "Exporting " << kpList.size() << " Keypoints " << std::endl;

  for(int i=0; i<kpList.size(); i++)
  {
    //std::cout << "kp " << i << std::endl;
    cv::KeyPoint nkp;
    int oct = kpList[i].octave;
    int lay = kpList[i].scale;

    packOpenCVOctave( ocv_kp[i], oct, lay );

    nkp.pt.x = kpList[i].x;
    nkp.pt.y = kpList[i].y;
    nkp.response = kpList[i].resp;
    nkp.angle = kpList[i].direction;
    nkp.octave = (int) oct;

    /*
    nkp.x = (float) ocv_kp[i].pt.x;
    nkp.y = (float) ocv_kp[i].pt.y;
    nkp.resp = (float) ocv_kp[i].response;
    nkp.direction = (float) ocv_kp[i].angle;
    nkp.octave = (int) uOctave;
    nkp.scale = (int) uLayer;
    */

    //std::cout << "X, Y: " << nkp.x << ", " << nkp.y << std::endl;
    //std::cout << "Resp: " << uLayer << std::endl;
    //std::cout << "Octv: " << nkp.octave << std::endl;
    //std::cout << "Angl: " << nkp.direction << std::endl;
    //std::cout << "Scal: " << nkp.scale << std::endl;

    //std::cout << "push_back 1" << std::endl;;
    ocv_kp.push_back(nkp);
    //std::cout << "push_back 2" << std::endl;;
  }
}
