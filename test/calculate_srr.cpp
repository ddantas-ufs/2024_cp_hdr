#include "../include/cphdr.h"

std::string finalOut = "Image 1;Image 2; Repeatability Rate\n";

void loadRanaDataset( char** args, std::vector< std::vector<KeyPoints> > &lKps )
{
    for( int i = 2; i < 9; i++ )
    {
      std::string arq = std::string( args[i] );
      std::cout << i << ": " << arq << std::endl;
      lKps.push_back( loadKeypoints( arq ) );
    }
}

float calculateSRR( std::vector< std::vector<KeyPoints> > &lKps )
{
  float sumRR = 0.0f;
  int counter = 0;
  for(int i=0; i < lKps.size(); i++)
  {
    for(int j=i+1; j < lKps.size(); j++ )
    {
      std::cout << i << ", " << j << std::endl;
      
      float rr = 0.0f;
      int cc = 0;
      calculateRR(cv::Mat(), lKps[i], lKps[j], cc, rr);
      
      finalOut += std::to_string(i) +";" +std::to_string(j) +";" +std::to_string(rr) +"\n";
      sumRR += rr;
      counter++;
    }
  }
  return sumRR/counter;
}

/**
 * @brief Program that will calculate de Summarized Repeatability Rate from a dataset (SRR).
 * 
 * @param args will contain an integer that says which dataset is being calculated.
 *             after that, the detected keypoints and all the Homography matrixes filepaths.
 *             The homography matrixes need to be ordered with input keypoints.
 *  Integers: 1 Rana LightRoom
 *            2 Rana ProjectRoom
 *            3 Pribyl 2D Lighting
 *            4 Pribyl 2D Distance
 *            5 Pribyl 2D Viewpoint
 */
int main(int argv, char** args)
{
  // DATASET
  int dataset = std::stoi( std::string(args[1]) );
  std::vector< std::vector<KeyPoints> > lKps;

  std::string outDir = std::string( args[argv-1] );

  std::cout << outDir << std::endl;

  //return 0;

  if( dataset == 1 || dataset == 2 )
    loadRanaDataset( args, lKps );

  for( int i = 0; i < lKps.size(); i++ )
  {
    printf( "Quantidade de Keypoints da imagem %i: %li\n", i, lKps[i].size() );
    //  for( int j = 0; j < lKps[i].size(); j++ )
    //    printf( "x: %.4f,\ty: %.4f,\toct: %i,\tscl: %i,\tResp: %.4f\n", lKps[i][j].x, lKps[i][j].y, lKps[i][j].octave, lKps[i][j].scale, lKps[i][j].resp );
  }

  float SRR = calculateSRR( lKps );
  finalOut += "Total;;" +std::to_string(SRR) +"\n";

  writeTextFile( outDir + "SRR.csv", finalOut );

  return 0;
}
