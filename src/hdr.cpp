#include "../include/detectors/hdr.h"
#include "../include/detectors/aux_func.h"

void logTransform(cv::Mat img, cv::Mat &img_log10)
{
  cv::Mat img_aux, img_ln;
  img_log10 = cv::Mat::zeros( cv::Size( img.cols, img.rows ), CV_32F );
  float ln10 = 2.302585;
  int rmax = 255;

  double minVal; 
  double maxVal; 
  cv::Point minLoc; 
  cv::Point maxLoc;

  cv::minMaxLoc( img, &minVal, &maxVal, &minLoc, &maxLoc );
  printf("Original image: Max value = %.10f, Min value = %.10f\n", maxVal, minVal);
  minVal = 0.0, maxVal = 0.0;

  for( int i = 0; i < img.rows; i++ )
  {
    for( int j = 0; j < img.cols; j++ )
    {
      float r = img.at<float>(i, j);
      float val =  ln10 * std::log10( r + 1.0f );
      img_log10.at<float>(i, j) = val;
    }
  }
  
  cv::minMaxLoc( img_log10, &minVal, &maxVal, &minLoc, &maxLoc );
  printf("Log10 image: Max value = %.10f, Min value = %.10f\n", maxVal, minVal);

  mapPixelValues(img_log10, img_log10);
}


/**
 * Implementation of the Coefficient of Variation 
 * 
 * @param img: input image that will be convoluted
 * @param res: output image, where result will be stored
 * 
**/
void applyCVMask( cv::Mat img, cv::Mat &res )
{
  // Initial validations
  if( img.empty() ) {
    return;
  } else if( img.depth() != CV_32F ) {
    img.convertTo(img, CV_32F);
  }

  float mediaGeral = 0.0f;
  int n = CV_SIZE; // maskSize is odd, cp_hdr defaults to 5 
  int N = n*n; // total amount of pixels being analised

  cv::Mat aux = img;
  cv::Mat saida = cv::Mat::zeros( cv::Size( aux.cols, aux.rows ), CV_32F );
  cv::Mat resp1 = cv::Mat::zeros( cv::Size( aux.cols, aux.rows ), CV_32F ); // convoluted image
  
  // Convoluting
  for(int i = (n/2)+1; i < aux.rows - (n/2); i++)
  {
    int yBeg = i-(n/2);
    int yEnd = i+(n/2);
    for(int j = (n/2); j < aux.cols - (n/2); j++)
    {
      float mean = 0.0f, variation = 0.0f;
      float sum = 0.0f;
      //double sqSum = 0.0f;

      int xBeg = j-(n/2);
      int xEnd = j+(n/2);
      
      // Calculating mean
      for(int y = yBeg; y <= yEnd; y++)
      {
        for(int x = xBeg; x <= xEnd; x++)
        {
          sum += aux.at<float>(y, x);
          //sqSum += std::pow(aux.at<float>(y, x), 2);
        }
      }

      mean = sum / N;
      
      // Calculating variation
      for(int y = yBeg; y <= yEnd; y++)
      {
        for(int x = xBeg; x <= xEnd; x++)
        {
          float v = std::pow( (aux.at<float>(y, x) - mean), 2 );
          variation += v;
        }
      }

      variation = variation / N;

      float SD = sqrt( variation ); // standard deviation
      float CV = std::abs(mean) < 0.0001 ? 0.0f : SD/N; // coefficient of variation

      if( std::isnan( SD ) || std::isnan( CV ) ) {
        printf("          -----: media=%.10f, variancia=%.10f, DP=%.10f, CV=%.10f\n", mean, variation, SD, CV);
      }

      resp1.at<float>(i, j) = CV * 100.0f;
      saida.at<float>(i, j) = img.at<float>(i, j);
      mediaGeral += CV;
    }
  }

  int denominador = ((aux.cols-n)*(aux.rows-n));
  float mediaGeral2 = mediaGeral/denominador;
  printf("Media do Coefv = %.10f / %d = %.10f\n", mediaGeral, ((aux.cols-n)*(aux.rows-n)), mediaGeral2);
  
  // Auxiliar matrix gets the "convoluted" image
  //aux = resp1;

  // normalizing image and copying data to output matrix
  mapPixelValues( resp1, res );

  mapPixelValues( saida, saida );
  cv::imwrite( "out/saida_teste.png", saida );
}

void applyCVMask( cv::Mat &img )
{
  cv::Mat res;
  applyCVMask(img, res);
  res.copyTo( img );
}

void calculaPonto( int rowAtual, int colAtual, int row, int col, int maxRow, int maxCol, 
                   int &rowRes, int &colRes )
{
  rowRes = rowAtual + row;
  colRes = colAtual + row;

  if( rowRes < 0 ) rowRes = cv::abs(rowRes);
  if( colRes < 0 ) colRes = cv::abs(colRes);

  if( rowRes >= maxRow-1 ) rowRes = ( maxRow-1 - (rowRes - maxRow-1) );
  if( colRes >= maxCol-1 ) colRes = ( maxCol-1 - (colRes - maxCol-1) );

//  printf("--> rowAtual: %d, colAtual: %d, row: %d, col: %d\n", rowAtual, colAtual, row, col );
//  printf("----> maxRow: %d,  maxCol: %d, rowRes: %d, colRes: %d\n", maxRow, maxCol, rowRes, colRes );

}
