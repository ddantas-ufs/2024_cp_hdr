#include "../include/detectors/hdr.h"
#include "../include/detectors/aux_func.h"

/*
void coefVar(cv::Mat img, cv::Mat &img_cv, int mask_size, bool gauss, float sigma)
{
  cv::Mat cv = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
  cv::Mat img2 = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
  cv::Mat g_kernel;
  int mask_mid = mask_size / 2;
  float N = float(mask_size * mask_size);
  int kernel_sum;
  float mask_sum, mean;
  double mask_sum2, var, std_dev, coef_var;

  for (int y = 0; y < img.rows; y++)
  {
    for (int x = 0; x < img.cols; x++)
    {
      img2.at<double>(y, x) = img.at<float>(y, x) * img.at<float>(y, x);
    }
  }

  if (gauss)
  {
    gaussKernel(g_kernel, mask_size, sigma);
  }
  else
  {
    kernel_sum = N;
  }

  for (int y = mask_mid; y < (img.rows - mask_mid); y++)
  {
    for (int x = mask_mid; x < (img.cols - mask_mid); x++)
    {
      mask_sum = 0.0;
      mask_sum2 = 0.0;

      for (int r = y - mask_mid, i = 0; r < (y + mask_mid + 1); r++, i++)
      {
        for (int c = x - mask_mid, j = 0; c < (x + mask_mid + 1); c++, j++)
        {
          mask_sum += img.at<float>(r, c);
          if (gauss)
          {
            mask_sum2 += img2.at<double>(r, c) * g_kernel.at<float>(i, j);
          }
          else
          {
            mask_sum2 += img2.at<double>(r, c);
          }
        }
      }
      mean = mask_sum / N ;
      var = (mask_sum2 / N) - (mean * mean * kernel_sum) / N;
      std_dev = sqrt(var);

      if (mean < 1e-5)
      {
        coef_var = 0.0;
      }
      else
      {
        coef_var = std_dev / mean;
      }
      cv.at<double>(y, x) = coef_var;
    }
  }
//  cv.convertTo(img_cv, CV_32FC1);
//  cv::normalize(img_cv, img_cv, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  mapPixelValues(cv, img_cv);
}
*/

void logTransform(cv::Mat img, cv::Mat &img_log10)
{
  cv::Mat img_aux, img_ln;
  float ln10 = 2.302585;
  int rmax = 255;

  double minVal; 
  double maxVal; 
  cv::Point minLoc; 
  cv::Point maxLoc;

  cv::minMaxLoc( img, &minVal, &maxVal, &minLoc, &maxLoc );
  printf("Original image: Max value = %.10f, Min value = %.10f\n", maxVal, minVal);
  minVal = 0.0, maxVal = 0.0;

  img_aux = (img * rmax) + 1;
  cv::log(img_aux, img_ln);
  img_log10 = img_ln / ln10;
  img_log10 = img_log10 / std::log10(rmax + 1);

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
  if( img.empty() )
    return;
  else if( img.depth() != CV_32F )
    img.convertTo(img, CV_32F);
  
  float mediaGeral = 0.0f;
  int n = CV_SIZE; // maskSize is odd, cp_hdr defaults to 5 
  int N = n*n, cont = 0; // total amount of pixels being analised

  cv::Mat aux = img;
  cv::Mat resp1 = cv::Mat::zeros( cv::Size( aux.cols, aux.rows ), CV_32F ); // convoluted image
  
  float sum1 = 0, sum2 = 0;
  
  for(int y = 1; y < n; y++)
    for(int x = 0; x <= n; x++)
      sum1 += aux.at<float>(y, x);
  
  // Convoluting
  for(int i = (n/2)+1; i < aux.rows - (n/2); i++)
  {
    int yBeg = i-(n/2), yEnd = i+(n/2);
    for(int j = (n/2); j < aux.cols - (n/2); j++)
    {
      float mean = 0.0f, variation = 0.0f;
      float sum = 0.0f;
      double sqSum = 0.0f;
      int xBeg = j-(n/2), xEnd = j+(n/2);
      
      // Calculating mean
      for(int y = yBeg; y <= yEnd; y++)
        for(int x = xBeg; x <= xEnd; x++)
        {
          sum += aux.at<float>(y, x);
          sqSum += std::pow(aux.at<float>(y, x), 2);
        }

      mean = sum / N;
      
      /* Calculating variation
      for(int y = yBeg; y <= yEnd; y++)
        for(int x = xBeg; x <= xEnd; x++)
          variation += std::pow( std::abs( aux.at<float>(y, x) - mean ), 2.0f); 
      */
      // if mean is 0, variation is 0.
      if( std::abs( mean ) < 0.0001f ) variation = 0.0f;
      else variation = float( (sqSum/N) - (mean*mean) );
      //else variation = variation / mean;

      float SD = sqrt( variation ); // standard deviation
      float CV = std::abs(mean) < 0.0001 ? 0.0f : SD/mean; // coefficient of variation

      if( std::isnan( SD ) || std::isnan( CV ) )
        printf("          -----: media=%.10f, variancia=%.10f, DP=%.10f, CV=%.10f\n", mean, variation, SD, CV);

      resp1.at<float>(i, j) = CV * 100.0f;
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
}

//Coeficiente de Variacao
void applyCVMask_old(cv::Mat aux, cv::Mat &res)
{
  if(aux.depth() != CV_32F)
    aux.convertTo(aux, CV_32F);
  
  float mediaGeral = 0.0f;  
  res = aux;
  
  cv::Mat auxResponse = cv::Mat::zeros(cv::Size(res.cols, res.rows), CV_32F);
  
  int n = 5;//maskSize impar
  int N = n*n, cont = 0;//quantidade de pixels visitados
  
  cv::Mat response2 = cv::Mat::zeros(cv::Size(res.cols, res.rows), CV_64F);
  //response * response 
  for(int y = 0; y < res.rows; y++)
    for(int x = 0; x < res.cols; x++)
      response2.at<float>(y, x) = (res.at<float>(y, x) * res.at<float>(y, x));
  
  float sum1 = 0, sum2 = 0;
  
  for(int y = 1; y < n; y++){
    for(int x = 0; x <= n; x++){
      sum1 += res.at<float>(y, x);
      sum2 += response2.at<float>(y, x);
    }
  }
  
  //"Convolution"
  for(int i = (n/2)+1; i < res.rows - (n/2); i++){
    int yBeg = i-(n/2), yEnd = i+(n/2);
    for(int j = (n/2); j < res.cols - (n/2); j++){
      //passando mascara 
      float sumVal = 0, sumVal2 = 0, maior = 0;
      int xBeg = j-(n/2), xEnd = j+(n/2);
      
      for(int y = yBeg; y <= yEnd; y++){
        for(int x = xBeg; x <= xEnd; x++){
          sumVal += res.at<float>(y, x);
          sumVal2 += response2.at<float>(y, x);
          maior = cv::max(maior, res.at<float>(y, x));
        }
      }
            
      float media = sumVal/N;
      
      float variancia = (sumVal2/N) - (media*media);

      float S = sqrt(variancia); // desvio padrao
      float CV = media == 0? 0 : S/media; // Coef de Variacao
      auxResponse.at<float>(i, j) = CV * 100;
      
      //printf("%d %d %f\n", i, j, CV);
      //printf("%.8f %.8f %.8f %.8f %.8f %.8f\n", sumVal, sumVal2, media, variancia, S, CV);
      
      mediaGeral += CV;	
    }
  }

  mediaGeral = mediaGeral/((res.cols-n)*(res.rows-n));
  printf("Media do Coefv = %.10f\n", mediaGeral);//
  
  //Response recebe o valor de coef salvo em aux
  res = auxResponse;
  
  mapPixelValues( res, res );
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

void applyCVMask_new( cv::Mat in, cv::Mat &out )
{
  if(in.depth() != CV_32F)
    in.convertTo(in, CV_32F);

  cv::Mat aux = cv::Mat::zeros( cv::Size( in.cols, in.rows ), CV_32F ); // convoluted image
  out = cv::Mat::zeros( cv::Size( in.cols, in.rows ), CV_32F ); // convoluted image
  
  float mediaGeral = 0.0f;
  int n = CV_SIZE; // maskSize is odd, cp_hdr defaults to 5 
  int N = n*n, cont = 0; // total amount of pixels being analised
  int halfMask = int(n / 2);

  // CONVOLUTING IN IMAGE
  for( int i = 0; i < in.rows; i++ )
  {
    for( int j = 0; j < in.cols; j++ )
    {
      float mean = 0.0f, variation = 0.0f;
      float sum = 0.0f;
      double sqSum = 0.0f;

      for(int y = -halfMask; y <= halfMask; y++)
        for(int x = -halfMask; x <= halfMask; x++)
        {
          int row = 0, col = 0;
          calculaPonto( i, j, y, x, in.rows, in.cols, row, col );

          sum += in.at<float>(row, col);
          sqSum += std::pow(in.at<float>(row, col), 2);
        }
      
      mean = sum / N;
      
      // if mean is 0, variation is 0.
      if( std::abs( mean ) < 0.0001f ) variation = 0.0f;
      else variation = float( (sqSum/N) - (mean*mean) );
      //else variation = variation / mean;

      float SD = sqrt( variation ); // standard deviation
      float CV = std::abs(mean) < 0.0001 ? 0.0f : SD/mean; // coefficient of variation

      if( std::isnan( SD ) || std::isnan( CV ) )
        printf("          -----: media=%.10f, variancia=%.10f, DP=%.10f, CV=%.10f\n", mean, variation, SD, CV);

      aux.at<float>(i, j) = CV * 100.0f;
      mediaGeral += CV;
    }
  }
  int denominador = ((in.cols-n)*(in.rows-n));
  float mediaGeral2 = mediaGeral/denominador;
  printf("Media do Coefv = %.10f / %d = %.10f\n", mediaGeral, ((in.cols-n)*(in.rows-n)), mediaGeral2);
  
  // Auxiliar matrix gets the "convoluted" image
  //aux = resp1;

  // normalizing image and copying data to output matrix
  mapPixelValues( aux, out );
}

