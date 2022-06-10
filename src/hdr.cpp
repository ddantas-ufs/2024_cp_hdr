#include "../include/detectors/hdr.h"
#include "../include/detectors/aux_func.h"

void logTransform( cv::Mat img, cv::Mat &out )
{
	out = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32F);

	for(int y = 0; y < img.rows; y++)
	{
		for(int x = 0; x < img.cols; x++)
		{
			float r = img.at<float>(y, x);
			float val = LOG_TRANSFORM_CONSTANT * log10(r + 1);
			out.at<float>(y, x) = val;
		}
	}
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
  int n = 3; // maskSize is odd, cp_hdr defaults to 3
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
          float v = 0.0f;
          v = std::pow( (aux.at<float>(y, x) - mean), 2 );
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
  //mapPixelValues( resp1, res );
  resp1.copyTo( res );

  //cv::normalize( saida, saida, 0.0f, 255.0f, cv::NORM_MINMAX, CV_32F, cv::Mat() );
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


/* ------------------------------ METODOS ORIGINAIS ------------------------------ */
/*
//função para aplicar a tranformação logaritmica na image HDR
//Parametros: c constante de multiplicacao da formula
void logTranformUchar( cv::Mat src, int c, cv::Mat &out ) {
  cv::Mat ret = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_32F);
	for(int y = 0; y < src.rows; y++){
		for(int x = 0; x < src.cols; x++){
			float r = src.at<float>(y, x);
			float val = c * log10(r + 1);
			ret.at<float>(y, x) = val;
		}
	}
  //return src;
  ret.copyTo( out );
}
*/
//função para aplicar a tranformação logaritmica na image HDR
//Parametros: c constante de multiplicacao da formula
void logTranformUchar( cv::Mat src, int c, cv::Mat &out)
{
	for(int y = 0; y < src.rows; y++){
		for(int x = 0; x < src.cols; x++){
			float r = src.at<uchar>(y, x);
			float val = c * log10(r + 1);
			src.at<uchar>(y, x) = val;
		}
	}
	src.copyTo( out );
}
/*
//Coeficiente de Variacao
void coefficienceOfVariationMask( cv::Mat aux, cv::Mat &out ) {
	
  float mediaGeral = 0; //Media geral dos coefV
	if(aux.depth() != CV_32F)
		aux.convertTo(aux, CV_32F);
	
	cv::Mat response = aux;
	
	cv::Mat auxResponse = cv::Mat::zeros(cv::Size(response.cols, response.rows), CV_32F);
	
	int n = 3; //maskSize impar
	int N = n*n, cont = 0;//quantidade de pixels visitados
	
	cv::Mat response2 = cv::Mat::zeros(cv::Size(response.cols, response.rows), CV_64F);
	//response * response 
	for(int y = 0; y < response.rows; y++)
		for(int x = 0; x < response.cols; x++)
			response2.at<float>(y, x) = (response.at<float>(y, x) * response.at<float>(y, x));
	
	float sum1 = 0, sum2 = 0;
	
	for(int y = 1; y < n; y++){
		for(int x = 0; x <= n; x++){
			sum1 += response.at<float>(y, x);
			sum2 += response2.at<float>(y, x);
		}
	}
	
	//"Convolution"
	for(int i = (n/2)+1; i < response.rows - (n/2); i++){
		int yBeg = i-(n/2), yEnd = i+(n/2);
		for(int j = (n/2); j < response.cols - (n/2); j++){
			//passando mascara 
			float sumVal = 0, sumVal2 = 0, maior = 0;
			int xBeg = j-(n/2), xEnd = j+(n/2);
			
			for(int y = yBeg; y <= yEnd; y++){
				for(int x = xBeg; x <= xEnd; x++){
					sumVal += response.at<float>(y, x);
					sumVal2 += response2.at<float>(y, x);
					maior = std::max(maior, response.at<float>(y, x));
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

	mediaGeral = mediaGeral/((response.cols-n)*(response.rows-n));
	printf("Media do Coefv = %.10f\n", mediaGeral);//
	
	//Response recebe o valor de coef salvo em aux
	response = auxResponse;
	
	cv::Mat aux2;
	cv::normalize(response, aux2, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());
	
	//return aux2;
  out = aux2;
}
*/
//cv::Mat coefficienceOfVariationMask(Mat aux){
void coefficienceOfVariationMask( cv::Mat aux, cv::Mat &out ) {
	
	if(aux.depth() != CV_32F)
		aux.convertTo(aux, CV_32F);
	
  float mediaGeral = 0.0f;
	cv::Mat response = aux;
	
	cv::Mat auxResponse = cv::Mat::zeros(cv::Size(response.cols, response.rows), CV_32F);
	
	int n = 3;//maskSize impar
	int N = n*n, cont = 0;//quantidade de pixels visitados
	
	cv::Mat response2 = cv::Mat::zeros(cv::Size(response.cols, response.rows), CV_64F);
	//response * response 
	for(int y = 0; y < response.rows; y++)
		for(int x = 0; x < response.cols; x++)
			response2.at<float>(y, x) = (response.at<float>(y, x) * response.at<float>(y, x));
	
	float sum1 = 0, sum2 = 0;
	
	for(int y = 1; y < n; y++){
		for(int x = 0; x <= n; x++){
			sum1 += response.at<float>(y, x);
			sum2 += response2.at<float>(y, x);
		}
	}

  //"Convolution"
  for(int i = (n/2)+1; i < response.rows - (n/2); i++){
    int yBeg = i-(n/2), yEnd = i+(n/2);
    for(int j = (n/2); j < response.cols - (n/2); j++){
      //passando mascara 
      float sumVal = 0, sumVal2 = 0, maior = 0;
      int xBeg = j-(n/2), xEnd = j+(n/2);
      
      for(int y = yBeg; y <= yEnd; y++){
        for(int x = xBeg; x <= xEnd; x++){
          sumVal += response.at<float>(y, x);
          sumVal2 += response2.at<float>(y, x);
          maior = std::max(maior, response.at<float>(y, x));
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

  mediaGeral = mediaGeral/((response.cols-n)*(response.rows-n));
  printf("Media do Coefv = %.10f\n", mediaGeral);//

  //Response recebe o valor de coef salvo em aux
  response = auxResponse;

  //cv::Mat aux2;
  normalize(response, out, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat());

  //return aux2;
}
