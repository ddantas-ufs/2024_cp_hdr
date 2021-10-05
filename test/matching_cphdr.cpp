#include "../include/cphdr.h"

/*
** THIS PROGRAM PERFORM THE MATCHING OF TWO IMAGES,
** USING ENTIRELY OPENCV2 TOOLS.
**
** THIS PROGRAM EXPECTS ARGUMENTS
** 
** args[1]: image1
** args[2]: image2
** args[3]: matching output
** 
** @author arturxz, 01/10/2021 (created)
*/
int main(int argv, char** args)
{
  // CREATING OBJECTS
  cv::Mat inputImage1, inputImage2, grayInputImage1, grayInputImage2,
          descriptor1, descriptor2, outputImage;
  std::string outImageName1, outImageName2, outputPath;
  std::vector<cv::KeyPoint> kpsImage1, kpsImage2;

  // SHOWING INPUTS
  std::cout << "received " << argv << " arguments." << std::endl;
  
  for( int i = 0; i < argv; i++ )
    std::cout << "  args[" << i << "]: " << args[i] << std::endl;

  // READING IMAGES AND SETTING OUTPUT IMAGE NAME
  std::cout << "Reading images..." << std::endl;
  readImg(args[1], inputImage1, grayInputImage1, outImageName1);
  outputPath = std::string(args[3]) + "match_" +outImageName1;

  readImg(args[2], inputImage2, grayInputImage2, outImageName2);
  outputPath = outputPath + "_"+ outImageName2 +".hdr";

  if ( inputImage1.empty() || inputImage2.empty() || grayInputImage1.empty() || grayInputImage2.empty() )
  {
    std::cout << "Could not open or find the image!" << std::endl;

    std::cout << "inputImage1 is empty? " << inputImage1.empty() << std::endl;
    std::cout << "inputImage2 is empty? " << inputImage2.empty() << std::endl;
    std::cout << "grayInputImage1 is empty? " << grayInputImage1.empty() << std::endl;
    std::cout << "grayInputImage2 is empty? " << grayInputImage2.empty() << std::endl;
    return -1;
  }

  std::cout << " ####################################" << std::endl;
  std::cout << " --> Printing in array .at<cv::Vec3f>(5, 5)" << std::endl;
  std::cout << inputImage1.at<cv::Vec3f>(5, 5) << std::endl;

  std::cout << " --> Mapping pixels" << std::endl;
  mapPixelValues01(inputImage1, outputImage);
  std::cout << " --> Pixels mapped" << std::endl;

//  printMat( grayInputImage1, "Original subregion" );
//  printMat( outputImage, "Mapped subregion" );

  // SAVING OUTPUT IMAGE
  std::cout << "Saving original image..." << std::endl;
  cv::imwrite(outputPath+"_original.hdr", inputImage1);

//  std::cout << "Saving output image..." << std::endl;
//  cv::imwrite(outputPath+".hdr", outputImage);

  return 0;
}