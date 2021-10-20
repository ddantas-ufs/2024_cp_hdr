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
  cv::Mat inputImage1, inputImage2, grayInputImage1, grayInputImage2,
          descriptor1, descriptor2, outputImage;
  std::string outImageName1, outImageName2, outputPath;
  
  std::vector<KeyPoints> kp1, kp2;//, subKp1, subKp2;

  std::string imgPath = args[1], hdrSuf = ".hdr";
  std::string outDir = std::string(args[3]);

  // Showing inputs
  std::cout << "----------------------------------" << std::endl;
  std::cout << "Received " << argv << " arguments." << std::endl;
  for( int i = 0; i < argv; i++ )
    std::cout << "  args[" << i << "]: " << args[i] << std::endl;

  // Reading images and setting output image name
  std::cout << "Reading images..." << std::endl;
  readImg(args[1], inputImage1, grayInputImage1, outImageName1);
  readImg(args[2], inputImage2, grayInputImage2, outImageName2);

  // Setting up output image name extension based on input image name
  if( 0 == imgPath.compare(imgPath.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
    outputPath = outDir +outImageName1 + "_"+ outImageName2 +"_matching.hdr";
  else
    outputPath = outDir +outImageName1 + "_"+ outImageName2 +"_matching.png";

  // Granting that all images are not empty
  if ( inputImage1.empty() || inputImage2.empty() || grayInputImage1.empty() || grayInputImage2.empty() )
  {
    std::cout << "Could not open or find the image!" << std::endl;

    std::cout << "inputImage1 is empty? " << inputImage1.empty() << std::endl;
    std::cout << "inputImage2 is empty? " << inputImage2.empty() << std::endl;
    std::cout << "grayInputImage1 is empty? " << grayInputImage1.empty() << std::endl;
    std::cout << "grayInputImage2 is empty? " << grayInputImage2.empty() << std::endl;
    return -1;
  }

  mapPixelValues( inputImage1, inputImage1 );
  mapPixelValues( inputImage2, inputImage2 );
//  mapPixelValues01( grayInputImage1, grayInputImage1 );
//  mapPixelValues01( grayInputImage2, grayInputImage2 );

//  concatenateImages(grayInputImage1, grayInputImage2, outputImage);

  /*
  double imgMin = 0.0, imgMax = 0.0;
  cv::minMaxLoc( inputImage1, &imgMin, &imgMax );
  std::cout << "----> inputImage1 imgMin: " << imgMin << ", imgMax: " << imgMax << std::endl;
  cv::minMaxLoc( inputImage2, &imgMin, &imgMax );
  std::cout << "----> inputImage2 imgMin: " << imgMin << ", imgMax: " << imgMax << std::endl;
  cv::minMaxLoc( outputImage, &imgMin, &imgMax );
  std::cout << "----> outputImage imgMin: " << imgMin << ", imgMax: " << imgMax << std::endl;

  std::cout << "Saving concatenated image into: " << outputPath << std::endl;
  cv::imwrite(outputPath, outputImage);
  //cv::imwrite(outputPath, outputImage);
  
  return 0;
  */
  /*////////////////////////////////////
  imgMin = 0.0, imgMax = 0.0;
  cv::minMaxLoc( inputImage1, &imgMin, &imgMax );
  std::cout << "----> imgMin: " << imgMin << ", imgMax: " << imgMax << std::endl;

  std::cout << " --> Mapping pixels" << std::endl;
  mapPixelValues01( inputImage1, outputImage );
  std::cout << " --> Pixels mapped" << std::endl;

  /////////////////////////////////////
  imgMin = 0.0, imgMax = 0.0;
  cv::minMaxLoc( outputImage, &imgMin, &imgMax );
  std::cout << "----> imgMin: " << imgMin << ", imgMax: " << imgMax << std::endl;
  */

  std::cout << "Detecting Keypoints" << std::endl;
  dogKp(grayInputImage1, kp1);
  dogKp(grayInputImage2, kp2);

  std::cout << "Sorting Keypoint vectors" << std::endl;
  sortKeypoints( kp1 );
  sortKeypoints( kp2 );
  
  std::cout << "Slicing Keypoint vectors" << std::endl;
  std::cout << " --> Before kp1: " << kp1.size() << " kp2: " << kp2.size() << std::endl;
  kp1 = vectorSlice( kp1, 0, 10000); // 10k items
  kp2 = vectorSlice( kp2, 0, 10000); // 10k items
  std::cout << " --> After kp1: " << kp1.size() << " kp2: " << kp2.size() << std::endl;

  /*
  for(int k = 0; k < 1000; k++)
  {
    subKp1.push_back( kp1[k] );
    subKp2.push_back( kp2[k] );
  }
  */
  
  std::cout << "Describing KP1" << std::endl;
  siftDescriptor(kp1, inputImage1, grayInputImage1);
  std::cout << "KP1 described" << std::endl;

  std::cout << "Describing KP2" << std::endl;
  siftDescriptor(kp2, inputImage2, grayInputImage2);
  std::cout << "KP2 described" << std::endl;

  /*
  // Generating images description
  std::cout << "Detecting and Describing Keypoints Image 1" << std::endl;
  dogKp(grayInputImage1, kp1);
  siftDescriptor(kp1, inputImage1, grayInputImage1);

  std::cout << "Detecting and Describing Keypoints Image 2" << std::endl;
  dogKp(grayInputImage2, kp2);
  siftDescriptor(kp2, inputImage2, grayInputImage2);
  */

  // Saving images description
  if( 0 == imgPath.compare(imgPath.size()-hdrSuf.size(), hdrSuf.size(), hdrSuf) )
  {
    std::cout << "--> outImageName1: " +outDir+outImageName1+".hdr.sift.txt" << std::endl;
    std::cout << "--> outImageName2: " +outDir+outImageName2+".hdr.sift.txt" << std::endl;
    saveKeypoints(kp1, outDir+outImageName1+".hdr.sift.txt", kp1.size());
    saveKeypoints(kp2, outDir+outImageName2+".hdr.sift.txt", kp2.size());

    plotKeyPoints(inputImage1, kp1, outDir+outImageName1+"_kps.hdr", kp1.size());
    plotKeyPoints(inputImage2, kp2, outDir+outImageName2+"_kps.hdr", kp2.size());
  }
  else
  {
    std::cout << "--> outImageName1: " +outDir+outImageName1+".ldr.sift.txt" << std::endl;
    std::cout << "--> outImageName2: " +outDir+outImageName2+".ldr.sift.txt" << std::endl;
    saveKeypoints(kp1, outDir+outImageName1+".ldr.sift.txt", kp1.size());
    saveKeypoints(kp2, outDir+outImageName2+".ldr.sift.txt", kp2.size());

    plotKeyPoints(inputImage1, kp1, outDir+outImageName1+"_kps.png", kp1.size());
    plotKeyPoints(inputImage2, kp2, outDir+outImageName2+"_kps.png", kp2.size());
  }

  // SAVING OUTPUT IMAGE
  std::cout << "Matching FPs and saving resulting image" << std::endl;
  matchFPs(inputImage1, kp1, inputImage2, kp2, outputImage);
  cv::imwrite(outputPath, outputImage);

  kp1.clear();
  kp2.clear();
//  subKp1.clear();
//  subKp2.clear();
  return 0;
}