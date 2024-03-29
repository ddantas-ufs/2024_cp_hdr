# Control Points for High Dynamic Range images (CP_HDR)

This is the README for CP_HDR library. Please, feel free to suggest new features, improvements or make a pull request.

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

This library is developed to easily generate detection and description of feature points (aka: keypoints) using HDR images as input and metrics to evaluate algorithms performance in detection and description. 

For now, we provide canonical Harris Corner Detector and SIFT detector and descriptor. The SIFT descriptor can easely be used with other detectors (Harris, for example). We also provide a Harris Corner and SIFT HDR-specialized algorithms, based on [Melo et al.](https://doi.org/10.1109/ISCC.2018.8538716) and [Nascimento et al.](https://doi.org/10.5220/0010779700003124) papers.

### Prerequisites

First of all, install OpenCV 3.4.x.
We recommend the following tutorials: [Tutorial 1](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/), [Tutorial 2](https://learnopencv.com/install-opencv3-on-ubuntu/), [Tutorial 3](https://linuxhint.com/install-opencv-ubuntu/)

Secondly, on use your package manager to install OpenCV headers.
In debian-like distros, you can use something like:

```
sudo apt install libopencv-dev
```

## Getting Started <a name = "getting_started"></a>

Using CP_HDR you can: 
- Detect feature points using LDR and HDR images
- Describe feature points using LDR and HDR images
- Save a .txt archive with detected the keypoints detected (optionally, description for each feature point)
- Read a previously saved list of feature points saved
- Plot feature points into the image
- Make a feature point matching between two images
- Calculate some detection and description metrics
  - Repeatability rate
  - Uniformity rate
  - Mean average precision
  - Matching performance
- Segment image using a Retinex-based algorithm based in the image ilumination

### Installing

For now, no installation needed.

## Usage <a name = "usage"></a>

