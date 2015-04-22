/* 
GPCVLibrary Copyright (C) 2014 
Victor Raul Lopez Lopez,  
Leonardo Trujillo Reyes, 
Gustavo Olague Caballero, 
Pierrick Legrand.

GPCVLibrary is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

GPCVLibrary is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with GPCVLibrary. If not, see <http://www.gnu.org/licenses/>. 
*/

#include "GPCV.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>
#include <iostream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"



using namespace GPCV;
using namespace cv;
using namespace std;


int main( int argc, char** argv )
{

  if( argc != 3 )
  { std::cout << " Usage: ./GPCVexample3 <img1> <img2>" << std::endl;
	return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { return -1; }

  //-- Step 1: Convert images to CV_32FC1 

	img_1.convertTo(img_1,CV_32FC1);
  img_2.convertTo(img_2,CV_32FC1);

  //-- Step 2: Detect the keypoints using detectorMOP()
  
  // Detector Variables {
  Mat output1, output2;
  vector<cv::KeyPoint> keypoints_1, keypoints_2;

	double w1 = 0.80; // Weight parameter of the MOP operator from the eq.(14) ref ().  
	double QL1 = 0.55; // Threshold parameter

	double w2 = 0.80;
	double QL2 = 0.40;

 	double s = 55;
  double md= ((s/2)); 
  bool ori = true; 
  // }


	detectorMOP(img_1, output1, keypoints_1, w1, QL1, md, Mat(), s, ori);
  detectorMOP(img_2, output2, keypoints_2, w2, QL2, md, Mat(), s, ori);


  if (keypoints_1.empty())
  cerr << " Keypoint from image 1 is empty " << endl;

  if (keypoints_2.empty())
  cerr << " Keypoint from image 2 is empty " << endl;

  //-- Step 3: Calculate descriptors (feature vectors) using holderDescriptor()

  // Descriptor Variables {
  
  vector<int> samples;
	samples.push_back(20); // # of samples of the first circle
	samples.push_back(60); // # of samples of the second circle
	samples.push_back(80); // # of samples of the third circle


  int sum = accumulate(samples.begin(), samples.end(),0) + 1; // Total number of samples, + 1 is the center point

  cv::Mat descriptors_1(keypoints_1.size(),sum,CV_32F);
  cv::Mat descriptors_2(keypoints_2.size(),sum,CV_32F);
  // }

  holderDescriptor(img_1, keypoints_1, samples, descriptors_1);
  holderDescriptor(img_2, keypoints_2, samples, descriptors_2);

  //-- Step 4: Matching descriptor vectors using FLANN matcher

  BFMatcher matcher(NORM_L2,true);
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  img_1.convertTo(img_1,CV_8UC1);
  img_2.convertTo(img_2,CV_8UC1);

  drawKeypoints(img_1, keypoints_1, img_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints(img_2, keypoints_2, img_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Draw matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

  //-- Show detected matches
  imshow("Matches", img_matches );


  waitKey(0);

  return 0;

}


