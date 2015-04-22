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

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>

using namespace cv;
using namespace std;

namespace GPCV{

void nonMaximaSuppression(const cv::Mat& src,  cv::Mat& dst1, const int sz, double qualityLevel, const cv::Mat mask1);
void detectorMOP(cv::Mat image, cv::Mat &Operator, vector<cv::KeyPoint> &keypoints_,double W,double qualityLevel, double minDistance, cv::Mat Mask, int patchSize , bool orientation);
void holderOperator(cv::Mat im, cv::Mat &outim);
void holderDescriptor(cv::Mat imageInput1, vector<cv::KeyPoint> keypoints, vector<int> samplesInput, cv::Mat &descriptorOuput);


}
