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

using namespace GPCV;
using namespace cv;
using namespace std;


int main()
{
cvNamedWindow("MOP Keypoints", 1); //Create window
CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY); //Capture using any camera connected to your system

Mat output;
char key;
int w_=50;
int QL=50;
int mD=10;

double iw_;
double iQL;

createTrackbar("W % ", "MOP Keypoints", &w_, 100);
createTrackbar("Quality % ","MOP Keypoints", &QL, 100);
createTrackbar("Min Distance [pixels] ", "MOP Keypoints", &mD, 100);

while(true)
{

//-- Step 1:Create image frames from capture and Convert images to CV_32FC1 

iw_  = (w_)*0.01;
iQL  = (QL)*0.01;

vector<cv::KeyPoint> keypoints_1;

Mat img_1 = cvQueryFrame(capture); 
img_1.convertTo(img_1,CV_32FC1);
cvtColor(img_1, img_1, CV_RGB2GRAY );

//Condition to avoid minDistance = 0 
if(mD==0)
mD=1;

//-- Step 2: Detect the keypoints using detectorMOP() 

detectorMOP(img_1, output, keypoints_1, iw_,iQL, mD, Mat(), 21 , false);

//-- Step 3: Draw Keypoints and show frames on created window

img_1.convertTo(img_1,CV_8UC1);
drawKeypoints(img_1, keypoints_1, img_1, Scalar(0,0,255), DrawMatchesFlags::DEFAULT );

normalize(output,output, 0, 1, CV_MINMAX);

imshow("MOP detector operator", output);
imshow("MOP Keypoints", img_1); 

key = cvWaitKey(2); //Capture Keyboard stroke
if (char(key) == 27){
break; //If you hit ESC key loop will break.
}

}

return 0;

}
