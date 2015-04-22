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


using namespace cv;
using namespace std;

namespace GPCV{

void nonMaximaSuppression(const cv::Mat& src,  cv::Mat& dst1, const int sz, double qualityLevel, const cv::Mat mask1) {


	double minStrength; 
	double maxStrength;
	int threshold1;

        minMaxLoc(src,&minStrength,&maxStrength);
	threshold1= qualityLevel*maxStrength;
        threshold(src,src,threshold1,255,3);

	const int M = src.rows;
	const int N = src.cols;
	const bool masked = !mask1.empty();
	Mat block = 255*Mat_<uint8_t>::ones(Size(2*sz+1,2*sz+1));
	dst1 = Mat_<uint8_t>::zeros(src.size());

	for (int m = 0; m < M; m+=sz+1) {
		for (int n = 0; n < N; n+=sz+1) {
			Point  ijmax;
			double vcmax, vnmax;

			Range ic(m, min(m+sz+1,M));
			Range jc(n, min(n+sz+1,N));

			minMaxLoc(src(ic,jc), NULL, &vcmax, NULL, &ijmax,masked ? mask1(ic,jc) : noArray() );
			Point cc = ijmax + Point(jc.start,ic.start);

			Range in(max(cc.y-sz,0), min(cc.y+sz+1,M));
			Range jn(max(cc.x-sz,0), min(cc.x+sz+1,N));

			Mat_<uint8_t> blockmask;
			block(Range(0,in.size()), Range(0,jn.size())).copyTo(blockmask);
			Range iis(ic.start-in.start, min(ic.start-in.start+sz+1, in.size()));
			Range jis(jc.start-jn.start, min(jc.start-jn.start+sz+1, jn.size()));
			blockmask(iis, jis) = Mat_<uint8_t>::zeros(Size(jis.size(),iis.size()));

			minMaxLoc(src(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? mask1(in,jn).mul(blockmask) : blockmask);
			Point cn = ijmax + Point(jn.start, in.start);

			if (vcmax > vnmax) {
				dst1.at<uint8_t>(cc.y, cc.x) = 255;
			}
		}
	}


}



void detectorMOP(cv::Mat image, cv::Mat &Operator, vector<cv::KeyPoint> &keypoints_,double W,double qualityLevel, double minDistance, cv::Mat Mask, int patchSize , bool orientation)
{

Mat dx,dy;
Mat m_angle;
Mat hist;
int nimages = 1; 
int channels[] = {0} ;
int dims = 1 ;
int histSize[] = {360}; 
float hranges[] = { 0, 360 }; 
const float *ranges[] = {hranges};
int angle=0;
double minVal1; double maxVal1; Point minLoc1; Point maxLoc1;
Point matchLoc1;

Mat mag,ang;

Mat dst,frame2;
GaussianBlur(image, dst, cv::Size(9,9),1.0,1.0);
absdiff(dst,image,dst); 
GaussianBlur(dst, dst, cv::Size(9,9),2.0,2.0); 
frame2=dst*W;
pow(image,2,dst);
GaussianBlur(dst, dst, cv::Size(9,9), 1.0,1.0);
log(dst,dst);
GaussianBlur(dst, dst, cv::Size(9,9), 1.0,1.0);
frame2=dst+frame2;
GaussianBlur(image, dst, cv::Size(9,9),1.0,1.0);   
divide(dst,image,dst);
dst=dst+frame2;
dst=abs(dst);
pow(dst,2,frame2);
GaussianBlur(frame2, frame2, cv::Size(9,9), 2.0,2.0);



if(Mask.empty())
{
Mask = cv::Mat::zeros(frame2.size(),CV_8UC1);
Mask(cv::Rect(minDistance,minDistance,(frame2.cols)-(2*minDistance),(frame2.rows)-(2*minDistance)))=1;
}

nonMaximaSuppression(frame2, dst, minDistance, qualityLevel, Mask);



	for( int y = 0; y < dst.rows; y++ ) {
    
			  const uchar* rowPtr = dst.ptr<uchar>(y);
    
			  for( int x = 0; x < dst.cols; x++ ) {
				  
				  if (rowPtr[x]) {

if ( orientation == false )
{


Mat image1(image(cv::Rect( x - int(patchSize/2), y - int(patchSize/2), patchSize, patchSize)));

Sobel(image1,dx,-1,0,1,7);
Sobel(image1,dy,-1,1,0,7); 


cartToPolar(dx, dy, mag, ang, false);

calcHist(&ang, nimages, channels, Mat(), hist, dims, histSize, ranges, true,false);


for(int x=0; x < mag.cols ;x++)
{

for(int y=0; y < mag.rows ;y++)
{

hist.at<float>(1,(int)(ang.at<float>(x,y))-1 ) = mag.at<float>(x,y)*hist.at<float>( 1,(int)(ang.at<float>(x,y))-1 );

}
}


     minMaxLoc( hist, &minVal1, &maxVal1, &minLoc1, &maxLoc1, Mat() );
 angle = maxLoc1.y;




}

					  keypoints_.push_back(cv::KeyPoint(cv::Point2f(x,y),patchSize,angle));
				  }
			  } 
	}

frame2.copyTo(Operator);

}



void holderOperator(cv::Mat im, cv::Mat &outim)
{

cv::Mat dst; 							
cv::GaussianBlur(im, dst, cv::Size(9,9), 1.0,1.0);            
dst=im-dst;                                       		
dst=dst*1.941;//0.041
cv::GaussianBlur(dst,dst, cv::Size(9,9), 1.0,1.0);            
dst=abs(dst);                                 			 
log(dst,dst);                                 			   
dst=abs(dst);                                  		
cv::GaussianBlur(dst,outim, cv::Size(9,9), 1.0,1.0);        

}

void holderDescriptor(cv::Mat imageInput1, vector<cv::KeyPoint> keypoints, vector<int> samplesInput, cv::Mat &descriptorOuput)
{

for (int i =0 ; i< keypoints.size(); i++)
{

Mat imageInput2(imageInput1(cv::Rect( keypoints[i].pt.x - int(keypoints[i].size/2), keypoints[i].pt.y - int(keypoints[i].size/2), keypoints[i].size, keypoints[i].size)));

Mat imageInput;
holderOperator(imageInput2,imageInput);

int ucenter=(imageInput.cols/2);
int vcenter=(imageInput.rows/2);

descriptorOuput.at<float>(i,0) = imageInput.at<float>(ucenter,vcenter);

int x, y, count=1, count1= 0;

for (double r=((imageInput.cols/(2*samplesInput.size()+1)));r <= imageInput.cols/2 ; r = r + ((imageInput.cols/(2*samplesInput.size()+1))))
    {

for(double teta=keypoints[i].angle; teta < keypoints[i].angle+(2*M_PI);teta=teta+((2*M_PI)/( samplesInput[count1]))) 
            {

                x=r*cos(teta);
                y=r*sin(teta);

		descriptorOuput.at<float>(i,count) = imageInput.at<float>(ucenter+x,vcenter-y);
		count++;

 	    }

count1++;
     }
   }
 }
}

