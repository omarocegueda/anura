#include "volume_io.h"
#include "image_io.h"
#include <math.h>
#include <cv.h>
#include <highgui.h>
void showDWVolumeSlices(const char *title, double *dwVolume, int nrows, int ncols, int nslices, int numGradients){
	int imagesWidth=sqrt(double (nslices*numGradients));
	int imagesHeight=(nslices*numGradients)/imagesWidth;
	cv::Mat globalInput;
	int width=(ncols+1)*imagesWidth+1;
	int height=(nrows+1)*imagesHeight+1;
	globalInput.create(height, width, CV_8UC1);
	unsigned char *gInputData=(unsigned char *)globalInput.data;
	for(int i=0;i<height*width;++i){
		gInputData[i]=127;
	}
	//---
	cv::Mat inputImg;
	inputImg.create(nrows,ncols,CV_8UC1);
	unsigned char *inputImg_data=inputImg.data;
	int imageIndex=0;
	double *img=new double[nrows*ncols];
	for(int k=0;k<numGradients;++k){
		int v=0;
		for(int s=0;s<nslices;++s, ++imageIndex){
			int imgPos_row=imageIndex/imagesWidth;
			int imgPos_col=imageIndex%imagesWidth;
			cv::Rect rect(1+imgPos_col*(ncols+1),1+imgPos_row*(nrows+1), ncols,nrows);
			cv::Mat input_roi=globalInput(rect);
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++v){
					img[i*ncols+j]=dwVolume[v*numGradients+k];
				}
			}
			showImg(img, nrows, ncols, inputImg_data);
			inputImg.copyTo(input_roi);
		}
	}
	delete[] img;
	cv::imshow(title,globalInput);
}

