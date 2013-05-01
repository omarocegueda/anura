#include "image_io.h"
#include "macros.h"
#include "utilities.h"
#include "statisticsutils.h"
#include <highgui.h>
using namespace cv;

void evalClusterPalette(int k, double &r, double &g, double &b){
	int kk=k;
	r=((kk&1)!=0)*0.5;
	g=((kk&2)!=0)*0.5;
	b=((kk&4)!=0)*0.5;
	if(k%2){
		r*=2;
		g*=2;
		b*=2;
	}
}

void evalClusterPalette(int k, int &r, int &g, int &b){
	int kk=k;
	r=((kk&1)!=0)*127;
	g=((kk&2)!=0)*127;
	b=((kk&4)!=0)*127;
	if((k%2)==0){
		r*=2;
		g*=2;
		b*=2;
	}
}

void showBand(double *d_img, int rows, int cols, int numBands, int idx, unsigned char *c_img){
	double minVal=getMinVal_offset(d_img, rows*cols*numBands, numBands, idx);
	double maxVal=getMaxVal_offset(d_img, rows*cols*numBands, numBands, idx);
	double diff=maxVal-minVal;
	if(diff<1e-9){
		memset(c_img, 0, sizeof(unsigned char)*rows*cols);
		return;
	}
	for(int i=0;i<rows;++i){
		for(int j=0;j<cols;++j){
			unsigned char val=(unsigned char)(255*(d_img[numBands*(i*cols+j)+idx]-minVal)/diff);
			c_img[i*cols+j]=val;
		}
	}
}

double getMinimum_local(double *data, int n){
	double best=*data;
	for(double *current=data+n-1;current!=data;--current){
		if((*current)<best){
			best=*current;
		}
	}
	return best;
}

double getMaximum_local(double *data, int n){
	double best=*data;
	for(double *current=data+n-1;current!=data;--current){
		if((*current)>best){
			best=*current;
		}
	}
	return best;
}

void showImg(double *d_img, int rows, int cols, unsigned char *c_img, bool color){
	double minVal=getMinimum_local(d_img, rows*cols);
	double maxVal=getMaximum_local(d_img, rows*cols);
	double diff=maxVal-minVal;
	if(diff<1e-9){
		memset(c_img, 0, sizeof(unsigned char)*rows*cols);
		return;
	}
	if(color){
		for(int i=0;i<rows;++i){
			for(int j=0;j<cols;++j){
				unsigned char val=(unsigned char)(255*(d_img[i*cols+j]-minVal)/diff);
				unsigned char r,g,b;
				getIntensityColor<unsigned char>(val, r, g, b);
				c_img[3*(i*cols+j)]=b;
				c_img[3*(i*cols+j)+1]=g;
				c_img[3*(i*cols+j)+2]=r;
			}
		}
	}else{
		for(int i=0;i<rows;++i){
			for(int j=0;j<cols;++j){
				unsigned char val=(unsigned char)(255*(d_img[i*cols+j]-minVal)/diff);
				c_img[i*cols+j]=val;
			}
		}
	}
}

void showHorizontalLattice(double *zh, int rows, int cols, unsigned char *c_img){
	double minVal=zh[0];
	double maxVal=zh[0];
	for(int i=0;i<rows;++i){
		for(int j=0;j<cols-1;++j){
			minVal=MIN(minVal, zh[i*cols+j]);
			maxVal=MAX(maxVal, zh[i*cols+j]);
		}
	}
	double diff=maxVal-minVal;
	if(diff<1e-9){
		memset(c_img, 0, sizeof(unsigned char)*rows*cols);
		return;
	}
	for(int i=0;i<rows;++i){
		for(int j=0;j<cols-1;++j){
			c_img[i*cols+j]=(unsigned char)(255*(zh[i*cols+j]-minVal)/diff);
		}
	}
}

void showVerticalLattice(double *zv, int rows, int cols, unsigned char *c_img){
	double minVal=zv[0];
	double maxVal=zv[0];
	for(int i=0;i<rows-1;++i){
		for(int j=0;j<cols;++j){
			minVal=MIN(minVal, zv[i*cols+j]);
			maxVal=MAX(maxVal, zv[i*cols+j]);
		}
	}
	double diff=maxVal-minVal;
	if(diff<1e-9){
		memset(c_img, 0, sizeof(unsigned char)*rows*cols);
		return;
	}
	for(int i=0;i<rows-1;++i){
		for(int j=0;j<cols;++j){
			c_img[i*cols+j]=(unsigned char)(255*(zv[i*cols+j]-minVal)/diff);
		}
	}
}

void showPointCloud(double *x, int n, Mat &img, int w, int h, int r, int g, int b){
	double minX=x[0], maxX=x[0];
	double minY=x[1], maxY=x[1];
	for(int i=1;i<n;++i){
		minX=MIN(minX, x[2*i]);
		maxX=MAX(maxX, x[2*i]);
		minY=MIN(minY, x[2*i+1]);
		maxY=MAX(maxY, x[2*i+1]);
	}
	double dx=maxX-minX;
	double dy=maxY-minY;
	double scale=MAX(dx,dy);
	img.create(h,w,CV_8UC3);
	for(int i=0;i<n;++i){
		int px=(w-1)*(x[2*i]-minX)/scale;
		int py=(h-1)*(x[2*i+1]-minY)/scale;
		cv::Point center(px,h-1-py);
		cv::Size axes(3,3);
		cv::Scalar color(r,g,b);
		cv::ellipse(img, center,axes, 0, 0, 360, color);
	}
	
	
}

void showColoredPointCloud(double *x, int n, int dim, cv::Mat &img, int w, int h, int *rgbColors){
	double minX=x[0], maxX=x[0];
	double minY=x[1], maxY=x[1];
	for(int i=1;i<n;++i){
		minX=MIN(minX, x[dim*i]);
		maxX=MAX(maxX, x[dim*i]);
		minY=MIN(minY, x[dim*i+1]);
		maxY=MAX(maxY, x[dim*i+1]);
	}
	double dx=maxX-minX;
	double dy=maxY-minY;
	double scale=MAX(dx,dy);
	img.create(h,w,CV_8UC3);
	for(int i=0;i<n;++i){
		int px=(w-1)*(x[dim*i]-minX)/scale;
		int py=(h-1)*(x[dim*i+1]-minY)/scale;
		cv::Point center(px,h-1-py);
		cv::Size axes(3,3);
		if(rgbColors==NULL){
			cv::Scalar color(0,0,0);
			cv::ellipse(img, center,axes, 0, 0, 360, color);
		}else{
			cv::Scalar color(rgbColors[3*i],rgbColors[3*i+1],rgbColors[3*i+2]);
			cv::ellipse(img, center,axes, 0, 0, 360, color);
		}
	}
}

void showClassifiedPointCloud(double *x, int n, int dim, cv::Mat &img, int w, int h, int *labels){
	double minX=x[0], maxX=x[0];
	double minY=x[1], maxY=x[1];
	for(int i=1;i<n;++i){
		minX=MIN(minX, x[dim*i]);
		maxX=MAX(maxX, x[dim*i]);
		minY=MIN(minY, x[dim*i+1]);
		maxY=MAX(maxY, x[dim*i+1]);
	}
	double dx=maxX-minX;
	double dy=maxY-minY;
	double scale=MAX(dx,dy);

	int margin=6;
	
	
	
	img.create(h,w,CV_8UC3);
	//rectangle(img, cv::Point(0,0), cv::Point(w-1, h-1), cv::Scalar(127,127,127), CV_FILLED);
	rectangle(img, cv::Point(0,0), cv::Point(w-1, h-1), cv::Scalar(255,255,255), CV_FILLED);

	w-=2*margin;
	h-=2*margin;
	for(int i=0;i<n;++i){
		int px=(w-1)*(x[dim*i]-minX)/scale;
		int py=(h-1)*(x[dim*i+1]-minY)/scale;
		cv::Point center(margin+px,margin+h-1-py);
		cv::Size axes(3,3);
		if(labels==NULL){
			cv::Scalar color(0,0,0);
			cv::ellipse(img, center,axes, 0, 0, 360, color);
		}else if(labels[i]>=0){
			int r,g,b;
			evalClusterPalette(labels[i], r,g,b);
			cv::Scalar color(b,g,r);
			cv::ellipse(img, center,axes, 0, 0, 360, color);
		}
	}
}

#ifdef USE_VTK

#include "vtkSphereSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkActor.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkRenderWindowInteractor.h" 

void showClassifiedPointCloud3D(double *x, int n, int dim, double proportion, vtkRenderer *renderer, int w, int h, int *labels){
	renderer->RemoveAllViewProps();
	double minX=x[0], maxX=x[0];
	double minY=x[1], maxY=x[1];
	for(int i=1;i<n;++i){
		minX=MIN(minX, x[dim*i]);
		maxX=MAX(maxX, x[dim*i]);
		minY=MIN(minY, x[dim*i+1]);
		maxY=MAX(maxY, x[dim*i+1]);
	}
	double dx=maxX-minX;
	double dy=maxY-minY;
	double scale=MIN(dx,dy)/sqrt(double(n));
	// create sphere geometry
	vtkSphereSource *sphere = vtkSphereSource::New();
	sphere->SetRadius(1.0);
	sphere->SetThetaResolution(18);
	sphere->SetPhiResolution(18);

	// map to graphics library
	vtkPolyDataMapper *map = vtkPolyDataMapper::New();
	map->SetInput(sphere->GetOutput());
	//add points
	for(int i=0;i<n;++i){
		double rs=uniform(0,1);
		if(rs>proportion){
			continue;
		}
		if(labels[i]<0){
			continue;
		}
		vtkActor *aSphere = vtkActor::New();
		aSphere->SetMapper(map);
		//aSphere->SetPosition(&x[dim*i]); // sphere color blue
		double pos_x=x[dim*i];
		double pos_y=0;
		double pos_z=0;
		if(dim>1){
			pos_y=x[dim*i+1];
		}
		if(dim>2){
			pos_z=x[dim*i+2];
		}
		aSphere->SetPosition(pos_x, pos_y, pos_z);
		aSphere->SetScale(scale, scale, scale);
		double r,g,b;
		evalClusterPalette(labels[i], r,g,b);
		aSphere->GetProperty()->SetColor(r,g,b);
		//aSphere->GetProperty()->SetBackfaceCulling()
		renderer->AddActor(aSphere);
		aSphere->Delete();
	}
	renderer->ResetCamera();
	map->Delete();
	sphere->Delete();
}

void showVectorField2D_orientation(double *fx, double *fy, int nrows, int ncols, cv::Mat &img, int w, int h){
}

void showVectorField2D_magnitude(double *fx, double *fy, int nrows, int ncols, cv::Mat &img, int w, int h){
}


#endif