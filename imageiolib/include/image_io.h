#ifndef IMAGE_IO_H
#define IMAGE_IO_H
#include <highgui.h>

#ifdef USE_VTK
	#include "vtkRenderer.h"
#endif

void evalClusterPalette(int k, double &r, double &g, double &b);
void evalClusterPalette(int k, int &r, int &g, int &b);

void showBand(double *d_img, int rows, int cols, int numBands, int idx, unsigned char *c_img);
void showImg(double *d_img, int rows, int cols, unsigned char *c_img, bool color=false);
void showHorizontalLattice(double *zh, int rows, int cols, unsigned char *c_img);
void showVerticalLattice(double *zv, int rows, int cols, unsigned char *c_img);

void showPointCloud(double *x, int n, cv::Mat &img, int w, int h, int r=0, int g=0, int b=0);
void showColoredPointCloud(double *x, int n, int dim, cv::Mat &img, int w, int h, int *rgbColors=NULL);
void showClassifiedPointCloud(double *x, int n, int dim, cv::Mat &img, int w, int h, int *labels=NULL);
void showVectorField2D_orientation(double *fx, double *fy, int nrows, int ncols, cv::Mat &img, int w, int h);
void showVectorField2D_magnitude(double *fx, double *fy, int nrows, int ncols, cv::Mat &img, int w, int h);

#ifdef USE_VTK
void showClassifiedPointCloud3D(double *x, int n, int dim, double proportion, vtkRenderer *renderer, int w, int h, int *labels=NULL);
#endif

#endif
