#ifndef DENOISING_H
#define DENOISING_H
#include "TotalVariation.h"
#include <string.h>
enum EDenoisingType{EDT_GEMAN_MCCLURE, EDT_TUKEY_BIWEIGHT};

int averageVolumeDenoising(double *img, int r, int c, int s, double lambda, double *params, double *f, unsigned char *mask=NULL);
int selectiveAverageVolumeDenoising(double *img, int r, int c, int s, double *denoised, unsigned char *mask, double *neighFeature, double neighSimThr);
int medianVolumeDenoising(double *img, int r, int c, int s, double lambda, double *params, double *f);

/*
Robust Denoising methods with explicit line and outlier processes based on:
M. Black and A. Rangarajan, “On the unification of line processes,
outlier rejection, and robust statistics with applications in early
vision,” Int’l J. Computer Vision, vol. 19, no. 1, pp. 57–92, 1996.
*/
//-----2D-----
int robustDenoising(double *img, int r, int c, double lambda, EDenoisingType denoisingType, double *params, double *f, double *Z);
int robustDenoisingOR(double *img, int r, int c, double lambda, EDenoisingType denoisingType, double *params, double *f, double *Z, double *M);
//-----3D-----
/**
	Scalar-valued volume denoising with explicit line process
*/
int robustVolumeDenoising(double *img, int r, int c, int s, double lambda, EDenoisingType denoisingType, double *params, double *f);
/**
	Scalar-valued volume denoising with explicit line and outlier processes
*/
int robustVolumeDenoisingOR(double *img, int r, int c, int s, double lambda, EDenoisingType denoisingType, double *params, double *f);

/**
	Vector-valued volume denoising with explicit line and outlier processes
*/
int robust3DVectorFieldDenoisingOR(double *img, int nrows, int ncols, int nslices, int dim, double lambda, EDenoisingType denoisingType, double *params, double *f, double *lineProc);
//=============================================================================
/*
Total Generalized Variation minimization: 
Kristian Bredies, Karl Kunisch, Thomas Pock, "Total Generalized Variation"SIAM J. Imaging Sci., 3(3), 492–526. (35 pages) doi: 10.1137/090769521 

Using a primal-dual optimization method :
Antonin Chambolle and Thomas Pock, "A first-order primal-dual algorithm for convex problems with applications to imaging"
Journal of Mathematical Imaging and Vision archive Volume 40 Issue 1, May 2011 Pages 120-145 
*/

/**
	Scalar-Volume denoising, using TGV-L2, slice by slice.
*/
int volumeDenoising_TGV(double *img, int nrows, int ncols, int nslices, double lambda, double *params, double *f);
//-----3D vector field denoising-----

#endif
