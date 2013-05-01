#ifndef SPHERICALHARMONICS_H
#define SPHERICALHARMONICS_H
#include <math.h>
#include <vector>
namespace SphericalHarmonics{
	const double EPSILON=1e-12;
	#ifdef WIN32
		#ifndef NAN
			static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
			#define NAN (*(const float *) SphericalHarmonics::__nan)
		#endif
	#endif
	/*
		Number of SH coefficients up to order lmax
	*/
	inline int NforL (int lmax) { return ((lmax+1)*(lmax+2)/2); }

	/*
		Consecutive indexing of the SH coefficients
	*/
	inline int index (int l, int m) { return (l*(l+1)/2 + m); }

	/*
		Maximum SH order that can be estimated with N samples
	*/
	inline int LforN (int N) { return (2*(((int) (sqrt((float) (1+8*N)))-3)/4)); }

	/*
		directions is a nrowsx2 matrix. Each direction is given by two angles [azimuth elevation]
		sht will contain the spherical harmonics transform.
	*/
	void initTransformation(double *directions, int nrows, int lmax, double *sht);

	void getAzimuthElevationPairs(double *gradients, int ngrads, double *directions);
	void getAmplitudeAndAzimuthElevationPairs(double *gradients, int ngrads, double *directions, double *amplitudes);
	void SH2RH(const std::vector<double> &SH, std::vector<double> &RH);
}

#endif
