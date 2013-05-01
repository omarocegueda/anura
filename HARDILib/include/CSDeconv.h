#ifndef CSDECONV_H
#define CSDECONV_H
#include <vector>
class CSDeconv{
	private:
		//----------
		int ngrads;//rows of the design matrix
		int nharm;//columns of the design matrix
		int lmax_data;//maximum SH order
		double *fconv;//design matrix (SH coefficients to amplitudes)
		double *rconv;//generalized inverse of fconv
		//----------
		int lmax_constraints;//maximum SH order that can be estimated with nConstraints
		int nConstraints;//Number of non-negativity constraints
		int nHarmConstraints;
		double *HR_trans;
		//----------

		double *HR_amps;
		int *neg;

		double lambda;
		double thresholdFraction;
		double thresholdValue;
		
		double *F;
		double *F_ant;
		double *init_F;
		double *S;
		double *S_padded;
	public:
		CSDeconv();
		~CSDeconv();
		void readConstraintDirections(const char *fname, double *&dirs, int &n);
		void readGradients(const char *fname, double *&dirs, double *&bs, int &n);
		void dellocate(void);
		/*
			directions must be given as consecutive [azimuth,elevation] pairs
		*/
		void init(const std::vector<double> &responseCoefs, const std::vector<double> &initFilter, double *directions, int _ngrads, double *constraintDirs, int ndirs, int lmax, double _threshold=0.1, double _lambda=1.0);
		
		/*
			Estimates the response function considering high anisotropy voxels only, aligning them so that their pdd is oriented 
			towards the z axis. Returns the number of samples used.
		*/
		int estimateResponseFunction(double *s0Volume, double *dwVolume, int nslices, int nrows, int ncols, unsigned char *mask, unsigned char filterMask, unsigned char filterVal, double *pddField, double *gradients, int numGradients, int lmax, double *coefs);
		
		/*
			Denoises the DWMRI volume by aligning neighboring signals and fitting SH to well aligned shapes. Finally it samples the fitted Spherical function
			along the given gradients
		*/
		void denoiseDWVolume_align(double *s0Volume, double *dwVolume, int nrows, int ncols, int nslices, int numGradients, double *gradientOrientations, double *&aligningErrors, int &nSamples, int &sampleLength);
		void setInputSignal(double s0, double *S, int len);
		bool iterate(void);
		double *getSHCoefficients(void);
		
};	
#endif
