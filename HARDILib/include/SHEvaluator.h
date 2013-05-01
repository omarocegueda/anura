#ifdef USE_GSL
#ifndef SHEVALUATOR_H
#define SHEVALUATOR_H
//--Addapted from MRTrix--
//--Evaluates a spherical function and its derivatives (given in terms of its SH representation) along a set of pre-defined orientations--
class SHEvaluator{
	protected:
		double *rows;
		/*	Structure of each row:
			____________________________________________________________________
			|  dx  |  dy  |  dz  |  r  |  ...  |  daz  |  ...  |  del  |  ...  |
			--------------------------------------------------------------------
			dx, dy, dz is the normalized vector along which the spherical function is evaluated
			r points to the beginning of the radii basis functions. daz points to the beginning of the derivatives wrt azimuthal angle,
			and del points to the beginning of the derivatives wrt elevation angle. Each section (r, daz and del) contains nsh_computed coefficient
			for a total length of 3*(nsh_computed+1) coefficients.
		*/
		int numDirections;
		int lmax_computed;
		int nsh_computed;
		int rowLength;
		void precomputeRow(double* row);
		double *getRow(int n);
		double *getRadiiPointer(int n);
		double *getDazPointer(int n);
		double *getDelPointer(int n);
	public:
		/*
			Initializes the evaluator with numDirections predefined evaluation orientations given by the normalized 3D vectors "directions", 
			for a maximum Spherical Harmonic order "lmax"
		*/
		SHEvaluator(double *directions, int _numDirections, int lmax);

		/*
			Returns in vertices the position of the vertices (along the predefined directions) of the spherical function whose SH representation is
			given by the SHCoeffs (SHCoeffs must be an array of (lmax+1)*(lmax+2)/2 coefficients)
		*/
		int evaluateFunction_vertices(double *SHCoeffs, int numCoeffs, int lmax, double *vertices);

		int evaluateFunction_amplitudes(double *SHCoeffs, int numCoeffs, int lmax, double *amplitudes);

		/*
			Returns in normals the normal vectors (at the predefined sphere positions given by rows) of the spherical function whose SH representation is
			given by the SHCoeffs (SHCoeffs must be an array of (lmax+1)*(lmax+2)/2 coefficients)
		*/
		int evaluateFunctionAndNormals(double *SHCoeffs, int numCoeffs, int lmax, double *vertices, double *normals);
		~SHEvaluator();
};
#endif
#endif
