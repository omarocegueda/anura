/**	Author: Omar Ocegueda
	Özarslan E. etal, (2003) Generalized Diffusion Tensor Imaging 
	and analytical relationships between diffusion tensor imaging 
	and high angular resolution diffusion imaging. Magn. Reson. Med.
*/
#ifndef GDTI_H
#define GDTI_H
#include <vector>
#include <string>
#define EPS_GDTI 1e-9
const unsigned char TT_MASK=3;
const unsigned char S0_BIT=1<<2;
const unsigned char FA_BIT=1<<3;
const unsigned char FA_HIST_BIT=1<<4;
const unsigned char AVSIGNAL_BIT=1<<5;
const unsigned char LCI_BIT=1<<6;


class GDTI{
	protected:
		double b;
		bool useNeighbors;
		int rank;
		int numCoefficients;
		int *coeffPositions;
		int *coeffMultiplicities;
		int *nxsyszs;
		double *M;//design matrix that defines the linear system to be solved to determine the tensor elements
		double *MtMinv;//the inverse of the product of M^t and M (for least squares)
		double *gradients;
		int numGradients;
	public:
		GDTI(int _rank, double _b, double *orientations, int _numOrientations);
		~GDTI();
		int getRank(void);
		int getNumCoefficients(void);
		void allocate(int _numGradients);
		void initialize(double b, double *orientations, int numOrientations);
		void dellocate(void);
		double solve(double S0, double *Si, double *tensor);
		void createMask(double *S0, double *DW, int nslices, int nrows, int ncols, double *fittedTensors, double *eigenvalues, double *pdds, unsigned char *mask, double thrFA, double thrS0, double thrAvSignal);
		void solveField(double *S0, double *DW, int nslices, int nrows, int ncols, double *fittedTensors, double *eigenvalues, double *pdds);
		int computeLinearPlanarSphericalCoeffs(double *lambda, double &linear, double &planar, double &spherical);
		int computeAverageProfile(double *eigenvalues, int nvoxels, unsigned char *mask, unsigned char filterMask, unsigned char filterVal, double *averageProfile);
		void setUseNeighbors(bool flag);
		double get_b(void);
		double *getGradients(void);
		int getNumGradients(void);
		
};

struct GDTI_params{
	int rank;
	double b;
	std::string orientationsName;
	std::vector<std::string> dwNames;
};

struct GDTI_output{
	int nr;
	int nc;
	int ns;
	unsigned char *mask;
	double *fractionalAnisotropy;
	double *pdd;
	double *meanDifusivity;
	double *lambdas;
	GDTI_output();
	~GDTI_output();

};
int fitGDTI(const GDTI_params &params, GDTI_output &output);

#endif
