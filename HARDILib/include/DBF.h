#ifndef DBF_H
#define DBF_H
#include <vector>
#include <string>
#include "utilities.h"
#include "MultiTensorField.h"
#define EPS_DBF (1e-9)
typedef void (*RegularizationCallbackFunction)(void*);
enum DBFSolverType{DBFS_NNLS, DBFS_LARS, DBFS_NNSLS, DBFS_PGS};
class DBF{
	protected:
		double b;
		double longitudinalDiffusion;
		double transversalDiffusion;
		double *diffusionBasis;
		double *diffusionDirections;
		double *gradients;
		int numBasisFunctions;
		int numGradients;
		void createDiffusionBasisFunctions(double *orientations, int numOrientations, double *basisDirections, int numBasisDirections);
		DBFSolverType solverType;
	public:
		DBF(double *_gradients, int _numGradients, double *_basisDirections, int _numBasisDirections);
		DBF(double _b, double _longitudinalDiffusion, double _transversalDiffusion, double *_gradients, int _numGradients, double *_basisDirections, int _numBasisDirections);
		~DBF();
		void setDiffusionBasis(double *Phi);
		double solve(double *Si, double *alphas, double lprop=-1, bool NNOLS_fit=true, double *s0=NULL);
		int getNumBasisFunctions(void);
		double *getDiffusionBasis(void);
		double *getDiffusionDirections(void);
		double *getGradients(void);
		int getNumGradients(void);
		void setSolverType(DBFSolverType _solverType);
		void reComputeDiffusionBasisFunctions(double _b, double _longitudinalDiffusion, double _transversalDiffusion);
		void computeDiffusionFunction(double *dir, double _lambdaMin, double _lambdaMiddle, double _lambdaLong, double *phi, int idxInc);
};



struct DBF_params{
	int nr;
	int nc;
	int ns;
	double b;
	double lProp;//maximum proportion of |alpha| to be kept
	unsigned char *mask;
	std::string orientationsFileName;
	std::string dbfDirectionsFileName;
	std::vector<std::string> dwNames;
	void setDefault(void);
	DBF_params();
	~DBF_params();
};

struct DBF_output{
	double *pdds;
	double *sizeCompartments;
	double *alphas;
	double *error;
	int *nzCount;
	int numBasisDirections;
	int ns;
	int nr;
	int nc;
	int maxDirections;
	DBF_output();
	~DBF_output();
};

void loadDiffusionDirections(const std::string fname, double *&directions, int &numDirections);
int regularizeDBF(double *alpha, double lambda, double mu, 
				  double *orientations, int numOrientations,
				  double *basisDirections, int numBasisDirections,
				  double *dwVolume, double *s0Volume, int ns, int nr, int nc, 
				  double bParam, double longDiffusion, double transDiffusion,
				  RegularizationCallbackFunction callback, void*callbackParam);
int fitDBF(const DBF_params &params, DBF_output &output);
int regularizeDBF(const DBF_params &params, double lambda, double mu, DBF_output &output, RegularizationCallbackFunction callback, void *callbackParam);

int regularizeDBF_alram(double *alphaVolume, MultiTensorField &reconstructed, double longDiffusion, double transDiffusion, double *HRT_coeffs,
					DBF &dbfInstance, double *dwVolume, double *s0Volume, int ns, int nr, int nc, 
					RegularizationCallbackFunction callback, void*callbackParam);
#endif
