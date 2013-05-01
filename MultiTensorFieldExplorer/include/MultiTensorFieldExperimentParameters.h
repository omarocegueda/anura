#ifndef MULTITENSORFIELDEXPERIMENTPARAMETERS
#define MULTITENSORFIELDEXPERIMENTPARAMETERS
#include <string>
#include "MultiTensorField.h"
#include "DBF.h"
#include "ConfigManager.h"
#include "macros.h"


#define DEFAULT_DBF_FILE_NAME "DBF_129orientations.dat"

enum MultiTensorFieldInputType{MTFIT_GTGenerator, MTFIT_DWMRI_FILES, MTFIT_NIFTI};
enum MultiTensorFieldGroundTruthType{MTFGTT_None, MTFGTT_GTGenerator, MTFGTT_NIFTI};
class MultiTensorFieldExperimentParameters{
	public:
		bool loaded;
		MultiTensorField *reconstructed;
		//===========configuration===============
		//-----masking----------
		bool loadMask;
		bool saveMask;
		bool saveODFField;
		bool WM_useS0HistogramThresholding;
		double WM_S0Threshold;
		double WM_FAThreshold;
		double WM_AVSignalThreshold;
		int WM_erosionSteps;
		char WM_maskFname[MAX_STR_LEN];

		bool RF_useS0HistogramThresholding;
		bool RF_useFAHistogramThresholding;
		double RF_S0Threshold;
		double RF_FAThreshold;
		char RF_maskFname[MAX_STR_LEN];
		//-----preprocessing----
		bool useS0MedianFilter;
		bool estimateDiffusivities;
		double longDiffusion;
		double transDiffusion;
		//----------------------
		MultiTensorFieldInputType inputType;
		MultiTensorFieldGroundTruthType groundTruthType;
		char outputDir[MAX_STR_LEN];
		char reconstructionFname[MAX_STR_LEN];
		char gtFname[MAX_STR_LEN];
		char solutionFname[MAX_STR_LEN];
		char solutionListFname[MAX_STR_LEN];
		char gradientDirsFname[MAX_STR_LEN];
		char basisDirsFname[MAX_STR_LEN];
		char ODFDirFname[MAX_STR_LEN];
		double b;
		bool denoisingTest;
		bool fitWithNeighbors;
		bool fitGDTIOnly;
		bool useFAIndicator;
		bool useGDTIPostProcessing;
		bool applyTensorSplitting;
		bool iterativeBasisUpdate;
		bool iterateDiffProf;
		bool evaluate;
		bool evaluateList;
		bool nonparametric;
		int clusteringNeighSize;
		int numSamplings;
		double bigPeaksThreshold;
		bool regularizeAlphaField;
		
		int SNR;
		DBFSolverType solverType;

		char nTensFname[MAX_STR_LEN];
		char sizeCompartmentFname[MAX_STR_LEN];
		char fPDDFname[MAX_STR_LEN];

		char dwListFname[MAX_STR_LEN];
		char rawDataBaseDir[MAX_STR_LEN];

		char dwFname[MAX_STR_LEN];
		char diffUnits[MAX_STR_LEN];
		//----loadable data----
		double *ODFDirections;
		int numODFDirections;
		MultiTensorField *GT;

		double *dwVolume;
		double *s0Volume;

		double *gradients;
		int numGradients;

		double *DBFDirections;
		int numDBFDirections;

		double lambda;//spatial smoothing parameter
		double alpha0;//1st order TGV penalty
		double alpha1;//2st order TGV penalty
		bool spatialSmoothing;
		//---------------------
		

		void setReconstructed(MultiTensorField *_reconstructed);
		void dellocate(void);
		int loadData(void);
		MultiTensorFieldExperimentParameters();
		~MultiTensorFieldExperimentParameters();
		void printConfig(void);
		void configFromFile(const char *fname);
};
std::vector<double> testMultiTensorField(MultiTensorFieldExperimentParameters &params, RegularizationCallbackFunction callbackFunction, void *callbackData);
void common_evaluate(MultiTensorField &GT, MultiTensorField &recovered);
void common_evaluate(MultiTensorField &GT, MultiTensorField &recovered, FILE *FNegative, FILE *FPositive, FILE *FAngular);
int denoiseDWVolume(double *dwVolume, int nrows, int ncols, int nslices, int numGradients, double lambda, double *denoisingParams, unsigned char *mask=NULL);
void testDenoisingMethod(MultiTensorFieldExperimentParameters &params);
#endif
