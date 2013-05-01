#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H
#include <fstream>
#include <sstream>
#include <vector>
/**
	returns the number of discriminant directions computed
*/
int computeHighDimensionalLDA(double *data, int *labels, int n, int m, int numClasses, double **&w, double *&lambda, bool maxLikelihood=false);
int computeHighDimensionalLDA(double **data, int *labels, int n, int m, int numClasses, double **&w, double *&lambda, bool maxLikelihood=false);

template<class T> void projectData(T **data, int n, int m, double **w, int numDirections, double **projected){
	for(int i=0;i<n;++i){
		for(int k=0;k<numDirections;++k){
			double projection=0;
			for(int j=0;j<m;++j){
				projection+=data[i][j]*w[k][j];
			}
			projected[i][k]=projection;
		}
	}
}

template<class T> void projectData(T *data, int m, double **w, int numDirections, double *projected){
	for(int k=0;k<numDirections;++k){
		double projection=0;
		for(int j=0;j<m;++j){
			projection+=data[j]*w[k][j];
		}
		projected[k]=projection;
	}
}


void computeProjectedCentroids(double **projectedData, int *labels, int n, int numDirections, int numClasses, double **centroids);
void computeInvCov(double **projectedData, int *labels, int n, int numDirections, int numClasses, double **centroids, double **invCov, double *determinants);
void predictHDLDA(double **projectedData, int n, double **centroids, int numClasses,int numDirections, int *predicted, double *priors=NULL);
void predictHDLDA(double *projectedData, double **centroids, int numClasses,int numDirections, int &predicted);

int computeHDLDA_CV(double **_data, int *_labels, int n, int m, int numClasses, bool shuffle,
					 double **&w, double *&lambda, int trainingSize, int *predictions, 
					 double &inSampleError, double &outSampleError, bool evalWithROC=true);

int HDLDA_nFoldCV(double **_data, int *_labels, int *ids, int n, int m, int numClasses, int folds, 
				  double &inSampleError, double &outSampleError, double &inSampleDev, double &outSampleDev, double *&CM, double *&CMsd);
					 
void generateLDASummary(const char *fileName, double **data, int *labels, int n, int m, int numClasses, 
						double **w, double *lambda, int numDirections, std::vector<std::string> &labelDescription, std::vector<std::string> &names);
void generateClassROCCurves(const char *fileName, double **data, int *labels, int n, int m, int numClasses, double **testData, int *testLabels, int nTest,
							double **w, double *lambda, int numDirections, std::vector<std::string> &labelDescription, std::vector<std::vector<double> > &verificationThresholds);
void HDLDA_toyExample_CV(const char *fileName, int n, int m, int numClasses, double trainingProportion);
void HDLDA_toyExample(const char *fileName, int n, int m, int numClasses, bool subSample=false, bool libSVMFormat=false);
double computeVerificationRate(double *D, int *labels, int n, double falseAR);
double computeVerificationRate_AllPairs(double *D, int *labels, int n, double falseAR);
void HDLDA_labelGeneralization(const char *fileName, double trainingProportion, 
						  double &inSampleError, double &outSampleError);
/**
	Returns the TOTAL number of discriminant directions (min(C-1, m)) where C is the number of categories). 
	The minimum number of directions that minimize the estimated out-of-sample error is returned in kMax. The
	sampleUsage array contains information about how each sample was used: training (0) or testing (1). The 
	in-sample error and estimated out-of-sample error are returned in inSampleError and outSampleError respectively.
	The discriminant directions are returned in w and their corresponding eigenvalues in lambda.
*/
int HDLDA_labelGeneralization(double **data, int *labels, int n, int m, int numClasses, double trainingProportion, 
						  double **&w, double *&lambda, double &inSampleError, double &outSampleError, int &kMax, int *sampleUsage, bool verbose=true);
double validateLDA(double **w, int rank, double **data, int *labels, int n, int m);
#endif
