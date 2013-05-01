#ifndef DTIUTILS_H
#define DTIUTILS_H
#include <string>
#include <vector>
#include <set>
#include "GDTI.h"
#include "MultiTensor.h"
#include "cv.h"
//---Hierarchical clustering code ---
//void groupCoefficientsHC(GDTI &H, double *alpha, double *DBFDirections, double *Phi, int numDBFDirections, double b, double *diffusivities, std::vector<std::pair<std::pair<int, int>, double > > &F, MultiTensor &result);
//-----------------------------------
void showOrientationHistogram(double *H, int *groups, int n, double *directions, int rows, int cols, cv::Mat &M);
//void showOrientationSimilarityMatrix(double *H, int n, double *directions, GDTI &hardi, double transDiffusion, double longDiffusion, int rows, int cols, cv::Mat &M, cv::Mat &expM);
void showOrientationSimilarityMatrix(double *H, int n, double *directions, int rows, int cols, cv::Mat &M, cv::Mat &expM);
void buildNeighborhood(double *pdds, int npdds, int neighSize, std::vector<std::set<int> > &neighborhoods);
void buildNeighborhood(double *pdds, int npdds, double maxAngle, std::vector<std::set<int> > &neighborhoods);
void computeDiffusionFunction(double *dir, double _lambdaMin, double _lambdaMiddle, double _lambdaLong, double b, double *gradients, int numGradients, double *phi, int idxInc);
void computeDiffusionFunction(double *dir, double _lambdaScale, double _lambdaDifference, double b, double *gradients, int numGradients, double *phi, int idxInc);
void groupCoefficients(double *alpha, double *diffusionDirections, int numDirections, std::vector<std::set<int> > &neighborhoods, double *RES_pdds, double *RES_amount, int &RES_count);
void groupCoefficients(double *alpha, double *diffusionDirections, int numDirections, std::vector<std::set<int> > &neighborhoods, double bigPeaksThreshold, double transDiffusion, double longDiffusion, MultiTensor &result);
void groupCoefficients(double *_alpha, int nalphas, double threshold, std::vector<std::set<int> > &neighborhoods, std::vector<std::set<int> > &groups);
void createMultiTensorSingleComponent(double *S, GDTI &H, double *diffusionDirections, std::set<std::pair<double, int> > &group, MultiTensor &result);
void groupCoefficientsGDTI(GDTI &H, double longDiffusion, double transDiffusion, double *alpha, double *diffusionDirections, int numDirections, std::vector<std::set<int> > &neighborhoods, MultiTensor &result, std::vector<std::set<int> > &groups);
void getBigPeaks(double prop, double *RES_pdds, double *RES_amount, int &RES_count);
void computeCentroid(double *pdds, int npdds, double *alphas, const std::set<int> &cluster, double *centroid, double &amountDiff);
void loadRandomPDDs(const std::string &fname, double *&pdds, int &n);
int loadOrientations(const std::string &fname, double *&orientations, int &numOrientations, int *&s0Indices, int &numS0);
void save4DNifti(const std::string &fname, double *data, int nslices, int nrows, int ncols, int len);
void load4DNifti(const std::string &fname, double *&data, int &nslices, int &nrows, int &ncols, int &len);
void load4DNifti(const std::string &fname, float *&data, int &nslices, int &nrows, int &ncols, int &len);
void niiToPlain4D(const std::string &ifname, const std::string &ofname);
void saveDWINifti(const std::string &fname, double *s0, double *dw, int nslices, int nrows, int ncols, int len);
void save3DNifti(const std::string &fname, unsigned char *vol, int nslices, int nrows, int ncols);
void save3DNifti(const std::string &fname, double *vol, int nslices, int nrows, int ncols);
void loadDWMRIFiles(const std::vector<std::string> &names, int *s0Indices, int numS0, double *&S0Volume, double *&dwVolume, int &nr, int &nc, int &ns);
void loadDWMRIFromNifti(std::string fname, int *s0Indices, int numS0, double *&s0, double *&dwVolume, int &nrows, int &ncols, int &nslices, int &signalLength);
void loadVolumeFromNifti(std::string fname, double *&dwVolume, int &nrows, int &ncols, int &nslices);
void getDWSignalAtVoxel(double *S0, double *dwVolume, int nr, int nc, int ns, int numOrientations, int pr, int pc, int ps, double &s0, double *dwSignal);
void getMaximumConnectedComponentMask(double *data, int nr, int nc, int ns, unsigned char *mask);
void getMaximumConnectedComponent(unsigned char *binaryMask, int nr, int nc, int ns, unsigned char *mask);

double computeFractionalAnisotropy(double *eigenValues, double meanDiffusivity);
double computeFractionalAnisotropy(double *eigenValues);

double getMinimum(double *data, int n);
double getMaximum(double *data, int n);
double getMean(double *data, int n);
double getStdev(double *data, int n);
double getMinimum(double *data, int n, unsigned char *mask);
double getMaximum(double *data, int n, unsigned char *mask);
double getMean(double *data, int n, unsigned char *mask);
double getStdev(double *data, int n, unsigned char *mask);



#endif
